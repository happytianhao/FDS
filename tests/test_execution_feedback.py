"""Check measured endpoint provenance and recovery without a simulator or API."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_decision_recovery import Environment, Planner, good

from fds.common.recovery import RecoveringPlanner, RecoveryConfig
from fds.common.retry import TransportError
from fds.task_harness.runtime import TaskHarness
from fds.types import Frame, Observation, StepResult


class MovingEnvironment(Environment):
    def observation(self):
        state = np.zeros(7)
        state[0] = self.steps
        return Observation("Move and inspect", [Frame(
            self.steps, {"front": np.zeros((4, 4, 3), dtype=np.uint8)}, state,
        )])

    def get_kinematics(self):
        def evaluate(q):
            pose = np.eye(4)
            pose[0, 3] = q[0]
            return pose, None

        return SimpleNamespace(
            metadata={"base_link": "base"}, evaluate=evaluate,
            pose=lambda q: {"position_m": [float(q[0]), 0, 0],
                            "rotation_matrix": np.eye(3).tolist()},
        )

    def step(self, action):
        self.steps += 1
        return StepResult(self.observation(), self.steps >= self.done_after, False, "ongoing")


@pytest.mark.parametrize("stop_after,expected", [(20, [2, 4]), (3, [2])])
def test_records_only_measured_original_endpoints(tmp_path, monkeypatch, stop_after, expected):
    from fds.task_harness.actions import TaskActions

    def prepare(self, tool, observation, cues, budget, log):
        solutions = []
        for x in (1.5, 2.5, 4.25):
            pose = np.eye(4)
            pose[0, 3] = x
            solutions.append({"target_pose": pose.tolist()})
        return [np.zeros(7)] * 4, {
            "requested_targets": tool["arguments"]["targets"],
            "ik_solutions": solutions,
            "resolved_targets": [
                {"steps": 1, "requested_target_index": 0, "subtarget_index": 0,
                 "subtarget_count": 2},
                {"steps": 1, "requested_target_index": 0, "subtarget_index": 1,
                 "subtarget_count": 2},
                {"steps": 2, "requested_target_index": 1, "subtarget_index": 0,
                 "subtarget_count": 1},
            ],
        }

    monkeypatch.setattr(TaskActions, "prepare", prepare)

    def respond(context, call):
        value = good(context)
        if call == 1:
            value["tool_batch"] = {"kind": "environment", "subgoal": "Move",
                "observation_steps": [], "tool_calls": [{"name": "move_ee", "arguments": {
                    "targets": [{"position_m": [x, 0, 0], "quaternion_wxyz": [1, 0, 0, 0],
                                 "gripper": "hold", "steps": 2} for x in (2.5, 4.25)],
                }}]}
        elif call == 2:
            value["tool_batch"] = {"kind": "auxiliary", "tool_calls": [{
                "name": "review_frames", "arguments": {
                    "window_id": context["observation"]["review_window"]["window_id"],
                    "frame_ids": [context["current_frame"]],
                },
            }]}
        else:
            value["tool_batch"] = {"kind": "environment", "subgoal": "Stop",
                "observation_steps": [], "tool_calls": [{
                    "name": "finish", "arguments": {"reason": "Fixture complete"},
                }]}
        return value

    env, planner = MovingEnvironment(stop_after), Planner(respond)
    out = tmp_path / "episode"
    TaskHarness(env, planner, max_calls=3, max_tool_steps=8, max_steps=20).run(0, out)
    receipt = json.loads((out / "audit/call_0000/outcome.json").read_text())["feedback"][0]
    points = receipt["target_observations"]
    assert [p["tool_step_offset"] for p in points] == expected
    assert [p["frame"] for p in points] == expected
    assert [p["requested_target_index"] for p in points] == list(range(len(expected)))
    assert points[0]["measured_position_error_m"] == pytest.approx(.5)
    assert points[0]["measured_tcp_pose"]["position_m"] == [2, 0, 0]
    assert all(p["semantic_success"] == "unverified" for p in points)
    if stop_after == 20:
        assert points[1]["measured_position_error_m"] == pytest.approx(.25)
        assert planner.inputs[1]["recent_motion_history"] == planner.inputs[2]["recent_motion_history"]
        assert env.steps == 4
    else:
        assert not receipt["tracking"]["target_fully_commanded"]


@pytest.mark.parametrize("succeed", [True, False])
def test_transport_retries_hold_state_and_stop_at_limit(tmp_path, succeed):
    env = Environment()
    calls = []

    def decide(context, images, directory):
        directory.mkdir()
        calls.append((context, env.steps))
        if not succeed or len(calls) == 1:
            raise TransportError("stream disconnected before completion")
        return "valid decision"

    planner = SimpleNamespace(decide=decide, model="fixture", effort="medium")
    wrapped = RecoveringPlanner(planner, RecoveryConfig(
        transport_retries=2, retry_delay_seconds=0, retry_max_delay_seconds=0, pause_seconds=0,
    ))
    from fds.task_harness.contracts import definitions
    context = {"decision_protocol": "task_harness", "contract_mode": "off",
               "tool_definitions": definitions(8, False, 4), "max_tool_steps": 8}
    if succeed:
        assert wrapped.decide(context, [], tmp_path / "call") == "valid decision"
        assert len(calls) == 2
    else:
        with pytest.raises(TransportError):
            wrapped.decide(context, [], tmp_path / "call")
        assert len(calls) == 3
    assert all(c == (context, 0) for c in calls)
    journal = json.loads((tmp_path / "call/retry.json").read_text())
    assert not journal["environment_advanced_during_retry"]
    assert journal["attempts"][0]["usage_known"] is False
