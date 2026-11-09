import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from fds.cli import episode_timing
from fds.common.recovery import RecoveringPlanner, RecoveryConfig
from fds.decision import Decision
from fds.task_harness.protocol import DecisionValidationError, mock_decision, schema_for
from fds.task_harness.runtime import TaskHarness
from fds.types import ActionSpec, Frame, Observation, StepResult


class Environment:
    action_spec = ActionSpec(-np.ones(7), np.ones(7))

    def __init__(self, done_after=1):
        self.steps, self.done_after, self.closed = 0, done_after, False

    def observation(self):
        frame = Frame(self.steps, {"front": np.zeros((4, 4, 3), dtype=np.uint8)}, np.zeros(7))
        return Observation("Wait for the environment", [frame])

    def reset(self, episode):
        return self.observation()

    def get_kinematics(self):
        return SimpleNamespace(
            metadata={"base_link": "base"},
            pose=lambda q: {"position_m": [0, 0, 0], "rotation_matrix": np.eye(3).tolist()},
        )

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.done_after
        return StepResult(self.observation(), done, False, "success" if done else "ongoing")

    def close(self):
        self.closed = True


def good(context):
    value = asdict(mock_decision(context))
    if context["contract_mode"] == "off":
        value.pop("fds_contract")
    return value


def invalid(context, kind="unresolved"):
    value = good(context)
    if kind == "unresolved":
        value["fds_contract"]["required_evidence"] = [
            {"id": "location", "purpose": "Need coordinates", "status": "unresolved", "evidence_refs": []}
        ]
    elif kind == "undefined":
        value["fds_contract"]["candidates"][0]["requires"] = ["missing"]
    elif kind == "provenance":
        value["fds_contract"]["required_evidence"] = [
            {"id": "location", "purpose": "Need coordinates", "status": "resolved", "evidence_refs": [999]}
        ]
    elif kind == "schema":
        value["tool_batch"]["observation_steps"] = "invalid"
    elif kind == "batch":
        value["tool_batch"]["observation_steps"] = [1, 1]
    return value


class Planner:
    model, effort = "fixture", "medium"

    def __init__(self, respond):
        self.respond, self.inputs, self.images = respond, [], []

    def decide(self, context, images, directory):
        self.inputs.append(deepcopy(context))
        self.images.append([p.read_bytes() for p in images])
        directory.mkdir(parents=True)
        value = self.respond(context, len(self.inputs))
        raw = value if isinstance(value, str) else json.dumps(value)
        (directory / "response.json").write_text(raw)
        (directory / "stats.json").write_text(json.dumps({
            "usage": {"input_tokens": 10, "output_tokens": 3}, "seconds": 0,
        }))
        return Decision.parse_json(raw, schema_for(context))


def run(tmp_path, respond, *, max_calls=3, corrections=2, mode="full", done_after=1):
    env, planner = Environment(done_after), Planner(respond)
    wrapped = RecoveringPlanner(planner, RecoveryConfig(pause_seconds=0))
    harness = TaskHarness(
        env, wrapped, max_calls=max_calls, max_steps=10, max_tool_steps=8,
        options={"max_decision_corrections": corrections}, fds_options={"contract_mode": mode},
    )
    output = tmp_path / "episode"
    summary = harness.run(0, output)
    return env, planner, output, summary


@pytest.mark.parametrize("kind", ["unresolved", "undefined", "provenance", "schema", "batch", "json"])
def test_rejection_preserves_state_then_correction_executes(tmp_path, kind):
    def respond(context, call):
        if call == 1:
            value = good(context)
            value["tool_batch"] = {"kind": "auxiliary", "tool_calls": [{
                "name": "review_frames", "arguments": {"window_id": "initial", "frame_ids": [0]},
            }]}
            value["memory_changes"]["experiences"] = [{"op": "upsert", "record": {
                "id": "keep", "condition": "Initial", "lesson": "Keep valid memory",
                "origin": "demonstration", "evidence_frames": [0],
            }}]
            value["keyframe_changes"] = [{
                "op": "save", "frame_id": 0, "description": "Initial", "reason": "Retain",
            }]
            return value
        if call == 2:
            if kind == "json":
                return "{invalid json"
            value = invalid(context, kind)
            value["memory_changes"]["experiences"] = [{"op": "delete", "id": "keep"}]
            value["keyframe_changes"] = [{"op": "delete", "frame_id": 0}]
            return value
        return good(context)

    env, planner, output, summary = run(tmp_path, respond)
    assert summary["success"] and summary["planner_calls"] == 3
    assert env.steps == 1 and env.closed
    assert summary["decision_recovery"] == {
        "max_corrections": 2, "rejected_decisions": 1, "correction_calls": 1,
        "recovered_decisions": 1, "last_rejection": None,
    }
    before, after = planner.inputs[1:]
    for key in ("memory", "keyframe_memory", "current_frame", "observation", "environment_steps"):
        assert before[key] == after[key]
    assert planner.images[1] == planner.images[2]
    assert after["remaining_calls"] == 1
    assert after["memory"]["experiences"][0]["id"] == "keep"
    assert after["decision_recovery"]["previous_rejection"]["status"] == "rejected"
    audit = output / "audit/call_0001"
    assert json.loads((audit / "before.json").read_text()) == json.loads((audit / "after.json").read_text())
    assert not (output / "memory_versions/call_0001.json").exists()
    assert not (audit / "fds_verdict.json").exists()
    timing = episode_timing(output, 0)
    assert (timing["prompt_tokens"], timing["completion_tokens"]) == (30, 9)
    assert timing["failed_attempt_usage_unknown"] == timing["transport_retries"] == 0
    assert timing["model_request_attempts_recorded"] == 3


@pytest.mark.parametrize("max_calls,corrections,calls,reason", [
    (10, 2, 3, "decision_recovery_exhausted"),
    (10, 0, 1, "decision_recovery_exhausted"),
    (1, 2, 1, "budget_exhausted"),
    (2, 2, 2, "budget_exhausted"),
])
def test_repeated_invalid_decisions_are_bounded(tmp_path, max_calls, corrections, calls, reason):
    env, planner, output, summary = run(
        tmp_path, lambda context, call: invalid(context), max_calls=max_calls, corrections=corrections,
    )
    assert summary["stop_reason"] == reason
    assert not summary["success"] and env.steps == 0 and env.closed
    assert summary["planner_calls"] == len(planner.inputs) == calls
    assert summary["decision_recovery"]["correction_calls"] == calls - 1
    assert len(list((output / "audit").glob("call_*/decision_rejection.json"))) == calls


def test_successful_correction_resets_consecutive_limit(tmp_path):
    _, _, _, summary = run(
        tmp_path, lambda context, call: invalid(context) if call % 2 else good(context),
        max_calls=4, corrections=1, done_after=2,
    )
    assert summary["success"]
    assert summary["decision_recovery"]["recovered_decisions"] == 2


def test_contract_off_recovers_without_fds_instructions(tmp_path):
    _, planner, _, summary = run(
        tmp_path, lambda context, call: invalid(context, "batch") if call == 1 else good(context),
        mode="off",
    )
    assert summary["success"]
    for context in planner.inputs:
        assert "fds_protocol" not in context
        assert "decision_state" not in context["execution_contract"]
        assert "fds_contract" not in schema_for(context)["properties"]


@pytest.mark.parametrize("error", [RuntimeError("API failed"), ValueError("implementation bug")])
def test_other_errors_propagate_without_correction(tmp_path, error):
    env = Environment()
    planner = Planner(lambda *_: (_ for _ in ()).throw(error))
    harness = TaskHarness(env, planner)
    with pytest.raises(type(error), match=str(error)):
        harness.run(0, tmp_path / "episode")
    assert env.steps == 0 and env.closed and len(planner.inputs) == 1
    summary = json.loads((tmp_path / "episode/summary.json").read_text())
    assert summary["stop_reason"] == "error"
    assert summary["decision_recovery"]["rejected_decisions"] == 0


def test_typed_rejection_cannot_trigger_transport_retries(tmp_path):
    planner = Planner(lambda context, _: invalid(context))
    context = {
        "decision_protocol": "task_harness", "contract_mode": "full",
        "max_tool_steps": 8, "max_tool_calls": 8,
    }
    from fds.task_harness.contracts import definitions

    context["tool_definitions"] = definitions(8, False, 4)
    wrapped = RecoveringPlanner(planner, RecoveryConfig(pause_seconds=0))
    with pytest.raises(DecisionValidationError):
        wrapped.decide(context, [], tmp_path / "call")
    assert len(planner.inputs) == 1
    retry = json.loads((tmp_path / "call/retry.json").read_text())
    assert retry["attempts"][0]["status"] == "decision_rejected"
    assert retry["attempts"][0]["usage_known"]
