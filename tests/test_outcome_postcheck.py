"""Exercise the real recovery/memory/tool loop without a model or simulator."""

import hashlib
import json
from copy import deepcopy

import pytest
from test_decision_recovery import Environment, Planner, good

from fds.cli import resolve_executor_defaults
from fds.task_harness.runtime import TaskHarness


def start_event(context):
    value = good(context)
    value["fds_contract"]["action_binding"]["event_ids"] = ["arrival"]
    value["memory_changes"]["events"] = [{"op": "upsert", "record": {
        "id": "arrival", "objective": "Observe one control step",
        "success_criteria": "A post-action observation is available",
        "status": "active", "origin": "execution", "progress": "Attempt pending",
        "evidence_frames": [context["current_frame"]],
    }}]
    value["keyframe_changes"] = [{"op": "save", "frame_id": 0,
                                  "description": "Before action", "reason": "Comparison"}]
    return value


def reviewed_finish(context, status="succeeded"):
    value = good(context)
    pending = context["pending_outcome_review"]
    review = value["fds_contract"]["outcome_review"]
    for check, predicate in zip(review["checks"], pending["checks"], strict=True):
        check.update(
            status="unknown" if status == "unknown" else (
                "met" if (predicate["kind"] == "expected") == (status == "succeeded") else "violated"
            ),
            basis="unknown" if status == "unknown" else "visual",
            evidence_refs=[context["current_frame"]], reason="Fixture's explicit assessment",
        )
    review["event_resolutions"] = [{"event_id": "arrival", "status": status,
                                   "check_ids": pending["event_checks"]["arrival"]}]
    row = deepcopy(context["memory"]["events"][0])
    row.update(status=status, evidence_frames=[context["current_frame"]], progress="Reviewed")
    value["memory_changes"]["events"] = [{"op": "upsert", "record": row}]
    value["tool_batch"] = {"kind": "environment", "subgoal": "Stop honestly",
                           "tool_calls": [{
                               "name": "finish", "arguments": {"reason": "End fixture"},
                           }]}
    return value


def execute(tmp_path, respond, *, calls=3, postcheck=True, done_after=10):
    env, planner = Environment(done_after), Planner(respond)
    harness = TaskHarness(env, planner, max_calls=calls, max_steps=10, max_tool_steps=8,
                          fds_options={"postcheck_enabled": postcheck})
    out = tmp_path / "episode"
    return env, planner, out, harness.run(0, out)


def test_review_updates_event_and_binds_exact_action(tmp_path):
    def respond(context, call):
        if call == 1:
            return start_event(context)
        value = reviewed_finish(context)
        value["fds_contract"]["action_binding"]["event_dependencies"] = [
            {"event_id": "arrival", "revision": 1}
        ]
        return value

    env, planner, out, summary = execute(tmp_path, respond)
    assert env.steps == 1 and summary["planner_calls"] == 2
    assert summary["stop_reason"] == "planner_finish" and not summary["success"]
    assert summary["outcome_tracking"]["event_succeeded"] == 1
    binding = json.loads((out / "audit/call_0000/action_outcomes.json").read_text())
    decision = json.loads((out / "audit/call_0000/decision.json").read_text())
    canonical = json.dumps(decision["tool_batch"], sort_keys=True, separators=(",", ":"))
    assert binding["tool_batch_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    memory = json.loads((out / "memory_versions/call_0001.json").read_text())
    assert memory["memory"]["events"][0]["status"] == "succeeded"
    assert planner.inputs[1]["event_memory_versions"][0]["revision"] == 1
    for context in planner.inputs:
        encoded = json.dumps(context)
        assert '"official_status"' not in encoded and '"scorer_status"' not in encoded


@pytest.mark.parametrize("damage", [
    "old_frame", "unavailable_frame", "receipt_only", "missing_check", "wrong_action",
    "wrong_binding", "stale_dependency", "unknown_dependency", "missing_memory_update",
    "rejected_memory_update", "changed_criterion", "contradictory_success", "omitted_falsifier",
])
def test_bad_review_is_atomic_and_correction_explicitly_reviews_previous_observation(tmp_path, damage):
    def respond(context, call):
        if call == 1:
            return start_event(context)
        if call == 3:
            value = reviewed_finish(context, "unknown")
            for check in value["fds_contract"]["outcome_review"]["checks"]:
                check["evidence_refs"] = []
            value["memory_changes"]["events"][0]["record"]["evidence_frames"] = [0]
            value["tool_batch"] = {"kind": "auxiliary", "tool_calls": [{
                "name": "review_frames", "arguments": {
                    "window_id": "batch_0001", "frame_ids": [context["current_frame"]],
                },
            }]}
            return value
        value = reviewed_finish(context)
        if call != 2:
            return value
        review = value["fds_contract"]["outcome_review"]
        binding = value["fds_contract"]["action_binding"]
        if damage in ("old_frame", "unavailable_frame"):
            review["checks"][0]["evidence_refs"] = [0 if damage == "old_frame" else 999]
        elif damage == "receipt_only":
            review["checks"][0].update(basis="public_feedback", evidence_refs=[], receipt_indices=[0])
        elif damage == "missing_check":
            review["checks"].pop()
        elif damage == "wrong_action":
            review["action_call"] = 42
        elif damage == "wrong_binding":
            binding["tool_indices"] = []
        elif damage == "stale_dependency":
            binding["event_dependencies"] = [{"event_id": "arrival", "revision": 2}]
        elif damage == "unknown_dependency":
            value = reviewed_finish(context, "unknown")
            value["fds_contract"]["action_binding"]["event_dependencies"] = [
                {"event_id": "arrival", "revision": 1}
            ]
            value["fds_contract"]["commitment"] = "tentative"
        elif damage == "missing_memory_update":
            value["memory_changes"]["events"] = []
        elif damage == "rejected_memory_update":
            value["memory_changes"]["events"][0]["record"]["evidence_frames"] = [999]
        elif damage == "changed_criterion":
            value["memory_changes"]["events"][0]["record"]["success_criteria"] = "Always true"
        elif damage == "contradictory_success":
            review["checks"][1]["status"] = "met"
        elif damage == "omitted_falsifier":
            review["checks"][1]["status"] = "met"
            review["event_resolutions"][0]["check_ids"].pop()
        return value

    env, planner, out, summary = execute(tmp_path, respond, calls=4)
    assert env.steps == 1 and summary["planner_calls"] == 4
    assert summary["decision_recovery"]["rejected_decisions"] == 1
    assert summary["decision_recovery"]["recovered_decisions"] == 1
    assert summary["outcome_tracking"]["reviews"] == 2
    for key in ("memory", "keyframe_memory", "pending_outcome_review", "event_memory_versions",
                "environment_steps"):
        assert planner.inputs[1][key] == planner.inputs[2][key]
    assert planner.inputs[2]["observation"]["frames"] == []
    assert planner.inputs[2]["available_evidence_frames"] == [0]
    assert planner.images[1] == planner.images[3]
    audit = out / "audit/call_0001"
    assert json.loads((audit / "before.json").read_text()) == json.loads((audit / "after.json").read_text())
    assert not (out / "memory_versions/call_0001.json").exists()
    assert not (audit / "outcome_review.json").exists()


def test_unknown_review_survives_auxiliary_call_then_can_be_resolved(tmp_path):
    def respond(context, call):
        if call == 1:
            return start_event(context)
        value = reviewed_finish(context, "unknown" if call == 2 else "succeeded")
        if call == 2:
            value["tool_batch"] = {"kind": "auxiliary", "tool_calls": [{
                "name": "review_frames", "arguments": {
                    "window_id": context["observation"]["review_window"]["window_id"],
                    "frame_ids": [context["current_frame"]],
                },
            }]}
        return value

    env, planner, _, summary = execute(tmp_path, respond)
    assert env.steps == 1 and summary["planner_calls"] == 3
    assert planner.inputs[2]["pending_outcome_review"]["action_call"] == 1
    assert planner.inputs[2]["memory"]["events"][0]["status"] == "unknown"
    assert planner.inputs[2]["event_memory_versions"][0]["revision"] == 2
    assert summary["outcome_tracking"]["event_unknown"] == 1
    assert summary["outcome_tracking"]["event_succeeded"] == 1


def test_delayed_outcome_survives_physical_observation_and_uses_later_evidence(tmp_path):
    def respond(context, call):
        if call == 1:
            return start_event(context)
        value = reviewed_finish(context, "unknown" if call == 2 else "succeeded")
        if call == 2:
            # A real environment step, not auxiliary image retrieval.
            value["tool_batch"] = good(context)["tool_batch"]
        return value

    env, planner, out, summary = execute(tmp_path, respond)
    assert env.steps == 2 and summary["planner_calls"] == 3
    pending = planner.inputs[2]["pending_outcome_review"]
    assert pending["action_call"] == 2
    assert pending["event_checks"]["arrival"] == ["call_1_check_0", "call_1_check_1"]
    old = next(c for c in pending["checks"] if c["id"] == "call_1_check_0")
    assert old["action_start_frame"] == 0 and old["action_end_frame"] == 1
    assert planner.inputs[2]["current_frame"] == 2
    assert summary["outcome_tracking"]["event_succeeded"] == 1
    memory = json.loads((out / "memory_versions/call_0002.json").read_text())
    assert memory["memory"]["events"][0]["evidence_frames"] == [2]


def test_unknown_event_deleted_during_auxiliary_is_not_resurrected(tmp_path):
    def respond(context, call):
        if call == 1:
            return start_event(context)
        value = reviewed_finish(context, "unknown") if call == 2 else good(context)
        if call == 2:
            value["memory_changes"]["events"] = [{"op": "delete", "id": "arrival"}]
        if call < 4:
            value["tool_batch"] = {"kind": "auxiliary", "tool_calls": [{
                "name": "review_frames", "arguments": {
                    "window_id": context["observation"]["review_window"]["window_id"],
                    "frame_ids": [context["current_frame"]],
                },
            }]}
        else:
            value["tool_batch"] = {"kind": "environment", "subgoal": "Stop",
                                   "tool_calls": [{
                                       "name": "finish", "arguments": {"reason": "Unknown outcome"},
                                   }]}
        return value

    env, planner, _, summary = execute(tmp_path, respond, calls=4)
    assert env.steps == 1 and summary["planner_calls"] == 4
    assert not summary["decision_recovery"]["rejected_decisions"]
    for context in planner.inputs[2:]:
        assert context["memory"]["events"] == []
        assert context["pending_outcome_review"]["event_ids"] == []


def test_confirmed_event_is_not_counted_again_during_auxiliary_review(tmp_path):
    def respond(context, call):
        if call == 1:
            return start_event(context)
        value = reviewed_finish(context) if call == 2 else good(context)
        if call == 2:
            value["tool_batch"] = {"kind": "auxiliary", "tool_calls": [{
                "name": "review_frames", "arguments": {
                    "window_id": context["observation"]["review_window"]["window_id"], "frame_ids": [1],
                },
            }]}
        else:
            value["tool_batch"] = {"kind": "environment", "subgoal": "Stop",
                                   "tool_calls": [{
                                       "name": "finish", "arguments": {"reason": "Done"},
                                   }]}
            value["fds_contract"]["action_binding"]["event_dependencies"] = [
                {"event_id": "arrival", "revision": 2}
            ]
        return value

    _, planner, _, summary = execute(tmp_path, respond)
    assert not summary["decision_recovery"]["rejected_decisions"]
    assert summary["outcome_tracking"]["event_succeeded"] == 1
    assert planner.inputs[2]["pending_outcome_review"]["event_ids"] == []


def test_unchanged_prior_assessment_survives_archived_pixels_without_new_citation_rights(tmp_path):
    def respond(context, call):
        if call == 1:
            return start_event(context)
        value = reviewed_finish(context, "unknown" if call == 2 else "succeeded")
        checks = value["fds_contract"]["outcome_review"]["checks"]
        if call == 2:
            checks[0].update(status="met", basis="visual", evidence_refs=[1])
            value["tool_batch"] = good(context)["tool_batch"]
        else:
            old = next(c for c in context["pending_outcome_review"]["checks"] if c["id"] == "call_1_check_0")
            index = next(i for i, c in enumerate(checks) if c["id"] == old["id"])
            checks[index] = deepcopy(old["last_assessment"])
        return value

    env, planner, _, summary = execute(tmp_path, respond)
    assert env.steps == 2 and summary["planner_calls"] == 3
    assert 1 not in planner.inputs[2]["available_evidence_frames"]
    assert not summary["decision_recovery"]["rejected_decisions"]
    assert summary["outcome_tracking"]["event_succeeded"] == 1


def test_cannot_mark_success_before_any_action(tmp_path):
    def respond(context, call):
        value = start_event(context)
        value["memory_changes"]["events"][0]["record"]["status"] = "succeeded"
        return value

    env, _, _, summary = execute(tmp_path, respond, calls=5)
    assert env.steps == 0 and summary["planner_calls"] == 3
    assert summary["stop_reason"] == "decision_recovery_exhausted"


def test_zero_step_rejection_can_be_reviewed_without_fabricating_images(tmp_path):
    def respond(context, call):
        if call == 1:
            value = start_event(context)
            value["tool_batch"]["tool_calls"] *= 2
            for tool in value["tool_batch"]["tool_calls"]:
                tool["arguments"]["steps"] = 8
            value["fds_contract"]["action_binding"]["tool_indices"] = [0, 1]
            return value
        value = reviewed_finish(context, "failed")
        for check in value["fds_contract"]["outcome_review"]["checks"]:
            check.update(basis="public_feedback", receipt_indices=[0], evidence_refs=[])
        return value

    env, planner, _, summary = execute(tmp_path, respond)
    assert env.steps == 0 and summary["planner_calls"] == 2
    assert planner.inputs[1]["pending_outcome_review"]["receipts"][0]["status"] == "rejected"
    assert summary["outcome_tracking"]["event_failed"] == 1
    assert not summary["success"]


def test_terminal_environment_does_not_add_a_verifier_call(tmp_path):
    env, _, _, summary = execute(tmp_path, lambda c, n: start_event(c), done_after=1)
    assert summary["success"] and env.steps == summary["planner_calls"] == 1
    assert not summary["outcome_tracking"]["pending_action_reviewed"]
    assert summary["outcome_tracking"]["reviews"] == 0


def test_disabled_postcheck_keeps_original_contract_protocol(tmp_path):
    _, planner, _, summary = execute(tmp_path, lambda c, n: good(c), postcheck=False, done_after=1)
    assert summary["success"] and not summary["postcheck_enabled"]
    assert "outcome_tracking" not in summary
    assert "outcome_protocol" not in planner.inputs[0]
    assert "action_binding" not in good(planner.inputs[0])["fds_contract"]


@pytest.mark.parametrize("mode,enabled", [("off", False), ("full", True)])
def test_resolved_config_records_postcheck_default(mode, enabled):
    config = resolve_executor_defaults({"fds": {"contract_mode": mode}})
    assert config["fds"]["postcheck_enabled"] is enabled


def test_postcheck_flag_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        resolve_executor_defaults({"fds": {"postcheck_enabled": "false"}})
