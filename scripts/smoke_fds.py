"""Offline smoke checks for the FDS contract and task-harness protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fds.task_harness.contracts import definitions, execution_contract  # noqa: E402
from fds.task_harness.fds import execution_verdict  # noqa: E402
from fds.task_harness.protocol import TaskDecision, mock_decision, schema_for  # noqa: E402


def main():
    context = {
        "tool_definitions": definitions(32, True, 8),
        "max_tool_calls": 8,
        "max_tool_steps": 32,
        "max_observation_frames": 12,
    }
    decision = mock_decision(context)
    parsed = TaskDecision.parse(
        {
            "tool_batch": decision.tool_batch,
            "fds_contract": decision.fds_contract,
            "memory_changes": decision.memory_changes,
            "keyframe_changes": decision.keyframe_changes,
        },
        schema_for(context),
        delivered_frames=[0],
        retained_frames=[],
    )
    verdict = execution_verdict(parsed.fds_contract, [{"status": "executed"}])
    assert verdict["status"] == "uncertain"

    off_context = {**context, "contract_mode": "off"}
    off_schema = schema_for(off_context)
    assert "fds_contract" not in off_schema["properties"]
    off_value = {
        "tool_batch": decision.tool_batch,
        "memory_changes": decision.memory_changes,
        "keyframe_changes": decision.keyframe_changes,
    }
    off_decision = TaskDecision.parse(
        off_value,
        off_schema,
        delivered_frames=[0],
        retained_frames=[],
    )
    assert off_decision.fds_contract is None
    assert "decision_state" not in execution_contract(
        type("Limits", (), {"max_rotation_rad": 0.6, "max_joint_delta_rad": 1.2, "max_joint_step_rad": 0.08})(),
        contract_mode="off",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "full_contract_id": parsed.fds_contract["contract_id"],
                "full_verdict": verdict,
                "off_contract": None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
