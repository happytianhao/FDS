from fds.task_harness.contracts import definitions, execution_contract
from fds.task_harness.fds import execution_verdict, validate_contract
from fds.task_harness.protocol import TaskDecision, mock_decision, schema_for


def contract(status="supported"):
    return {
        "contract_id": "pick_blue",
        "selected_candidate": "approach",
        "assumptions": [
            {
                "id": "blue_identity",
                "claim": "The selected object is the requested blue cube",
                "evidence_refs": [0],
                "confidence": 0.8,
                "status": status,
            }
        ],
        "candidates": [
            {
                "id": "approach",
                "description": "Move above the selected cube",
                "requires": ["blue_identity"],
                "expected_outcome": "TCP reaches the approach pose",
                "falsifiers": ["IK fails", "Target is absent from the predicted region"],
                "risk": "low",
            }
        ],
        "required_evidence": [],
        "expected_outcome": "The TCP reaches the approach pose",
        "falsifiers": ["IK fails"],
        "risk": "low",
        "commitment": "committed",
    }


def test_contract_requires_supported_assumptions_for_commitment():
    validate_contract(contract(), delivered_frames=[0], retained_frames=[])
    try:
        validate_contract(contract("tentative"), delivered_frames=[0], retained_frames=[])
    except ValueError as exc:
        assert "unresolved" in str(exc)
    else:
        raise AssertionError("tentative assumptions must not support a committed contract")


def test_execution_receipt_does_not_claim_semantic_success():
    assert execution_verdict(contract(), [{"status": "executed"}])["status"] == "uncertain"
    assert execution_verdict(
        contract(), [{"status": "rejected", "environment_feedback": {}}]
    )["status"] == "contradicted"


def test_contract_off_removes_fds_schema_and_execution_rules():
    context = {
        "contract_mode": "off",
        "tool_definitions": definitions(8, True, 4),
        "max_tool_calls": 8,
        "max_tool_steps": 8,
        "max_review_frames": 4,
    }
    schema = schema_for(context)
    assert "fds_contract" not in schema["properties"]
    decision = mock_decision(context)
    value = {
        "tool_batch": decision.tool_batch,
        "memory_changes": decision.memory_changes,
        "keyframe_changes": decision.keyframe_changes,
    }
    parsed = TaskDecision.parse(value, schema, delivered_frames=[0], retained_frames=[])
    assert parsed.fds_contract is None
    limits = type(
        "Limits",
        (),
        {"max_rotation_rad": 0.6, "max_joint_delta_rad": 1.2, "max_joint_step_rad": 0.08},
    )()
    assert "decision_state" not in execution_contract(limits, contract_mode="off")
