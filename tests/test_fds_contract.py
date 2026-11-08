from fds.task_harness.fds import execution_verdict, validate_contract


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
