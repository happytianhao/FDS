"""Falsifiable Decision State contracts and host-side checks.

The model proposes claims; the host checks structure and provenance.  The host
never turns command execution into semantic success.  Public execution
feedback is recorded as evidence for the next decision and for meta-harness
diagnosis.
"""

from __future__ import annotations

from copy import deepcopy

from fds.common.schema import array, enum, obj, text

FDS_ID = {"type": "string", "minLength": 1, "maxLength": 64, "pattern": "^[a-zA-Z0-9_-]+$"}
FDS_FRAME = {
    "anyOf": [
        {"type": "integer", "minimum": 0},
        {"type": "string", "pattern": "^training:[0-9]+:[0-9]+$", "maxLength": 80},
    ]
}
FDS_REFS = array(FDS_FRAME, 8)

ASSUMPTION = obj(
    id=FDS_ID,
    claim=text(600),
    evidence_refs=FDS_REFS,
    confidence={"type": "number", "minimum": 0, "maximum": 1},
    status=enum("supported", "tentative", "contradicted", "unknown"),
)

CANDIDATE = obj(
    id=FDS_ID,
    description=text(500),
    requires=array(FDS_ID, 16),
    expected_outcome=text(600),
    falsifiers=array(text(500), 8),
    risk=enum("low", "medium", "high"),
)

REQUIRED_EVIDENCE = obj(
    id=FDS_ID,
    purpose=text(500),
    status=enum("resolved", "unresolved", "unavailable"),
    evidence_refs=FDS_REFS,
)

FDS_CONTRACT = obj(
    contract_id=FDS_ID,
    selected_candidate=FDS_ID,
    assumptions=array(ASSUMPTION, 32),
    candidates=array(CANDIDATE, 8),
    required_evidence=array(REQUIRED_EVIDENCE, 16),
    expected_outcome=text(800),
    falsifiers=array(text(500), 12),
    risk=enum("low", "medium", "high"),
    commitment=enum("tentative", "committed"),
)


def schema():
    """Return a copy so callers cannot mutate the shared protocol definition."""
    return deepcopy(FDS_CONTRACT)


def validate_contract(contract: dict, *, delivered_frames=None, retained_frames=None):
    """Validate cross-field invariants and evidence provenance.

    JSON schema checks types and bounds.  This function checks relationships
    that a provider schema cannot express, including candidate dependencies and
    references to frames that were actually supplied to the model.
    """
    if not isinstance(contract, dict):
        raise ValueError("fds_contract must be an object")
    assumptions = {item["id"]: item for item in contract["assumptions"]}
    candidates = {item["id"]: item for item in contract["candidates"]}
    if len(assumptions) != len(contract["assumptions"]):
        raise ValueError("fds_contract assumptions must have unique IDs")
    if len(candidates) != len(contract["candidates"]):
        raise ValueError("fds_contract candidates must have unique IDs")
    selected = contract["selected_candidate"]
    if selected not in candidates:
        raise ValueError("fds_contract.selected_candidate must name a candidate")
    for candidate in contract["candidates"]:
        missing = [ref for ref in candidate["requires"] if ref not in assumptions]
        if missing:
            raise ValueError(f"Candidate {candidate['id']} has unknown assumptions: {missing}")
    if delivered_frames is not None or retained_frames is not None:
        available = set(delivered_frames or ()) | set(retained_frames or ())
        invalid_refs = []
        for group in ("assumptions", "required_evidence"):
            for index, item in enumerate(contract[group]):
                unknown = [ref for ref in item["evidence_refs"] if ref not in available]
                if unknown:
                    invalid_refs.append(
                        f"fds_contract.{group}[{index}].evidence_refs "
                        f"(id={item['id']}) cites unavailable evidence: {unknown}"
                    )
        if invalid_refs:
            raise ValueError(
                "; ".join(invalid_refs)
                + f". Available citation IDs are {sorted(available, key=str)}. "
                "Correct the reference arrays themselves; changing confidence, status "
                "or commitment does not make an unavailable frame valid. Each evidence_refs "
                f"array accepts at most {FDS_REFS['maxItems']} individual frame IDs. "
                "Choose a relevant subset of already supplied evidence if necessary; "
                "never concatenate IDs or encode ranges as one integer. Check evidence_frames "
                "in proposed memory updates for the same issue. Do not invent replacement "
                "evidence or automatically split an invalid ID."
            )
    selected_candidate = candidates[selected]
    if contract["commitment"] == "committed":
        unresolved = [
            ref
            for ref in selected_candidate["requires"]
            if assumptions[ref]["status"] != "supported"
        ]
        unresolved += [
            item["id"]
            for item in contract["required_evidence"]
            if item["status"] != "resolved"
        ]
        if unresolved:
            raise ValueError(
                "A committed FDS contract has unresolved requirements: "
                + ", ".join(unresolved)
            )
    if selected_candidate["risk"] == "high" and not contract["falsifiers"]:
        raise ValueError("High-risk FDS candidates require at least one contract falsifier")
    return deepcopy(contract)


def execution_verdict(contract: dict, receipts: list[dict]) -> dict:
    """Classify only public mechanical feedback; semantic success stays unknown."""
    scope = {"scope": "mechanical_execution", "semantic_status": "unassessed"}
    rejected = [r for r in receipts if r.get("status") == "rejected"]
    public_errors = [
        r
        for r in receipts
        if r.get("environment_feedback", {}).get("action_error")
        or r.get("environment_feedback", {}).get("trial_reset")
    ]
    if rejected or public_errors:
        return {
            **scope,
            "status": "contradicted",
            "reason": "public_execution_feedback",
            "rejected_receipts": rejected,
            "public_error_receipts": public_errors,
        }
    if not receipts:
        return {**scope, "status": "uncertain", "reason": "no_execution_receipt"}
    return {
        **scope,
        "status": "uncertain",
        "reason": "command_execution_does_not_certify_semantic_outcome",
    }
