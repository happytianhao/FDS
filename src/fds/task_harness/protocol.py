"""Task-independent reasoning prompt and explicit, separately supplied contracts."""

import json
from copy import deepcopy
from dataclasses import dataclass

import jsonschema

from fds.common.schema import array, enum, obj, text
from fds.support.memory import ID, MEMORY
from fds.task_harness.fds import FDS_CONTRACT, validate_contract
from fds.task_harness.tools import validate_batch

SYSTEM_PROMPT = """You are the reasoning and decision-making component of an embodied agent.
Pursue the supplied overall goal using only the information and capabilities in this request.
Distinguish observations, interpretations, uncertainty, intentions, attempts and verified outcomes.
Treat prior conclusions and learned guidance as fallible; revise them when evidence warrants it.
Choose a coherent, bounded next objective and the information needed to assess its outcome.
Preserve relevant knowledge without inventing observations or claiming unobserved progress.
Follow the supplied interface contracts and budgets. Return only the specified structured response.
There is no implicit conversation history or access to external information.
"""

INSTRUCTION_KEYS = (
    "tool_definitions", "execution_contract", "memory_contract", "fds_protocol",
    "outcome_protocol", "coordinates", "joint_motion_limits", "task_language_conventions",
)

TRAINING_FRAME_ID = {"type": "string", "maxLength": 80, "pattern": "^training:[0-9]+:[0-9]+$"}
FRAME_ID = {"anyOf": [{"type": "integer", "minimum": 0}, TRAINING_FRAME_ID]}
REFS = array(FRAME_ID, 8)
RECORDS = {
    "objects": deepcopy(MEMORY["properties"]["objects"]["items"]),
    "events": deepcopy(MEMORY["properties"]["events"]["items"]),
    "experiences": obj(
        id=ID,
        condition=text(300),
        lesson=text(700),
        origin=enum("demonstration", "execution"),
        evidence_frames=REFS,
    ),
}
for record in RECORDS.values():
    record["properties"]["evidence_frames"] = deepcopy(REFS)
RECORDS["objects"]["properties"]["last_seen_frame"] = {
    "anyOf": [{"type": "integer", "minimum": -1}, TRAINING_FRAME_ID]
}
# Support either inventory scope; bounded groups retain their existing caps.
# None means no fixed record-count cap, including on object updates in one response.
LIMITS = {"objects": None, "events": 32, "experiences": 16}


class DecisionValidationError(ValueError):
    """A model response rejected before any memory update or tool execution."""

    def __init__(self, message, *, code, path, response):
        super().__init__(message)
        self.code, self.path, self.response = code, list(path), response

    def feedback(self):
        return {
            "code": self.code,
            "path": self.path,
            "message": str(self),
            "rejected_response": self.response,
        }


def schema_for(context):
    tools = context["tool_definitions"]
    batches = []
    for kind in ("environment", "auxiliary"):
        fields = {"kind": enum(kind)}
        if kind == "environment":
            fields["subgoal"] = text(1000)
        fields["tool_calls"] = {
            **array(
                {
                    "anyOf": [
                        obj(name=enum(t["name"]), arguments=t["arguments"])
                        for t in tools
                        if t["kind"] == kind
                    ]
                },
                context.get("max_tool_calls", 8),
            ),
            "minItems": 1,
        }
        if kind == "environment":
            fields["observation_steps"] = array(
                {"type": "integer", "minimum": 1, "maximum": context["max_tool_steps"]},
                context.get("max_observation_frames", 12) - 1,
            )
        batches.append(obj(**fields))
    memory_changes = {}
    for group, record in RECORDS.items():
        updates = {
            "type": "array",
            "items": {
                "anyOf": [
                    obj(op=enum("upsert"), record=record),
                    obj(op=enum("delete"), id=ID),
                ]
            },
        }
        if LIMITS[group] is not None:
            updates["maxItems"] = LIMITS[group]
        memory_changes[group] = updates
    response = {
        "tool_batch": {"anyOf": batches},
        "memory_changes": obj(**memory_changes),
        "keyframe_changes": array(
            {
                "anyOf": [
                    obj(
                        op=enum("save"),
                        frame_id=FRAME_ID,
                        description=text(400),
                        reason=text(400),
                    ),
                    obj(op=enum("delete"), frame_id=FRAME_ID),
                ]
            },
            16,
        ),
    }
    if context.get("contract_mode", "full") != "off":
        response["fds_contract"] = deepcopy(FDS_CONTRACT)
        if "outcome_protocol" in context:
            from fds.task_harness.postcheck import response_fields

            additions = response_fields(context)
            response["fds_contract"]["properties"].update(additions)
            response["fds_contract"]["required"].extend(additions)
        # Use exactly the provenance set enforced by the executor, not the
        # review-window inventory (whose images may not have been delivered).
        if "available_evidence_frames" in context:
            available = context["available_evidence_frames"]
            choices = []
            for kind, cls in (("integer", int), ("string", str)):
                values = list(dict.fromkeys(x for x in available if type(x) is cls))
                if values:
                    choices.append({"type": kind, "enum": values})
            for group in ("assumptions", "required_evidence"):
                refs = response["fds_contract"]["properties"][group]["items"]["properties"]["evidence_refs"]
                if choices:
                    refs["items"] = {"anyOf": deepcopy(choices)}
                else:
                    refs["maxItems"] = 0
    return obj(**response)


@dataclass
class TaskDecision:
    tool_batch: dict
    fds_contract: dict | None
    memory_changes: dict
    keyframe_changes: list

    @classmethod
    def parse(cls, value, schema, *, delivered_frames=None, retained_frames=None):
        # Only model-output validation errors enter recovery. Invalid schemas and
        # implementation errors must still propagate as fatal errors.
        response = json.dumps(value, ensure_ascii=False)
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError as exc:
            # Dynamic citation enums may reject at the provider-parse boundary,
            # before executor provenance validation. Preserve actionable citation
            # feedback there as well; do not weaken or repair the schema/value.
            contract_schema = schema.get("properties", {}).get("fds_contract", {})
            refs = contract_schema.get("properties", {}).get("assumptions", {}).get(
                "items", {}).get("properties", {}).get("evidence_refs", {})
            choices = refs.get("items", {}).get("anyOf", [])
            bounded = refs.get("maxItems") == 0 or bool(choices and all("enum" in c for c in choices))
            contract = value.get("fds_contract") if isinstance(value, dict) else None
            base_contract = {
                key: item for key, item in contract.items() if key in FDS_CONTRACT["properties"]
            } if isinstance(contract, dict) else None
            if bounded and jsonschema.Draft7Validator(FDS_CONTRACT).is_valid(base_contract):
                available = [] if refs.get("maxItems") == 0 else [x for c in choices for x in c["enum"]]
                try:
                    validate_contract(base_contract, delivered_frames=available)
                except ValueError as citation_error:
                    raise DecisionValidationError(
                        str(citation_error), code="contract_validation", path=["fds_contract"], response=response
                    ) from exc
            raise DecisionValidationError(
                exc.message, code="schema_validation", path=exc.absolute_path, response=response
            ) from exc
        try:
            json.dumps(value, allow_nan=False)
        except ValueError as exc:
            raise DecisionValidationError(
                str(exc), code="non_finite_json", path=[], response=response
            ) from exc
        # The provider's structured-output schema does not support uniqueItems.
        batch = value["tool_batch"]
        steps = batch.get("observation_steps", [])
        try:
            if len(set(steps)) != len(steps):
                raise ValueError("observation_steps must contain unique relative steps")
            validate_batch(batch)
            reviews = [t for t in batch["tool_calls"] if t["name"] == "review_frames"]
            if reviews:
                ids = reviews[0]["arguments"]["frame_ids"]
                if len(ids) != len(set(ids)):
                    raise ValueError("Review frame IDs must be unique")
        except ValueError as exc:
            raise DecisionValidationError(
                str(exc), code="batch_validation", path=["tool_batch"], response=response
            ) from exc
        fds_contract = value.get("fds_contract")
        if fds_contract is not None:
            try:
                validate_contract(
                    fds_contract,
                    delivered_frames=delivered_frames,
                    retained_frames=retained_frames,
                )
            except ValueError as exc:
                raise DecisionValidationError(
                    str(exc), code="contract_validation", path=["fds_contract"], response=response
                ) from exc
        return cls(
            tool_batch=deepcopy(value["tool_batch"]),
            fds_contract=deepcopy(fds_contract),
            memory_changes=deepcopy(value["memory_changes"]),
            keyframe_changes=deepcopy(value["keyframe_changes"]),
        )


def make_user_prompt(context):
    prompt_context = deepcopy(context)
    # contract_mode is an executor-side ablation switch, not a model instruction.
    prompt_context.pop("contract_mode", None)
    for key in INSTRUCTION_KEYS:
        prompt_context.pop(key, None)
    return json.dumps(prompt_context, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def make_system_prompt(context):
    """Base instructions and host-supplied interface rules, without episode observations."""
    contracts = {key: deepcopy(context[key]) for key in INSTRUCTION_KEYS if key in context}
    return SYSTEM_PROMPT + "\nInterface contracts:\n" + json.dumps(
        contracts, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def make_prompt(context):
    """Legacy combined representation used for audit comparisons and retry hashes."""
    return make_system_prompt(context) + "\n" + make_user_prompt(context)


def mock_decision(context):
    decision = TaskDecision(
        {
            "kind": "environment",
            "subgoal": "Observe the next state",
            "tool_calls": [{"name": "wait", "arguments": {"steps": 1}}],
            "observation_steps": [],
        },
        {
            "contract_id": "mock_wait",
            "selected_candidate": "wait",
            "assumptions": [],
            "candidates": [
                {
                    "id": "wait",
                    "description": "Wait for one environment step",
                    "requires": [],
                    "expected_outcome": "A new public frame is returned",
                    "falsifiers": ["The environment reports an action error"],
                    "risk": "low",
                }
            ],
            "required_evidence": [],
            "expected_outcome": "A new public frame is returned",
            "falsifiers": ["The environment reports an action error"],
            "risk": "low",
            "commitment": "committed",
        },
        {group: [] for group in RECORDS},
        [],
    )
    if "outcome_protocol" in context:
        pending = context.get("pending_outcome_review")
        decision.fds_contract.update(
            action_binding={"candidate_id": "wait", "tool_indices": [0],
                            "event_ids": [], "event_dependencies": []},
            outcome_review={
                "action_call": pending["action_call"] if pending else None,
                "checks": [
                    {"id": check["id"], "status": "unknown", "basis": "unknown",
                     "evidence_refs": [], "receipt_indices": [],
                     "reason": "Offline fixture makes no semantic assessment"}
                    for check in pending["checks"]
                ] if pending else [],
                "event_resolutions": [],
            },
        )
    return decision
