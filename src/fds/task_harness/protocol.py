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
        response["fds_contract"] = FDS_CONTRACT
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


def make_prompt(context):
    prompt_context = deepcopy(context)
    # contract_mode is an executor-side ablation switch, not a model instruction.
    prompt_context.pop("contract_mode", None)
    return (
        SYSTEM_PROMPT
        + "\n"
        + json.dumps(prompt_context, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )


def mock_decision(context):
    return TaskDecision(
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
