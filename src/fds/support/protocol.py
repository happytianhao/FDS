"""Small action-only interfaces; no memory, plan ledger or implicit dialogue."""

import json
from copy import deepcopy
from dataclasses import dataclass

import jsonschema

from fds.common.schema import array, enum, obj, text
from fds.common.vla_language import contract, template_pattern

VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
QUATERNION = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
GROUND = obj(
    subject=text(64),
    label=text(100),
    frame={"type": "integer", "minimum": 0},
    point_yx={
        "type": "array",
        "items": {"type": "number", "minimum": 0, "maximum": 1000},
        "minItems": 2,
        "maxItems": 2,
    },
    confidence=enum("high", "medium", "low"),
)


def definitions(method, budget, policy=None, gripper=True, max_pose_targets=8):
    steps = {"type": "integer", "minimum": 1, "maximum": budget}
    tools = [
        {
            "name": "wait",
            "arguments": obj(steps=steps),
            "description": "Hold measured joints and last applied gripper command; advance time to observe.",
        },
        {
            "name": "finish",
            "arguments": obj(reason=text(500)),
            "description": "Stop voluntarily. This does not declare official success.",
        },
    ]
    if method == "sys1+2" and policy is not None:
        subgoal = text(1000)
        arguments = obj(subgoal=subgoal, grounding=array(GROUND, 4), steps=steps)
        description = (
            "Generate one native VLA action chunk using CURRENT RGB/state and this subgoal."
        )
        if getattr(policy, "grounding_format", None) == "training_yx":
            subgoal["pattern"] = template_pattern()
            arguments["properties"]["grounding"]["description"] = (
                "Exactly one entry per {subject} coordinate placeholder in subgoal, with matching subject. "
                "If subgoal has no braces, this array must be empty even when its text mentions objects."
            )
            description += (
                " Use a published language template with {subject} placeholders and CURRENT FRONT "
                "normalized [y,x] points in 0..1000. Each placeholder needs exactly one grounding "
                "entry with the same subject. A template with no coordinate placeholders requires "
                "grounding=[]; do not add points for objects merely mentioned in its text. "
                "Every point must refer to the current front frame and have high or medium confidence. "
                "Low confidence is not executable. Do not upgrade confidence just to pass validation: "
                "choose wait only if new observations can resolve the uncertainty, or finish with "
                "the missing information when no supported informed action is available. "
                "No extra clauses. See policy_contract."
            )
        else:
            description += " Plain VLA models require natural-language text and grounding=[]."
        tools.append(
            {
                "name": "vla",
                "arguments": arguments,
                "description": description
                + " The native chunk may be shorter than requested; inspect fresh feedback.",
            }
        )
    if method == "sys2":
        target = obj(
            position_m=VECTOR,
            quaternion_wxyz=QUATERNION,
            gripper=enum(*(["hold", "open", "close"] if gripper else ["hold"])),
            steps={"anyOf": [steps, enum("auto")]},
        )
        tools.append(
            {
                "name": "move_ee",
                "arguments": obj(targets={**array(target, max_pose_targets), "minItems": 1}),
                "description": "Ordered ABSOLUTE TCP poses in robot base coordinates; quaternion [w,x,y,z], "
                "meters, right-handed axes. Each target is reached via bounded local IK and linear joint "
                "interpolation over its steps. steps=auto allocates bounded joint motion plus 4 endpoint settling steps "
                "from robot geometry; it does not inspect objects or detect contact. Explicit integer durations "
                "remain available. Gripper command takes effect at the START of that target, so separate an "
                "open approach from a stationary close target. Sum of resolved steps must fit max_tool_steps. All targets are "
                "validated before motion. Translation limits apply to the 3D Euclidean distance, "
                "not each coordinate separately. Limits compare consecutive requested poses; the first uses the "
                "measured TCP. IK and joint limits always use solved joint states. No collision planner. "
                "Use small steps near contact. "
                "Actual tracking feedback differs from IK solver residuals; inspect both.",
            }
        )
    return tools


def policy_contract(policy):
    if policy is None:
        return None
    model_id = getattr(policy, "metadata", {}).get("model_id", "unknown")
    result = {
        "model_id": model_id,
        "grounding_format": getattr(policy, "grounding_format", "plain"),
        "native_action_horizon": getattr(policy, "metadata", {}).get("action_horizon"),
        "language_contract": contract()
        if getattr(policy, "grounding_format", None) == "training_yx"
        else None,
    }
    if result["language_contract"] is not None:
        # Public directional-language convention; no task-name routing or state.
        result["language_contract"]["direction_convention"] = {
            "applies_to": "coordinate-free move commands, including diagonals and nearest-target routing",
            "reference": "robot base; the front camera faces the robot",
            "forward": "away from the robot base",
            "backward": "toward the robot base",
            "left": "robot-left, which appears to the RIGHT in the front camera",
            "right": "robot-right, which appears to the LEFT in the front camera",
            "diagonals": "combine the robot-relative forward/backward and left/right directions",
            "point_coordinates": "grounding points still use front IMAGE [y,x]; do not mirror those points",
        }
    return result


def schema_for(context):
    tools = context.get("tool_definitions")
    if tools is None:
        tools = definitions(
            context.get("decision_protocol", "sys1+2"), context.get("max_tool_steps", 50)
        )
    return obj(
        tool_call={"anyOf": [obj(name=enum(t["name"]), arguments=t["arguments"]) for t in tools]}
    )


@dataclass
class BaselineDecision:
    tool_call: dict

    @property
    def tool(self):
        return self.tool_call["name"]

    @classmethod
    def parse(cls, value, schema=None):
        if schema is None:
            # Broad offline parser; the runtime always validates its actual live interface too.
            variants = (
                definitions("sys2", 400, max_pose_targets=64)
                + definitions("sys1+2", 200, object())[2:]
            )
            schema = schema_for({"tool_definitions": variants})
        jsonschema.validate(value, schema)
        json.dumps(value, allow_nan=False)
        return cls(**deepcopy(value))


def make_prompt(context):
    method = context.get("decision_protocol", "sys1+2")
    role = (
        "Give the VLA one concrete subgoal grounded in the current images."
        if method == "sys1+2"
        else "Generate absolute end-effector target poses; the host converts them using IK."
    )
    if method == "sys1+2":
        role += (
            " A VLA call executes one short native action chunk, not necessarily a whole skill. "
            "Compare the supplied NEW frames in temporal order with the latest executed tool: "
            "what visible effect occurred, and which part of the public goal is actionable now? "
            "Continue the same physical skill when it is visibly incomplete; switch to the next "
            "subgoal when these frames show the intended effect. A closed gripper alone does not "
            "prove a grasp: look for the object moving with it. Placement includes release. "
            "A scene change after a control interaction can show its effect even after the hand "
            "has withdrawn; do not keep activating it merely because the hand is no longer touching it. "
            "Inspect early frames too: brief cues may disappear before the current frame. Use those "
            "new cues to choose the next action, but ground its coordinates in the current front image. "
            "Missing older history alone is not evidence that an earlier stage must be restarted: "
            "use the current visible task state, and acknowledge genuinely missing information. "
            "One completed subgoal is not the whole task. The environment stops terminal episodes "
            "automatically; finish is voluntary stopping and does not certify completion. "
            "Do not infer repetition counts from call numbers, elapsed steps, or a tool's wording."
        )
    if method == "sys2":
        role += (
            " Resolve the public goal and fresh cue before coordinate arithmetic. "
            "An ordered pattern or short coherent approach/close/lift sequence visible NOW should be "
            "encoded as bounded waypoints in ONE move_ee call when possible: its cue is gone next turn. "
            "For a fully specified short procedure, include its confident waypoints through the final operation "
            "within the supplied capacity; do not depend on remembering its count or cue on a later call. "
            "Use steps=auto for ordinary motion, explicit steps for deliberate dwell/contact. Batch free-space "
            "travel, then inspect new observations after uncertain contact. A gripper command starts at the "
            "start of its target: approach open, close at a stationary pose, then lift while holding. "
            "Maintain the current approach endpoint unless fresh visual evidence requires correction; do not "
            "silently switch an intended object/end during descent. current_robot is an optional arithmetic "
            "reference for the robot, never object detection; ignore off-image projections. "
            "Before finish, check every observable part of the goal. For pickup, verify the object moves "
            "with the gripper and is clearly free of its support; if clearance is uncertain, use a controlled "
            "verification lift. For placement, release and verify placement; for pressing, inspect its visible "
            "effect. Finger closure or revealing a target alone is not completion evidence. "
            "Use remaining_environment_steps and the rejection's attempted_tool_call/numeric error to "
            "budget or correct actions. No old image is available after rejection; wait briefly if needed."
        )
    return (
        f"You are Sys2 for a simulated robot, method={method}. Each request is STATELESS. {role}\n"
        "Use only the supplied public task goal, new RGB/proprioception, robot geometry, and latest execution. "
        "There is no external memory, stored-image retrieval, planning ledger or prior conversation. "
        "Initial cue/demo frames appear only in the first request. Every raw frame/action appears once. "
        "Unseen past events and success counts are unknown; do not invent them. "
        "Choose exactly one allowed tool_call matching the supplied schema. "
        "Tool execution and IK convergence do not prove task success. "
        "State: seven joint radians followed by one finger position in meters; gripper commands are distinct. "
        "Read coordinate and TCP conventions before moving. Use the latest measured pose and tracking error "
        "to adjust the next action. Do not use external files, tools, internet or benchmark oracle state.\n"
        + json.dumps(context, ensure_ascii=False, allow_nan=False)
    )
