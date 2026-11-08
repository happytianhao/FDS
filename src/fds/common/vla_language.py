"""Public RoboMME language templates; no task implementation or oracle access.

Contract sources: RoboMME policy examples/robomme/subgoal_prediction/gemini/prompts,
dataset_builder/{build_robomme_dataset,robomme_h5_utils}.py, and training/config.py.
Only the language interface is constrained: visual accuracy/distribution is not guaranteed.
"""

import re

from fds.grounding import POINT

CONTRACT = "robomme_training_v1"
COLORS = "red|blue|green|yellow|orange|purple|white|black"
ORDINALS = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
TEMPLATES = (
    "pick up the {ordinal} {color} cube at {point}",
    "put it into the bin at {point}",
    "press the button at {point}",
    "pick up the {color} cube at {point} for the {ordinal} time",
    "place the {color} cube onto the target at {point}",
    "press the button to stop at {point}",
    "pick up the cube at {point}",
    "place the cube onto the target at {point}",
    "place the cube onto the correct target at {point}",
    "pick up the peg at {point}",
    "hook the cube at {source} to the target at {target} with the peg",
    "close gripper and push the cube at {source} to the target at {target}",
    "pick up the container at {point} that hides the {color} cube",
    "put down the container",
    "press the first button at {point}",
    "press the second button at {point}",
    "pick up the peg by grasping the {end} end at {point}",
    "insert the peg from the {side} side of the box at {point}",
    "pick up the {ordinal} highlighted cube at {point}, which is {color}",
    "place the cube onto the table",
    "move to the top of the button at {point} to prepare",
    "remain static",
    "press the button to stop the cube on the target at {point}",
    "pick up the {color} cube at {point}",
    "move to the top of the right target at {point} for the {ordinal} time",
    "move to the top of the left target at {point} for the {ordinal} time",
    "put the {color} cube on the table",
    "pick up the correct cube at {point} for the {ordinal} time",
    "put it down",
    "press the button at {point} to finish",
    "move forward",
    "move left",
    "move right",
    "move backward",
    "move forward-left",
    "move forward-right",
    "move backward-left",
    "move backward-right",
    "move to the nearest {side} target by circling around the stick {direction}",
)
PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_-]+)\}")


def template_pattern():
    alternatives = []
    slots = {
        "color": f"(?:{COLORS})",
        "ordinal": f"(?:{ORDINALS})",
        "end": "(?:near|far)",
        "side": "(?:left|right)",
        "direction": "(?:clockwise|counterclockwise)",
        **dict.fromkeys(("point", "source", "target"), r"\{[a-zA-Z0-9_-]+\}"),
    }
    for template in TEMPLATES:
        pattern = re.escape(template)
        for name, replacement in slots.items():
            pattern = pattern.replace(re.escape("{" + name + "}"), replacement)
        alternatives.append(pattern)
    return "^(?:" + "|".join(alternatives) + ")$"


def validate_template(instruction):
    if re.fullmatch(template_pattern(), instruction) is None:
        raise ValueError(
            "GroundSG subgoal must match a published training template; no Targets suffix, explanations or extra clauses"
        )


def validate_wire(instruction, image_shape):
    if tuple(image_shape[:2]) != (256, 256):
        raise ValueError("RoboMME GroundSG training contract requires 256x256 front RGB")
    validate_template(POINT.sub("{point}", instruction))


def compile_training_subgoal(args, current_frame, visible):
    instruction = args["subgoal"]
    validate_template(instruction)
    used = PLACEHOLDER.findall(instruction)
    if not used and args["grounding"]:
        raise ValueError(
            "This template has no coordinate placeholders: use grounding=[]; mentioned objects need no points"
        )
    targets = {g["subject"]: g for g in args["grounding"]}
    if (
        len(targets) != len(args["grounding"])
        or len(used) != len(set(used))
        or set(used) != set(targets)
    ):
        raise ValueError("Each training-template point needs one distinct named grounding entry")
    if current_frame not in visible or visible[current_frame]:
        raise ValueError(
            "GroundSG requires the CURRENT live front image, including for coordinate-free subgoals"
        )
    for g in targets.values():
        if g["frame"] != current_frame or g["confidence"] == "low":
            raise ValueError("GroundSG requires confident CURRENT front-view grounding")
        if not g["label"].strip() or any(c in g["label"] for c in "{}<>"):
            raise ValueError(
                "Grounding label must identify a visible target without coordinate syntax"
            )
        if len(g["point_yx"]) != 2 or any(not 0 <= v <= 1000 for v in g["point_yx"]):
            raise ValueError("GroundSG point must be finite normalized [y,x] in 0..1000")
    return PLACEHOLDER.sub(
        lambda m: "<{}, {}>".format(*(round(v) for v in targets[m[1]]["point_yx"])), instruction
    )


def contract():
    return {
        "name": CONTRACT,
        "templates": list(TEMPLATES),
        "image_size": [256, 256],
        "camera": "front",
        "coordinate_order": "y,x",
        "model_points": "normalized_0_1000",
        "wire_points": "integer_pixels_0_255",
        "point_slots": "replace with {grounding_subject}",
        "color_values": COLORS.split("|"),
        "ordinal_values": ORDINALS.split("|"),
        "extra_clauses_allowed": False,
        "note": "Only supported public language templates are accepted; no oracle labels or task-state access.",
    }
