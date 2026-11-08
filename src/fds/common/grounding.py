"""Compile explicitly named Sys2 targets; no detector or metric geometry is implied."""

import re

PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def compile_subgoal(decision, current_frame, visible, memory_records, normalized):
    instruction = decision.subgoal
    if "<" in instruction or ">" in instruction:
        raise ValueError("Use structured grounding entries, not inline guessed coordinates")
    targets = {}
    for target in decision.grounding:
        key = target["subject"]
        if not key or key in targets or not target["label"].strip():
            raise ValueError("Grounding target identities and labels must be nonempty and unique")
        if any(c in target["label"] for c in "{}<>"):
            raise ValueError("Grounding label cannot contain coordinate syntax")
        if normalized:
            if target["frame"] != current_frame or target["confidence"] == "low":
                raise ValueError("GroundSG requires confident CURRENT front-view grounding")
            if len(target["point_yx"]) != 2:
                raise ValueError("Grounding requires a normalized [y,x] point")
            from_new_frame = current_frame in visible and not visible[current_frame]
            from_memory = any(
                r["subject"] == key
                and not r["stale"]
                and r["status"] == "observed"
                and r["position"]["space"] == "image_yx"
                and r["position"]["frame"] == current_frame
                and r["position"]["coordinates"] == target["point_yx"]
                for r in memory_records
            )
            if not (from_new_frame or from_memory):
                raise ValueError(
                    "Grounding needs new pixels or an explicitly stored current-frame position"
                )
            y, x = target["point_yx"]
            targets[key] = f"{target['label']} <{round(y)}, {round(x)}>"
        else:
            targets[key] = target["label"]
    used = set(PLACEHOLDER.findall(instruction))
    if normalized and (not used or used != set(targets)):
        raise ValueError("Each GroundSG target must have one named placeholder and grounding entry")
    if not used <= set(targets):
        raise ValueError("Subgoal contains an unresolved target")
    result = PLACEHOLDER.sub(lambda m: targets[m.group(1)], instruction)
    if "{" in result or "}" in result or not result.strip():
        raise ValueError("Invalid subgoal placeholder syntax")
    return result
