"""Public RoboMME GroundSG coordinate convention, without task templates."""

from __future__ import annotations

import re

POINT = re.compile(r"<\s*(\d+)\s*,\s*(\d+)\s*>")


def ground_subgoal(subgoal, image_shape, *, training_contract=False):
    """Convert front-view normalized <y, x> to pixels at the wire boundary."""
    matches = list(POINT.finditer(subgoal))
    if training_contract:
        from fds.common.vla_language import validate_wire

        validate_wire(subgoal, image_shape)
    if (
        not matches
        and not training_contract
        or "<" in POINT.sub("", subgoal)
        or ">" in POINT.sub("", subgoal)
    ):
        raise ValueError("GroundSG requires front-view <y, x> points normalized to 0..1000")
    height, width = image_shape[:2]

    def pixel(match):
        y, x = map(int, match.groups())
        if not 0 <= y <= 1000 or not 0 <= x <= 1000:
            raise ValueError("GroundSG coordinates must be in 0..1000")
        return (
            f"<{min(height - 1, int(y * height / 1000))}, {min(width - 1, int(x * width / 1000))}>"
        )

    simple = (
        re.sub(r" at\s*<\s*\d+\s*,\s*\d+\s*>", "", subgoal)
        if training_contract
        else POINT.sub("", subgoal)
    )
    return POINT.sub(pixel, subgoal), re.sub(r"\s+", " ", simple).strip()
