"""Explicit training provenance; immutable skill guidance for each episode."""

from copy import deepcopy

import jsonschema

from fds.common.schema import array, enum, obj, text
from fds.support.memory import ID

SOURCE = obj(
    benchmark=enum("robomme", "robomme_plus"),
    split=enum("train"),
    task=text(100),
    episode={"type": "integer", "minimum": 0},
)
SKILLS = array(
    obj(
        id=ID,
        applicability=text(500),
        instruction=text(2000),
        sources={**array(SOURCE, 16), "minItems": 1},
    ),
    32,
)


def validate_skills(skills):
    jsonschema.validate(skills, SKILLS)
    if len({s["id"] for s in skills}) != len(skills):
        raise ValueError("Skill IDs must be unique")
    return deepcopy(skills)


def validate_skill_changes(before, after, evidence):
    """New/changed guidance must cite completed development episodes supplied to RSI."""
    validate_skills(after)
    old = {s["id"]: s for s in before}
    allowed = {
        (e["benchmark"], e["split"], e["task"], e["episode"])
        for e in evidence
        if e["role"] == "development"
        and e["split"] == "train"
        and e.get("summary", {}).get("stop_reason")
        in ("environment_done", "planner_finish", "budget_exhausted", "error")
    }
    for skill in after:
        if old.get(skill["id"]) == skill:
            continue
        for source in skill["sources"]:
            key = tuple(source[k] for k in ("benchmark", "split", "task", "episode"))
            if key not in allowed:
                raise ValueError(
                    "Skill changes must cite the proposal's completed training evidence"
                )
