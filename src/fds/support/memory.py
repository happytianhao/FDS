"""Explicit opt-in model-owned memory for direct Sys2 control, not source118."""

import json
from copy import deepcopy
from dataclasses import dataclass

import jsonschema

from fds.common.schema import array, enum, obj, text

ID = {"type": "string", "minLength": 1, "maxLength": 64, "pattern": "^[a-zA-Z0-9_-]+$"}
REFS = array({"type": "integer", "minimum": 0}, 8)
MEMORY = obj(
    events=array(
        obj(
            id=ID,
            objective=text(400),
            success_criteria=text(400),
            status=enum("planned", "active", "succeeded", "failed", "unknown"),
            origin=enum("instruction", "demonstration", "execution"),
            progress=text(500),
            evidence_frames=REFS,
        ),
        24,
    ),
    objects=array(
        obj(
            id=ID,
            identity=text(400),
            location=text(400),
            motion=text(300),
            last_seen_frame={"type": "integer", "minimum": -1},
            evidence_frames=REFS,
        ),
        16,
    ),
    skills=array(obj(id=ID, condition=text(300), advice=text(500), evidence_frames=REFS), 8),
)
FRAME = obj(frame_id={"type": "integer", "minimum": 0}, description=text(300), reason=text(300))


def schema_for(context):
    from fds.support.protocol import schema_for as action_schema

    return obj(
        working_memory=deepcopy(MEMORY),
        retained_frames=array(
            FRAME, context.get("memory_contract", {}).get("max_retained_frames", 4)
        ),
        tool_call=action_schema(context)["properties"]["tool_call"],
    )


@dataclass
class MemoryDecision:
    working_memory: dict
    retained_frames: list
    tool_call: dict

    @property
    def tool(self):
        return self.tool_call["name"]

    @classmethod
    def parse(cls, value, schema):
        jsonschema.validate(value, schema)
        json.dumps(value, allow_nan=False)
        return cls(**deepcopy(value))


class WorkingMemory:
    def __init__(self, max_frames=4):
        self.max_frames = max_frames
        self.memory = {group: [] for group in ("events", "objects", "skills")}
        self.frames = {}
        self.catalog = []
        self.last_validation = []

    def state(self):
        return deepcopy({"memory": self.memory, "retained": {"catalog": self.catalog}})

    def contract(self):
        return {
            "version": "sys2_memory_v5",
            "max_retained_frames": self.max_frames,
            "snapshot": "Replace all three groups each turn; omitted entries are deleted. All semantics are model estimates.",
            "frames": "Select only a fresh frame delivered this turn or a currently retained frame. Omission deletes it permanently from model access.",
            "counts": "Executed commands and demonstration events are not confirmed physical occurrences.",
        }

    def update(self, decision, fresh, call, log):
        available = {**self.frames, **{f.index: f for f in fresh}}
        old_refs = {
            ref for rows in self.memory.values() for r in rows for ref in r["evidence_frames"]
        }
        legal_refs = set(available) | old_refs
        updated, results = {}, []
        for group, rows in decision.working_memory.items():
            previous = {r["id"]: r for r in self.memory[group]}
            accepted = {}
            for row in rows:
                error = None
                if row["id"] in accepted:
                    error = "Duplicate memory id"
                elif not set(row["evidence_frames"]) <= legal_refs:
                    error = "Reference was neither delivered nor preserved in existing memory"
                elif (
                    group == "objects"
                    and row["last_seen_frame"] != -1
                    and row["last_seen_frame"] not in legal_refs
                ):
                    error = "last_seen_frame is not an available provenance frame"
                if error:
                    if row["id"] in previous and row["id"] not in accepted:
                        accepted[row["id"]] = previous[row["id"]]
                    results.append(
                        {
                            "group": group,
                            "id": row["id"],
                            "status": "rejected",
                            "error": error,
                            "resolution": "kept_previous" if row["id"] in previous else "not_added",
                        }
                    )
                else:
                    accepted[row["id"]] = deepcopy(row)
            updated[group] = list(accepted.values())
        chosen, catalog = {}, []
        old_catalog = {r["frame_ids"][0]: r for r in self.catalog}
        for row in decision.retained_frames:
            fid = row["frame_id"]
            if fid not in available or fid in chosen:
                log(
                    {
                        "event": "retention_operation",
                        "operation": {"operation": "save", **row},
                        "rejection": "Frame unavailable or duplicate; deleted/unsampled history cannot be recovered",
                    }
                )
                continue
            chosen[fid] = available[fid]
            catalog.append(
                {
                    "key": f"frame_{fid}",
                    "frame_ids": [fid],
                    "segment_ids": [],
                    "saved_call": old_catalog.get(fid, {}).get("saved_call", call),
                    "description": row["description"],
                    "reason": row["reason"],
                }
            )
            if fid not in self.frames:
                log({"event": "retention_operation", "operation": {"operation": "save", **row}})
        for fid in set(self.frames) - set(chosen):
            log(
                {
                    "event": "retention_operation",
                    "operation": {"operation": "delete", "frame_id": fid},
                }
            )
        self.memory, self.frames, self.catalog = updated, chosen, catalog
        self.last_validation = results
        log({"event": "memory_operation", "record_results": results, "memory": self.memory})
        return results


def make_prompt(context):
    from fds.support.protocol import make_prompt as stateless_prompt

    # Share mechanical instructions, replacing only the stateless memory policy.
    base = stateless_prompt({**context, "decision_protocol": "sys2"})
    base = base[: base.index("\n")] if "\n" in base else base
    base = base.replace(
        "Each request is STATELESS.",
        "Each request has no implicit conversation; explicit memory is supplied.",
    )
    base = base.replace(
        "its cue is gone next turn", "save an important cue if later verification requires it"
    )
    base = base.replace(
        "do not depend on remembering its count or cue on a later call",
        "save the procedure, verified progress and unresolved checks in working_memory",
    )
    base = base.replace(
        "No old image is available after rejection",
        "Only explicitly retained images remain available after rejection",
    )
    native = context.get("benchmark_control") == "roboquest_native_v1"
    if native:
        from fds.benchmarks.roboquest.direct import DIRECT_PROMPT

        base = DIRECT_PROMPT
    return (
        base
        + (
            "\nYou control the official native ARM/BASE/WAIT tools, with explicitly enabled memory. "
            if native
            else "\nYou control the same direct absolute-TCP/IK tools, with explicitly enabled memory. "
        )
        + "Return working_memory (events, objects, skills), retained_frames, and one tool_call. "
        "Resolve directional words using task_language_conventions when supplied, before naming objects or "
        "recording an ordered route. Robot-relative left/right and camera pixel left/right are different "
        "references: retain the reference used in memory, and do not mirror measured image coordinates. "
        "Read the previous snapshot, verify the last action using NEW observations, then replace the complete snapshot. "
        "Preserve goal-relevant identity, ordered demonstration, confirmed progress, pending action and failed attempts. "
        "Distinguish planned actions, attempts, observed outcomes and uncertainty. Never count a demonstration as execution. "
        "Do not infer success from an IK result, elapsed time, call number or closed fingers. "
        "A completed waypoint is not necessarily a completed task event. Avoid repeating a verified occurrence. "
        "Remember which goal conditions remain before finish or any irreversible final control action. "
        "Write each event's observable success_criteria before attempting it; preserve these criteria through retries. "
        "Criteria must express the public goal, not extra requirements you invented. If a verification assumption "
        "proves unobservable or unjustified, explicitly revise that assumption while preserving the original goal. "
        "Distinguish an observed entry/transition from continued visibility afterward. An object may become hidden "
        "after entering an opening or receptacle: invisibility alone neither erases the observed event nor proves "
        "success. Retain the before/transition evidence and any contrary evidence, label the residual uncertainty, "
        "and avoid adding another counted action merely because later views are occluded. "
        "When repeated inspections add no information, use the preserved event evidence to choose the remaining "
        "goal action; do not replace the task with an endless demand for an inaccessible view. "
        "For repetition or timed-motion goals, use memory to track distinct observed transitions. Choose wait/action "
        "durations so the available fresh-frame sampling can resolve those transitions; shorten the observation "
        "interval near a predicted crossing. An entire unobserved cycle cannot be counted from elapsed steps. "
        "On later calls compare fresh changes against those criteria, not against your previous success claim. "
        "A stable object beside a destination, an empty gripper, or arrival at the requested TCP pose may fail the "
        "required functional relation. Specify what remains unobserved rather than upgrading such evidence to success. "
        "For demonstrated motions, preserve an ordered route and the orientation/working end at meaningful stages, "
        "including the outbound path and transition into contact, not just the final position. "
        "Robot demonstration poses are reference observations, not known object poses. Compare the current grasp "
        "offset and object geometry before copying them. Retain decisive cue frames for later comparison. "
        "After a placement miss, remember the attempted coordinates and observed deviation; change the geometric "
        "hypothesis using fresh views before repeating. Account for the held object's offset from TCP when aligning "
        "and releasing. Use a short observation boundary at uncertain contact or release rather than completing "
        "many dependent steps with an unverified premise. "
        "Separate acquisition, a diagnostic test lift, completed pickup, and supported release in event progress. "
        "A small lift may verify attachment without completing the requested pickup: make the held object's "
        "clearance above its support and neighboring clutter unambiguous, observing its bottom as well as TCP. "
        "Choose clearance from visible object scale and the goal, not an assumed benchmark threshold. "
        "Before changing from carrying an object to operating an unrelated control or acquiring another object, "
        "check the hand-occupancy precondition. Unless the goal calls for using the held object as a tool, "
        "place it stably on a clear support, open, withdraw, and verify that it stays before the new interaction. "
        "Keep any completed count and pending final action in memory across this handover. "
        "For control activation distinguish actual visible depression/rebound from merely touching the housing "
        "or an unrelated scene change. If measured motion stalls, record the contact hypothesis and correct "
        "lateral alignment from fresh views rather than repeatedly pushing deeper at the same point. "
        "Object positions in memory are dated estimates; revise them with current RGB. "
        "Retain only useful frames from this request or the current retained set; omitted frames become unavailable. "
        "Historical retained images are labeled and never represent current state. Keep keyframes few. "
        "All fresh RGB/proprioception and raw executed actions are delivered once per logical request. "
        "No other external files, conversation, internet or benchmark oracle is available. "
        + (
            "Robot proprioception and servo commands are distinct; read robot_state_layout. "
            if native
            else "Seven joint radians and one finger position in meters are distinct from gripper commands. "
        )
        + "Use the provided coordinate conventions, geometry, measured tracking, limits and remaining budgets.\n"
        + json.dumps(context, ensure_ascii=False, allow_nan=False)
    )
