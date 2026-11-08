"""Validated memory deltas. Audits are never a memory or image retrieval source."""

from collections import Counter
from copy import deepcopy

from fds.task_harness.protocol import LIMITS, RECORDS


class EpisodicMemory:
    def __init__(self, max_frames=4):
        self.max_frames = max_frames
        self.memory = {group: {} for group in RECORDS}
        self.frames, self.catalog = {}, {}
        self.last_validation = []

    def state(self):
        return deepcopy(
            {
                "memory": {group: list(rows.values()) for group, rows in self.memory.items()},
                "retained": {"catalog": list(self.catalog.values())},
            }
        )

    def contract(self):
        return {
            "version": "fds_memory_v1",
            "limits": deepcopy(LIMITS),
            "max_retained_frames": self.max_frames,
            "capacity": "null means no fixed record-count cap. Objects are uncapped; events and experiences use the listed limits.",
            "updates": "upsert replaces one ID; delete removes it. Omission or [] preserves records. Each ID may occur once per group per response. Rejected updates preserve prior records and are reported next turn; valid tools remain independent.",
            "provenance": "New frame references must be visible in this request, not merely listed in the review window. A record may keep its own prior references as text, without restoring pixels. last_seen_frame=-1 means unknown. Integer IDs belong to this episode; training:<episode>:<index> IDs are extra training-video evidence, never current state or current execution. References do not certify the model's interpretation.",
            "semantics": "Entries are model estimates, not official scores. Keep demonstrations, plans, execution and observed outcomes distinct. Positions are dated estimates; experiences come only from this episode.",
            "object_inventory": {
                "scope_choices": {
                    "all_visible": "Individually record every visually distinguishable scene object, including non-goal items; categories do not replace individual records.",
                    "selective": "Record selected objects relevant to goals, decisions or retained evidence; exhaustive scene coverage is not required.",
                },
                "selection": "Both scopes are available each turn. Skills guide the model's preference. Express choices through memory_changes.objects; these labels are not response fields or host modes. Changing scope never deletes records automatically; deletion requires an explicit operation.",
                "observations": "For retained objects, keep actual last_seen_frame and distinguish last observed state from uncertain current state. Do not invent unseen objects or exact locations; cue-only observations are not current state. Deleting a record does not assert the object's physical disappearance or resolve an event.",
            },
            "object_identity": "IDs refer to stable identities, never reassigned to unrelated objects. Same appearance alone does not prove a match. Record uncertain matches and inferred relations explicitly; alternative identities are not independent confirmed objects. Evidence-backed corrections must preserve affected event references.",
            "events": "Event IDs identify objectives or distinct attempts. Preserve attempt identity when revising an outcome; another physical attempt is a separate event. A zero-step rejection is not a physical occurrence. unknown means unresolved, neither failed nor succeeded. Command completion, disappearance or occlusion alone does not certify a physical outcome.",
            "keyframes": "save retains or relabels a supplied frame with original pixels, state if available, cue flag and episode/training origin; delete removes it. Saved frames remain historical. Frames still in the active review window may be explicitly reviewed again before acting, even if not previously sampled or no longer retained. Once that window expires, only retained frames remain accessible; audit never restores deleted or unsampled history. A review's requested frames arrive next turn, so they cannot be saved or newly cited until supplied. Selection is explicit and bounded by max_retained_frames, shared across episode and training frames.",
            "skills": "Frozen guidance from completed training attempts, read by the model. Applicability is not a host-side selector; guidance may be combined or reconsidered each turn within the contracts. Inventory preferences belong in skills; they do not restrict the available scope choices. Skills cannot override budgets, identity or provenance rules. This response cannot edit skills.",
        }

    def update(self, decision, visible, call, log):
        available = {**self.frames, **{f.index: f for f in visible}}
        results = []
        for group, changes in decision.memory_changes.items():
            limit = LIMITS[group]
            ids = [c["id"] if c["op"] == "delete" else c["record"]["id"] for c in changes]
            duplicates = Counter(ids)
            for change, rid in zip(changes, ids, strict=True):
                previous = self.memory[group].get(rid)
                error = None
                if duplicates[rid] > 1:
                    error = "Duplicate operations for the same ID"
                elif change["op"] == "upsert":
                    row = change["record"]
                    old_refs = set(previous["evidence_frames"]) if previous else set()
                    if previous and group == "objects":
                        old_refs.add(previous["last_seen_frame"])
                    legal = set(available) | old_refs
                    if not set(row["evidence_frames"]) <= legal:
                        error = "Evidence frame was not supplied and is not this record's prior provenance"
                    elif group == "objects" and row["last_seen_frame"] not in legal | {-1}:
                        error = "last_seen_frame has no available provenance"
                    elif (
                        previous is None and limit is not None and len(self.memory[group]) >= limit
                    ):
                        error = "Memory capacity reached; delete an entry before adding one"
                if error is None:
                    if change["op"] == "delete":
                        self.memory[group].pop(rid, None)
                    else:
                        self.memory[group][rid] = deepcopy(change["record"])
                results.append(
                    {
                        "group": group,
                        "id": rid,
                        "operation": change["op"],
                        "status": "rejected" if error else "accepted",
                        "error": error,
                        "resolution": "kept_previous"
                        if error and previous
                        else "not_added"
                        if error
                        else "applied",
                    }
                )
        counts = Counter(c["frame_id"] for c in decision.keyframe_changes)
        for change in decision.keyframe_changes:
            fid, error = change["frame_id"], None
            if counts[fid] > 1:
                error = "Duplicate keyframe operations"
            elif change["op"] == "save":
                if fid not in available:
                    error = "Frame not supplied or retained; review the active window first if still available"
                elif fid not in self.frames and len(self.frames) >= self.max_frames:
                    error = "Keyframe capacity reached; delete a frame before saving another"
            if error is None:
                if change["op"] == "delete":
                    self.frames.pop(fid, None)
                    self.catalog.pop(fid, None)
                else:
                    self.frames[fid] = deepcopy(available[fid])
                    self.catalog[fid] = {
                        "key": f"frame_{fid}",
                        "frame_ids": [fid],
                        "segment_ids": [],
                        "saved_call": self.catalog.get(fid, {}).get("saved_call", call),
                        "cue": available[fid].cue,
                        "origin": "training_example" if isinstance(fid, str) else "episode",
                        "description": change["description"],
                        "reason": change["reason"],
                    }
            log(
                {
                    "event": "retention_operation",
                    "operation": {"operation": change["op"], **change},
                    "rejection": error,
                }
            )
            results.append(
                {
                    "group": "keyframes",
                    "id": str(fid),
                    "status": "rejected" if error else "accepted",
                    "error": error,
                }
            )
        self.last_validation = results
        log(
            {
                "event": "memory_operation",
                "record_results": results,
                "memory": self.state()["memory"],
            }
        )
        return results
