"""Model-assessed outcomes, mechanically bound to actions and event-memory revisions.

No evaluator state, extra images or model calls enter this protocol. A valid
citation establishes provenance, not the truth of a visual interpretation.
"""

import hashlib
import json
from copy import deepcopy

from fds.common.schema import array, enum, obj, text
from fds.task_harness.fds import FDS_FRAME, FDS_ID
from fds.task_harness.tools import TOOLS


def _unique(values, description):
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {description}")


def response_fields(context):
    fields = {
        "action_binding": obj(
            candidate_id=FDS_ID,
            tool_indices=array({"type": "integer", "minimum": 0}, 16),
            event_ids=array(FDS_ID, 32),
            event_dependencies=array(obj(
                event_id=FDS_ID, revision={"type": "integer", "minimum": 1},
            ), 32),
        ),
        "outcome_review": obj(
            action_call={"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
            checks=array(obj(
                id=FDS_ID,
                status=enum("met", "violated", "unknown"),
                basis=enum("visual", "public_feedback", "unknown"),
                evidence_refs=array(FDS_FRAME, 8),
                receipt_indices=array({"type": "integer", "minimum": 0}, 16),
                reason=text(600),
            ), 22),
            event_resolutions=array(obj(
                event_id=FDS_ID,
                status=enum("succeeded", "failed", "unknown"),
                check_ids=array(FDS_ID, 22),
            ), 32),
        ),
    }
    pending = context.get("pending_outcome_review")
    review = fields["outcome_review"]["properties"]
    if pending is None:
        review["action_call"] = {"type": "null"}
        review["checks"]["maxItems"] = review["event_resolutions"]["maxItems"] = 0
    else:
        ids = [check["id"] for check in pending["checks"]]
        review["action_call"] = {"type": "integer", "enum": [pending["action_call"]]}
        review["checks"]["items"]["properties"]["id"] = enum(*ids)
        review["checks"]["maxItems"] = len(ids)
        resolutions = review["event_resolutions"]
        resolutions["maxItems"] = len(pending["event_ids"])
        resolutions["items"]["properties"]["check_ids"]["items"] = enum(*ids)
        resolutions["items"]["properties"]["check_ids"]["maxItems"] = max(
            (len(ids) for ids in pending["event_checks"].values()), default=22
        )
        if pending["event_ids"]:
            resolutions["items"]["properties"]["event_id"] = enum(*pending["event_ids"])
    return fields


RULES = {
    "version": "fds_postcheck_v1",
    "assessment_scope": "All semantic reviews are model_assessed, not independent visual proof. "
        "Official status and hidden evaluator state are unavailable.",
    "review": "First review every check in pending_outcome_review, including through auxiliary "
        "calls. met means the predicate occurred; violated means it did not; unknown means "
        "insufficient evidence. For a falsifier, met challenges the prediction. Cite supplied "
        "post-action frames or the pending action's public receipts. Command execution alone "
        "cannot establish an expected physical outcome. Unresolved events retain their original "
        "predicates across later wait/observation actions. Later supplied images may clarify an "
        "earlier outcome; distinguish delayed evidence from a new occurrence caused by a later "
        "action. A check may copy its last_assessment unchanged as historical text provenance; "
        "this does not restore images or constitute a fresh verification. Changed assessments "
        "need currently supplied evidence. Do not promote unknown to pass validation.",
    "binding": "action_binding.candidate_id must be selected_candidate; tool_indices must list "
        "every tool index in execution order. It binds that candidate to these exact tool_calls, "
        "whose canonical hash is recorded by the host. event_ids names execution-memory events "
        "whose outcomes this new physical action will assess; create them as active/planned/unknown "
        "in memory_changes first. Keep distinct physical attempts distinct. Use [] for auxiliary/finish.",
    "memory": "Resolve every event bound to the pending action with event_resolutions and a matching "
        "memory_changes.events upsert. Preserve identity and criteria, cite the review evidence, and "
        "revise progress honestly. Success needs a met expected predicate grounded in post-action "
        "images; failure needs a violated expectation or met falsifier. Unknown stays unknown. "
        "A bound event inherits the predicates listed in event_checks: cite all of that event's "
        "checks, not unrelated later checks. Do not rebind an unresolved event to a new action. "
        "An explicit unknown resolution plus delete may discard a memory event under capacity "
        "pressure; it does not resolve its physical outcome. Success needs "
        "every expectation met and every falsifier violated; an unknown predicate keeps the event "
        "unknown. New or changed execution-event success/failure requires such a review. This does not "
        "verify arbitrary free-text counts or independently identify objects.",
    "dependencies": "List events whose confirmed occurrence the selected action depends on, with "
        "their current event_memory_versions revision. A dependency must remain succeeded after "
        "this review, even for a tentative candidate. Correcting a dependency in this response "
        "requires its event_resolution; omission is not a correction. Unknown does not prohibit "
        "all movement: choose a bounded discriminating action that does not assume the event occurred.",
    "budget": "Reviews share the normal response and correction budget. No extra verifier call. "
        "finish is always a model choice; it does not certify success. No mandatory final call "
        "is added after environment termination or budget exhaustion.",
}


class OutcomeTracker:
    def __init__(self):
        self.pending = None
        self.event_versions = {}
        self.resolutions = {}
        self.stats = {"reviews": 0, "met": 0, "violated": 0, "unknown": 0,
                      "event_succeeded": 0, "event_failed": 0, "event_unknown": 0}

    def context(self, memory):
        return {
            "outcome_protocol": deepcopy(RULES),
            "pending_outcome_review": deepcopy(self.pending),
            "event_memory_versions": [
                {"event_id": rid, "revision": self.event_versions[rid],
                 "last_assessment": deepcopy(self.resolutions.get(rid))}
                for rid in memory.memory["events"]
            ],
        }

    def validate(self, decision, before, after, available_frames):
        """Pure validation. The caller commits memory and review only on acceptance."""
        contract, batch = decision.fds_contract, decision.tool_batch
        binding, review = contract["action_binding"], contract["outcome_review"]
        candidate = next(c for c in contract["candidates"] if c["id"] == contract["selected_candidate"])
        if any(not predicate.strip() for item in (contract, candidate)
               for predicate in [item["expected_outcome"], *item["falsifiers"]]):
            raise ValueError("Expected outcomes and declared falsifiers must be nonempty predicates")
        if binding["candidate_id"] != contract["selected_candidate"]:
            raise ValueError("action_binding must name selected_candidate")
        if binding["tool_indices"] != list(range(len(batch["tool_calls"]))):
            raise ValueError("action_binding.tool_indices must bind every actual tool in order")
        _unique(binding["event_ids"], "bound event IDs")
        _unique([d["event_id"] for d in binding["event_dependencies"]], "event dependencies")
        moves = any(TOOLS[t["name"]].advances_time for t in batch["tool_calls"])
        if binding["event_ids"] and not moves:
            raise ValueError("Auxiliary/finish-only batches cannot register physical event outcomes")
        checks = review["checks"]
        resolutions = review["event_resolutions"]
        _unique([c["id"] for c in checks], "outcome check IDs")
        _unique([r["event_id"] for r in resolutions], "event resolutions")
        if self.pending is None:
            if review != {"action_call": None, "checks": [], "event_resolutions": []}:
                raise ValueError("No prior physical action is available for outcome review")
            predicates = {}
        else:
            pending = self.pending
            if review["action_call"] != pending["action_call"]:
                raise ValueError("outcome_review must refer to the pending physical action")
            predicates = {p["id"]: p for p in pending["checks"]}
            if set(predicates) != {c["id"] for c in checks}:
                raise ValueError("Review every pending expected outcome and falsifier exactly once")
            if set(pending["event_ids"]) != {r["event_id"] for r in resolutions}:
                raise ValueError("Resolve every event bound to the pending action exactly once")
            available = set(available_frames)
            for check in checks:
                refs, receipts = check["evidence_refs"], check["receipt_indices"]
                _unique(refs, "outcome evidence refs")
                _unique(receipts, "receipt indices")
                predicate = predicates[check["id"]]
                if check == predicate.get("last_assessment"):
                    continue
                if not set(refs) <= available:
                    raise ValueError("Outcome evidence must be supplied or retained in this request")
                if not set(receipts) <= set(predicate["receipt_indices"]):
                    raise ValueError("Outcome receipt index does not belong to this check's action")
                fresh = any(type(f) is int and predicate["action_start_frame"] < f <= pending["end_frame"]
                            for f in refs)
                if check["status"] != "unknown":
                    if check["basis"] == "visual" and not fresh:
                        raise ValueError("Visual outcome assessment requires a supplied post-action frame")
                    if check["basis"] == "public_feedback" and not receipts:
                        raise ValueError("Public-feedback assessment requires an explicit receipt")
                    if check["basis"] == "unknown":
                        raise ValueError("An assessment without evidence must remain unknown")
                    if predicates[check["id"]]["kind"] == "expected" and check["status"] == "met":
                        if (check["basis"] != "visual" or not fresh
                                or not predicate["executed_steps"]):
                            raise ValueError("An execution receipt alone cannot confirm an expected outcome")
        assessed = {c["id"]: c for c in checks}
        resolved = {r["event_id"]: r for r in resolutions}
        changed = {c["record"]["id"]: c["record"] for c in decision.memory_changes["events"]
                   if c["op"] == "upsert"}
        deleted = {c["id"] for c in decision.memory_changes["events"] if c["op"] == "delete"}
        for rid, resolution in resolved.items():
            ids = resolution["check_ids"]
            _unique(ids, "event resolution check IDs")
            if not ids or set(ids) != set(self.pending["event_checks"][rid]):
                raise ValueError("Event resolution must cite every predicate bound to its action")
            if rid in deleted and rid not in after.memory["events"] and resolution["status"] == "unknown":
                continue
            row = changed.get(rid)
            if (row is None or after.memory["events"].get(rid) != row
                    or row["origin"] != "execution" or row["status"] != resolution["status"]):
                raise ValueError("Event resolution requires an accepted matching execution-event upsert")
            old = before.memory["events"].get(rid)
            if old and (row["objective"], row["success_criteria"]) != (old["objective"], old["success_criteria"]):
                raise ValueError("Outcome review cannot replace an event's objective or success criteria")
            supported = all(
                assessed[i]["status"] == ("met" if predicates[i]["kind"] == "expected" else "violated")
                for i in ids
            )
            contradicted = any(
                (predicates[i]["kind"] == "expected" and assessed[i]["status"] == "violated")
                or (predicates[i]["kind"] == "falsifier" and assessed[i]["status"] == "met")
                for i in ids
            )
            if resolution["status"] == "succeeded" and (not supported or contradicted):
                raise ValueError("Event success needs a met expectation without a cited contradiction")
            if resolution["status"] == "failed" and not contradicted:
                raise ValueError("Event failure needs a violated expectation or met falsifier")
            evidence = {f for i in ids for f in assessed[i]["evidence_refs"]}
            if resolution["status"] == "succeeded" and not evidence.intersection(row["evidence_frames"]):
                raise ValueError("Successful event memory must retain the outcome review's evidence")
        for rid, row in after.memory["events"].items():
            previous = before.memory["events"].get(rid)
            if previous and previous["origin"] == "execution" and row["origin"] != "execution":
                raise ValueError("An execution event cannot be relabelled as demonstration/instruction")
            if row != previous and row["origin"] == "execution" and row["status"] in ("succeeded", "failed"):
                if rid not in resolved:
                    raise ValueError("Changed execution-event success/failure needs an outcome review")
        for rid in binding["event_ids"]:
            if rid in resolved and resolved[rid]["status"] == "unknown":
                raise ValueError("Keep unresolved event predicates; do not rebind the event to a new action")
            row = after.memory["events"].get(rid)
            if row is None or row["origin"] != "execution" or row["status"] not in ("planned", "active", "unknown"):
                raise ValueError("Bind physical outcomes to unresolved execution-memory events")
        for dependency in binding["event_dependencies"]:
            rid = dependency["event_id"]
            if rid not in before.memory["events"] or self.event_versions.get(rid) != dependency["revision"]:
                raise ValueError("Event dependency names an unavailable or stale memory revision")
            row = after.memory["events"].get(rid)
            if row is None or row["origin"] != "execution" or row["status"] != "succeeded":
                raise ValueError("Action depends on an event whose occurrence is not confirmed")
            if row != before.memory["events"][rid] and rid not in resolved:
                raise ValueError("A changed event dependency requires an explicit outcome reconciliation")

    def accept_review(self, decision, before, after, call):
        review = deepcopy(decision.fds_contract["outcome_review"])
        for rid in before.memory["events"].keys() | after.memory["events"].keys():
            if before.memory["events"].get(rid) != after.memory["events"].get(rid):
                self.event_versions[rid] = self.event_versions.get(rid, 0) + 1
            if rid not in after.memory["events"]:
                self.resolutions.pop(rid, None)
        if self.pending is not None:
            self.stats["reviews"] += 1
            checks = {c["id"]: c for c in self.pending["checks"]}
            for check in review["checks"]:
                self.stats[check["status"]] += 1
                checks[check["id"]]["last_assessment"] = deepcopy(check)
            self.pending["last_review"] = {"call": call, **deepcopy(review)}
        for resolution in review["event_resolutions"]:
            rid = resolution["event_id"]
            if rid in after.memory["events"]:
                self.resolutions[rid] = {"review_call": call,
                                         "revision": self.event_versions[rid], **deepcopy(resolution)}
            self.stats["event_" + resolution["status"]] += 1
        if self.pending is not None:
            # Terminal/deleted events no longer require repeated memory upserts.
            # Keep the most recent action's checks, and older checks only while
            # an unresolved event still depends on them.
            self.pending["event_checks"] = {
                rid: ids for rid, ids in self.pending["event_checks"].items()
                if rid in after.memory["events"]
                and self.resolutions.get(rid, {}).get("status") == "unknown"
            }
            self.pending["event_ids"] = list(self.pending["event_checks"])
            needed = {cid for ids in self.pending["event_checks"].values() for cid in ids}
            self.pending["checks"] = [
                check for check in self.pending["checks"]
                if check["id"] in needed or check["action_call"] == self.pending["action_call"]
            ]
        return {"assessment_scope": "model_assessed", "review_call": call, **review}

    def register_action(self, decision, batch, end_frame):
        if not any(TOOLS[t["name"]].advances_time for t in decision.tool_batch["tool_calls"]):
            return None
        contract = decision.fds_contract
        candidate = next(c for c in contract["candidates"] if c["id"] == contract["selected_candidate"])
        # Keep only unresolved events still in memory, plus the latest action.
        # The memory event cap bounds retained action sources; full history stays
        # in audit files and is never retrieved as fresh model context.
        carried_events = {
            rid: list(ids) for rid, ids in (self.pending or {}).get("event_checks", {}).items()
            if self.resolutions.get(rid, {}).get("status") == "unknown"
        }
        carried_ids = {cid for ids in carried_events.values() for cid in ids}
        checks = [deepcopy(c) for c in (self.pending or {}).get("checks", []) if c["id"] in carried_ids]
        receipt_ids = sorted({i for c in checks for i in c["receipt_indices"]})
        receipts = [deepcopy(self.pending["receipts"][i]) for i in receipt_ids]
        renumber = {old: new for new, old in enumerate(receipt_ids)}
        for check in checks:
            check["receipt_indices"] = [renumber[i] for i in check["receipt_indices"]]
            if "last_assessment" in check:
                check["last_assessment"]["receipt_indices"] = [
                    renumber[i] for i in check["last_assessment"]["receipt_indices"]
                ]
        current_receipts = list(range(len(receipts), len(receipts) + len(batch.receipts)))
        receipts.extend({"request_call": batch.call, **deepcopy(r)} for r in batch.receipts)
        current_checks, seen = [], set()
        for source, item in (("contract", contract), ("selected_candidate", candidate)):
            for kind, predicate in [("expected", item["expected_outcome"]),
                                    *[("falsifier", p) for p in item["falsifiers"]]]:
                if (kind, predicate) in seen:
                    continue
                seen.add((kind, predicate))
                current_checks.append({
                    "id": f"call_{batch.call}_check_{len(current_checks)}", "kind": kind,
                    "predicate": predicate, "source": source, "action_call": batch.call,
                    "action_start_frame": batch.start_frame, "action_end_frame": end_frame,
                    "executed_steps": sum(r.get("executed_steps", 0) for r in batch.receipts),
                    "receipt_indices": current_receipts,
                })
        checks.extend(current_checks)
        event_checks = {**carried_events, **{
            rid: [c["id"] for c in current_checks]
            for rid in contract["action_binding"]["event_ids"]
        }}
        canonical = json.dumps(decision.tool_batch, sort_keys=True, separators=(",", ":"), allow_nan=False)
        sources = [deepcopy(source) for source in (self.pending or {}).get("action_sources", [])
                   if source["action_call"] in {c["action_call"] for c in checks}]
        sources.append({"action_call": batch.call, "contract_id": contract["contract_id"],
                        "candidate_id": candidate["id"],
                        "tool_batch_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                        "tool_calls": deepcopy(decision.tool_batch["tool_calls"]),
                        "start_frame": batch.start_frame, "end_frame": end_frame})
        self.pending = {
            "action_call": batch.call,
            "contract_id": contract["contract_id"],
            "candidate_id": candidate["id"],
            "tool_batch_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "tool_calls": deepcopy(decision.tool_batch["tool_calls"]),
            "start_frame": batch.start_frame,
            "end_frame": end_frame,
            "receipts": receipts,
            "event_ids": list(event_checks),
            "event_checks": event_checks,
            "action_sources": sources,
            "checks": checks,
            "last_review": None,
        }
        return deepcopy(self.pending)

    def summary(self):
        return {"assessment_scope": "model_assessed", **self.stats,
                "pending_action_call": self.pending["action_call"] if self.pending else None,
                "pending_action_reviewed": bool(self.pending and self.pending["last_review"])}
