"""Tool classification and one dispatcher for environment and auxiliary batches."""

from copy import deepcopy
from dataclasses import dataclass, field

from fds.task_harness.geometry import check_image_correspondence, project_pixel_to_plane


@dataclass(frozen=True)
class ToolSpec:
    kind: str
    handler: str
    advances_time: bool = False
    ends_episode: bool = False
    exclusive: bool = False


TOOLS = {
    "wait": ToolSpec("environment", "_environment", advances_time=True),
    "move_ee": ToolSpec("environment", "_environment", advances_time=True),
    "finish": ToolSpec("environment", "_finish", ends_episode=True),
    "review_frames": ToolSpec("auxiliary", "_review", exclusive=True),
    "project_pixels_to_plane": ToolSpec("auxiliary", "_project"),
    "check_image_correspondence": ToolSpec("auxiliary", "_correspondence"),
}


def validate_batch(batch):
    """Check the declared kind against actual tool effects and sequencing constraints."""
    tool_calls = batch["tool_calls"]
    observation_steps = batch.get("observation_steps", [])
    if not tool_calls:
        raise ValueError("A Sys2 decision must contain at least one tool call")
    try:
        specs = [TOOLS[t["name"]] for t in tool_calls]
    except KeyError as exc:
        raise ValueError("Unknown task_harness tool") from exc
    if any(s.ends_episode for s in specs[:-1]):
        raise ValueError("finish must be the last tool in a batch")
    for tool, spec in zip(tool_calls, specs, strict=True):
        if spec.exclusive and len(tool_calls) != 1:
            raise ValueError(f"{tool['name']} must be the only tool in its auxiliary batch")
    if len({s.kind for s in specs}) != 1:
        raise ValueError("Environment and auxiliary tools must be in separate Sys2 decisions")
    if batch["kind"] != specs[0].kind:
        raise ValueError("Batch kind must match the registered tool kind")
    if not any(s.advances_time for s in specs) and observation_steps:
        raise ValueError("Tools that do not advance time require observation_steps=[]")
    return specs[0].kind


@dataclass
class ToolBatch:
    call: int
    kind: str
    subgoal: str | None
    start_step: int
    start_frame: int
    budget: int
    requested: set
    receipts: list = field(default_factory=list)
    sampled: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    stop_reason: str | None = None


class ToolSession:
    """Episode-local execution state. Auxiliary handlers have no environment-step path."""

    def __init__(
        self,
        env,
        actions,
        observation,
        pending,
        window,
        audit,
        log,
        *,
        max_steps,
        max_tool_steps,
        max_observation_frames,
        geometry,
    ):
        self.env, self.actions, self.observation, self.pending = env, actions, observation, pending
        self.window, self.audit, self.log = window, audit, log
        self.max_steps, self.max_tool_steps = max_steps, max_tool_steps
        self.max_observation_frames, self.geometry = max_observation_frames, geometry
        self.steps, self.status, self.phase = 0, "unknown", "initial"
        self.delivered, self.review_history = set(), []
        self.review_result = self.sampling = self.latest_execution = None
        self.last_result = self.last_environment_result = None
        self.motion_history = []
        self.environment_batches = 0
        self.batch_counts = {"environment": 0, "auxiliary": 0}

    def remaining(self):
        value = self.max_steps - self.steps
        if hasattr(self.env, "remaining_execution_steps"):
            value = min(value, self.env.remaining_execution_steps())
        return max(0, value)

    def expire(self, reason):
        self.log(
            {
                "event": "observation_window_expired",
                "window": self.window.describe(),
                "reason": reason,
            }
        )
        self.window.expire()
        self.review_history = []

    def execute(self, decision, call, retained):
        request = decision.tool_batch
        kind = validate_batch(request)
        tool_calls = request["tool_calls"]
        batch = ToolBatch(
            call,
            kind,
            request.get("subgoal"),
            self.steps,
            self.observation.current.index,
            min(self.max_tool_steps, self.remaining()),
            set(request.get("observation_steps", [])),
        )
        self.latest_execution = self.review_result = None
        self.batch_counts[kind] += 1
        explicit = sum(
            t["arguments"]["steps"]
            if t["name"] == "wait"
            else sum(p["steps"] for p in t["arguments"]["targets"] if p["steps"] != "auto")
            if t["name"] == "move_ee"
            else 0
            for t in tool_calls
        )
        if explicit > batch.budget:
            receipt = {
                "status": "rejected",
                "error": "Aggregate explicit steps exceed the batch budget",
                "executed_steps": 0,
            }
            batch.receipts.append(receipt)
            self.log({"event": "rejected_action", **receipt})
        else:
            for index, tool in enumerate(tool_calls):
                spec = TOOLS[tool["name"]]
                receipt, halt = getattr(self, spec.handler)(tool, batch, retained)
                receipt = {
                    "tool_index": index,
                    "tool": tool["name"],
                    "kind": spec.kind,
                    "semantic_success": "unverified",
                    **receipt,
                }
                batch.receipts.append(receipt)
                self.log({"event": "tool_result", "call": call, **receipt})
                if halt:
                    break
        executed = self.steps - batch.start_step
        if kind == "environment":
            selected = {f.index: f for _, f in sorted(batch.sampled.items())}
            selected[self.observation.current.index] = self.observation.current
            self.pending, self.phase = list(selected.values()), "execution"
            self.sampling = {
                "requested_steps": sorted(batch.requested),
                "resolved": [
                    {"step_offset": n, "frame": f.index} for n, f in sorted(batch.sampled.items())
                ],
                "unavailable_steps": sorted(batch.requested - batch.sampled.keys()),
                "final_frame": self.observation.current.index,
                "executed_steps": executed,
            }
            self.log({"event": "observation_selection", **self.sampling})
            if executed:
                self.latest_execution = {
                    "segment_id": f"call_{call:04d}",
                    "subgoal": batch.subgoal,
                    "tool_calls": deepcopy(tool_calls),
                    "actions": batch.actions,
                    "executed_steps": executed,
                    "start_frame": batch.start_frame,
                    "end_frame": self.observation.current.index,
                }
        self.last_result = {
            "request_call": call,
            "kind": kind,
            **({"subgoal": batch.subgoal} if kind == "environment" else {}),
            "tools": batch.receipts,
            "executed_steps": executed,
            "environment_advanced": executed > 0,
            "ends_episode": batch.stop_reason is not None,
            "environment_steps_before": batch.start_step,
            "environment_steps_after": self.steps,
            "observation_selection": deepcopy(self.sampling) if kind == "environment" else None,
            "semantic_success": "unverified",
        }
        if executed:
            self.last_environment_result = deepcopy(self.last_result)
        return batch

    def _finish(self, tool, batch, retained):
        arguments = tool["arguments"]
        batch.stop_reason = "planner_finish"
        return {"status": "accepted", "reason": arguments["reason"], "executed_steps": 0}, True

    def _review(self, tool, batch, retained):
        arguments = tool["arguments"]
        receipt = {
            "tool": "review_frames",
            "requested_window_id": arguments["window_id"],
            "requested_frames": deepcopy(arguments["frame_ids"]),
            "executed_steps": 0,
            "request_call": batch.call,
        }
        try:
            self.pending = self.window.review(
                arguments["window_id"],
                arguments["frame_ids"],
                self.observation.current,
                self.max_observation_frames,
            )
            requested = set(arguments["frame_ids"])
            accepted = [r for r in self.review_history if r["status"] == "accepted"]
            receipt.update(
                status="accepted",
                selected_frames=[f.index for f in self.pending],
                new_requested_frame_count=len(requested - self.delivered),
                same_frame_set_request_count=1
                + sum(set(r["requested_frames"]) == requested for r in accepted),
                accepted_review_count=len(accepted) + 1,
            )
        except ValueError as exc:
            self.pending = [self.observation.current]
            receipt.update(
                status="rejected", error=str(exc), selected_frames=[self.observation.current.index]
            )
        self.phase, self.review_result = "review", deepcopy(receipt)
        self.review_history.append(deepcopy(receipt))
        self.log({"event": "observation_review", **receipt})
        return receipt, receipt["status"] == "rejected"

    def _frame_geometry(self, frame_id, retained, *, active_only=False):
        if type(frame_id) is not int:
            raise ValueError("Geometry requires an episode frame with public calibration")
        if not active_only and frame_id in retained:
            frame = retained[frame_id]
            return deepcopy(frame.calibration), {
                c: [im.shape[1], im.shape[0]] for c, im in frame.images.items()
            }
        return self.window.geometry_metadata(frame_id, self.delivered)

    def _correspondence(self, tool, batch, retained):
        arguments = tool["arguments"]
        self.pending, self.phase = [self.observation.current], "auxiliary"
        try:
            if arguments["window_id"] != self.window.window_id or self.window.window_id is None:
                raise ValueError("Observation window expired or unknown")
            calibration, sizes = self._frame_geometry(
                arguments["frame_id"], retained, active_only=True
            )
            result = check_image_correspondence(calibration, sizes, arguments)
            return {"status": "accepted", "executed_steps": 0, "result": result}, False
        except ValueError as exc:
            return {"status": "rejected", "executed_steps": 0, "error": str(exc)}, True

    def _project(self, tool, batch, retained):
        arguments = tool["arguments"]
        self.pending, self.phase = [self.observation.current], "auxiliary"
        results = []
        for index, query in enumerate(arguments["queries"]):
            try:
                if not self.geometry or "base_to_world" not in self.geometry:
                    raise ValueError("Public base_to_world transform is unavailable")
                calibration, sizes = self._frame_geometry(query["frame_id"], retained)
                result = project_pixel_to_plane(
                    calibration, sizes, self.geometry["base_to_world"], query
                )
                results.append({"query_index": index, "status": "accepted", "result": result})
            except ValueError as exc:
                results.append({"query_index": index, "status": "rejected", "error": str(exc)})
        return {"status": "accepted", "executed_steps": 0, "queries": results}, False

    def _environment(self, tool, batch, retained):
        # This is the only handler allowed to step the environment.
        index = len(batch.receipts)
        name = tool["name"]
        available = min(batch.budget - (self.steps - batch.start_step), self.remaining())
        if available <= 0:
            return {
                "status": "rejected",
                "error": "Batch or environment budget exhausted",
                "executed_steps": 0,
            }, True
        try:
            chunk, details = self.actions.prepare(
                tool, self.observation, {f.index: f.cue for f in self.pending}, available, self.log
            )
        except ValueError as exc:
            receipt = {
                "status": "rejected",
                "tool_index": index,
                "tool": name,
                "attempted_tool_call": tool,
                "error": str(exc),
                "executed_steps": 0,
            }
            self.log({"event": "rejected_action", **receipt})
            return receipt, True
        executed, feedback, result, halt = 0, {}, None, False
        endpoint_steps, elapsed = set(), 0
        for target in details.get("resolved_targets", []):
            elapsed += target["steps"]
            if target["subtarget_index"] == target["subtarget_count"] - 1:
                endpoint_steps.add(elapsed)
        target_observations = []
        for action in chunk:
            if self.remaining() <= 0:
                halt = True
                break
            if self.steps == batch.start_step:
                previous = self.window.describe()
                self.window.begin(f"batch_{batch.call:04d}", "execution")
                self.review_history = []
                self.window.add(self.observation.current, 0)
                self.environment_batches += 1
                self.log(
                    {
                        "event": "observation_window_expired",
                        "window": previous,
                        "next_window_id": self.window.window_id,
                    }
                )
            result = self.env.step(action)
            self.actions.applied(action)
            self.steps += 1
            executed += 1
            batch.actions.append(action.tolist())
            self.status = result.status
            self.log(
                {
                    "event": "step",
                    "step": self.steps,
                    "tool_index": index,
                    "action": action.tolist(),
                    "scorer_status": self.status,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                }
            )
            if result.observation is not None:
                self.observation = result.observation
                self.audit.capture(self.observation.frames, self.steps)
                offset = self.steps - batch.start_step
                for frame in self.observation.frames:
                    self.window.add(frame, offset)
                if offset in batch.requested:
                    batch.sampled[offset] = deepcopy(self.observation.current)
                if executed in endpoint_steps:
                    tracking = self.actions.feedback(details, self.observation, executed).get(
                        "tracking"
                    )
                    if tracking is not None:
                        target_observations.append({
                            **tracking,
                            "tool_step_offset": executed,
                            "frame": self.observation.current.index,
                            "measured_tcp_pose": self.actions.kinematics.pose(
                                self.observation.current.state[:7]
                            ),
                        })
            feedback = dict(result.feedback)
            if result.done:
                batch.stop_reason, halt = "environment_done", True
                break
            if result.observation is None:
                raise RuntimeError("Environment returned no observation without terminating")
            if feedback.get("action_error") or feedback.get("trial_reset"):
                halt = True
                break
        receipt = {
            "status": "executed",
            "tool_index": index,
            "tool": name,
            "executed_steps": executed,
            "semantic_success": "unverified",
            "environment_feedback": feedback,
            **self.actions.feedback(
                details, self.observation if result and result.observation else None, executed
            ),
        }
        if name == "move_ee":
            receipt["target_observations"] = target_observations
            self.motion_history.append({
                "call": batch.call,
                "tool_index": index,
                "environment_steps_after": self.steps,
                "requested_targets": deepcopy(details.get("requested_targets", [])),
                "target_observations": deepcopy(target_observations),
                "final_tracking": deepcopy(receipt.get("tracking")),
            })
            self.motion_history = self.motion_history[-6:]
        self.log({"event": "execution_outcome", **receipt})
        return receipt, halt
