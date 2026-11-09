"""Sys2-owned decisions; host-owned bounded memory, observations and mechanical execution."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from fds.common.audit import AuditJournal
from fds.common.episode import EpisodeIO
from fds.support.coordinates import robot_reference
from fds.support.observation import select_temporal_frames
from fds.task_harness.actions import TaskActions
from fds.task_harness.contracts import definitions, execution_contract
from fds.task_harness.fds import execution_verdict
from fds.task_harness.memory import EpisodicMemory
from fds.task_harness.observation import (
    ObservationWindow,
    group_images_by_camera,
    image_inputs,
)
from fds.task_harness.protocol import DecisionValidationError, TaskDecision, schema_for
from fds.task_harness.skills import validate_skills
from fds.task_harness.tools import ToolSession


class TaskHarness(EpisodeIO):
    tool_step_limit = 400

    def __init__(self, *args, options=None, examples=None, fds_options=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.policy is not None:
            raise ValueError("task_harness is a direct Sys2 executor and does not use a VLA")
        opts = deepcopy(options or {})
        self.max_keyframes = opts.pop("max_keyframes", 4)
        self.demo_frames = opts.pop("demo_frames", 16)
        self.initial_live_frames = opts.pop("initial_live_frames", 6)
        self.max_tool_calls = opts.pop("max_tool_calls", 8)
        self.max_pose_targets = opts.pop("max_pose_targets", 32)
        self.max_observation_frames = opts.pop("max_observation_frames", 12)
        self.max_decision_corrections = opts.pop("max_decision_corrections", 2)
        self.skills = validate_skills(opts.pop("skills", []))
        self.limits = opts.pop("pose_limits", {})
        self.examples = examples
        fds_options = deepcopy(fds_options or {})
        enabled = fds_options.pop("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("fds.enabled must be a boolean")
        self.contract_mode = fds_options.pop("contract_mode", "full")
        if self.contract_mode not in ("off", "full"):
            raise ValueError("fds.contract_mode must be 'off' or 'full'")
        if not enabled:
            self.contract_mode = "off"
        if fds_options:
            raise ValueError(f"Unknown fds options: {sorted(fds_options)}")
        for name, value, low, high in (
            ("max_keyframes", self.max_keyframes, 0, 8),
            ("demo_frames", self.demo_frames, 1, 32),
            ("initial_live_frames", self.initial_live_frames, 1, 12),
            ("max_tool_calls", self.max_tool_calls, 1, 16),
            ("max_pose_targets", self.max_pose_targets, 1, 64),
            ("max_observation_frames", self.max_observation_frames, 1, 32),
            ("max_decision_corrections", self.max_decision_corrections, 0, 5),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}: expected integer in {low}..{high}")
        if opts:
            raise ValueError(f"Unknown task_harness options: {sorted(opts)}")

    def run(self, episode, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "frames").mkdir()
        (output / "memory_versions").mkdir()
        audit = AuditJournal(output, {"enabled": True, "capture_all_frames": True})
        window = ObservationWindow(output)
        memory = EpisodicMemory(self.max_keyframes)
        skills = deepcopy(self.skills)
        AuditJournal.write(
            output / "skills.json", {"skills": skills, "mutable_during_episode": False}
        )
        calls, stop = 0, "budget_exhausted"
        last_fds_contract, last_fds_verdict = None, None
        rejection = None
        consecutive_rejections = rejected_decisions = correction_calls = recovered_decisions = 0
        summary, session = None, None

        def log(event):
            self._log(output, event)

        def state():
            return {
                "method": "task_harness",
                "contract_mode": self.contract_mode,
                "step": session.steps if session else 0,
                "memory_enabled": True,
                "current_frame": session.observation.current.index if session else None,
                "skills": deepcopy(skills),
                "observation_window": window.describe(),
                **memory.state(),
            }

        try:
            observation = self.env.reset(episode)
            window.begin("initial", "initial")
            for frame in observation.frames:
                window.add(frame)
            kinematics = self.env.get_kinematics()
            actions = TaskActions(self.env, None, kinematics, self.limits)
            geometry = (
                self.env.get_robot_geometry() if hasattr(self.env, "get_robot_geometry") else None
            )
            pending = select_temporal_frames(
                [f for f in observation.frames if f.cue], self.demo_frames
            ) + select_temporal_frames(
                [f for f in observation.frames if not f.cue], self.initial_live_frames
            )
            # Every request includes the latest measured state, even after a rejected action.
            if observation.current.index not in {f.index for f in pending}:
                pending.append(observation.current)
            audit.capture(observation.frames, 0)
            if self.examples is not None:
                window.example = self.examples(output)
                pending.extend(window.example.initial())
            session = ToolSession(
                self.env,
                actions,
                observation,
                pending,
                window,
                audit,
                log,
                max_steps=self.max_steps,
                max_tool_steps=self.max_tool_steps,
                max_observation_frames=self.max_observation_frames,
                geometry=geometry,
            )
            while calls < self.max_calls and session.remaining() > 0:
                observation, pending, steps = session.observation, session.pending, session.steps
                budget = min(self.max_tool_steps, session.remaining())
                paths, labels = image_inputs(pending, output / "frames")
                for label in labels:
                    fid = label["frame"]
                    if session.phase == "review":
                        label["source"] = (
                            "current_repeat"
                            if fid == observation.current.index
                            else "window_review"
                        )
                        label["window_id"] = window.window_id
                    elif isinstance(fid, str):
                        label["source"] = "training_example"
                    else:
                        label["source"] = "current_repeat" if fid in session.delivered else "fresh"
                session.delivered.update(f.index for f in pending)
                saved_paths, saved_labels = image_inputs(
                    list(memory.frames.values()), output / "frames"
                )
                for label in saved_labels:
                    label["source"] = "retained_memory"
                paths.extend(saved_paths)
                labels.extend(saved_labels)
                paths, labels = group_images_by_camera(paths, labels)
                if saved_labels:
                    log({"event": "retained_read", "frame_ids": list(memory.frames)})
                context = {
                    "decision_protocol": "task_harness",
                    # Used by the local schema/audit path; make_prompt removes this
                    # experiment switch so the model never receives FDS instructions
                    # in the contract-off baseline.
                    "contract_mode": self.contract_mode,
                    "task_goal": observation.goal,
                    "tool_definitions": definitions(
                        budget,
                        self.env.action_spec.dim == 8,
                        self.max_pose_targets,
                        self.max_observation_frames,
                    ),
                    "max_tool_steps": budget,
                    "max_tool_calls": self.max_tool_calls,
                    "max_observation_frames": self.max_observation_frames,
                    "execution_contract": execution_contract(
                        actions.limits, contract_mode=self.contract_mode
                    ),
                    "image_labels": labels,
                    "current_frame": observation.current.index,
                    "observation": {
                        "phase": session.phase,
                        "frames": [
                            {
                                "frame": f.index,
                                "cue": f.cue,
                                **(
                                    {"origin": "training_example"}
                                    if isinstance(f.index, str)
                                    else {
                                        "origin": "episode",
                                        "state": f.state.tolist(),
                                        "tcp_pose": kinematics.pose(f.state[:7]),
                                        **({"calibration": f.calibration} if f.calibration else {}),
                                        "window_step_offset": window.frames.get(f.index, {}).get(
                                            "step_offset"
                                        ),
                                    }
                                ),
                            }
                            for f in pending
                        ],
                        "sampling": session.sampling,
                        "review_window": window.describe(),
                        "review_result": deepcopy(session.review_result),
                        "review_history": {
                            "window_id": window.window_id,
                            "requests": deepcopy(session.review_history),
                            "meaning": "Metadata of prior review requests in this window. "
                            "These entries do not restore images or make frame IDs valid "
                            "for task-harness memory citations; the existing provenance rules apply. "
                            "Accepted receipts count requested IDs not previously supplied "
                            "at request time, excluding automatically appended current frames. "
                            "Same-set and accepted-review counts include this request and only "
                            "accepted reviews in this window. Previously supplied frames may "
                            "still merit review; these counts do not measure understanding.",
                        },
                    },
                    "memory": memory.state()["memory"],
                    "memory_contract": memory.contract(),
                    "memory_validation": deepcopy(memory.last_validation),
                    "keyframe_memory": memory.state()["retained"],
                    "skills": deepcopy(skills),
                    "training_example": deepcopy(window.example.context)
                    if window.example is not None
                    else None,
                    "previous_subgoal": session.last_result.get("subgoal")
                    if session.last_result
                    else None,
                    "latest_execution": deepcopy(session.latest_execution),
                    "last_tool_result": deepcopy(session.last_result),
                    "last_environment_result": deepcopy(session.last_environment_result),
                    "recent_motion_history": deepcopy(session.motion_history),
                    "environment_steps": steps,
                    "remaining_environment_steps": session.remaining(),
                    "remaining_calls": self.max_calls - calls,
                    "action_lower": self.env.action_spec.low.tolist(),
                    "action_upper": self.env.action_spec.high.tolist(),
                    "kinematics": kinematics.metadata,
                    "robot_geometry": geometry,
                    "current_robot": robot_reference(
                        observation.current, kinematics, geometry, actions.last_gripper
                    ),
                    "joint_motion_limits": {
                        name: value
                        for name, value in asdict(actions.limits).items()
                        if name.startswith("max_joint_")
                    },
                    "coordinates": {
                        "target_frame": kinematics.metadata["base_link"],
                        "units": "meters/radians",
                        "orientation": "absolute unit quaternion [w,x,y,z]",
                        "robot_state": "seven joint radians followed by one finger position in meters; gripper commands are separate",
                        "camera": "OpenCV: x right, y down, z forward; world_to_camera extrinsics",
                        "tcp_source": "FK from measured robot joints; no object ground truth",
                    },
                }
                if self.contract_mode == "full":
                    context["fds_protocol"] = {
                        "version": "fds_v1",
                        "contract_required": True,
                        "semantic_success": "never inferred from command execution alone",
                        "statuses": ["supported", "contradicted", "uncertain"],
                    }
                    context["last_fds_contract"] = deepcopy(last_fds_contract)
                    context["last_fds_verdict"] = deepcopy(last_fds_verdict)
                conventions = getattr(self.env, "task_language_conventions", None)
                if conventions:
                    context["task_language_conventions"] = conventions()
                if rejection is not None:
                    context["decision_recovery"] = {
                        "previous_rejection": deepcopy(rejection),
                        "instruction": "The previous response was rejected before any tool or "
                        "memory update. Return a complete corrected decision using the unchanged "
                        "observations and valid memory. Do not invent evidence or mark unresolved "
                        "claims supported just to pass validation. This is not a new physical "
                        "attempt; previous execution feedback remains historical. Each correction "
                        "uses one remaining planner call.",
                        "corrections_remaining_including_this_call": min(
                            self.max_decision_corrections - consecutive_rejections + 1,
                            self.max_calls - calls,
                        ),
                    }
                    correction_calls += 1
                calls += 1
                audit.begin(calls, context, paths, state())
                log(
                    {
                        "event": "observation_delivery",
                        "call": calls,
                        "step": steps,
                        "frames": [f.index for f in pending],
                    }
                )
                try:
                    response = self.planner.decide(context, paths, output / f"call_{calls - 1:04d}")
                    value = asdict(response)
                    if self.contract_mode == "off":
                        value.pop("fds_contract", None)
                    audit.decision(value)
                    decision = TaskDecision.parse(
                        value,
                        schema_for(context),
                        delivered_frames=[f.index for f in pending],
                        retained_frames=list(memory.frames),
                    )
                except DecisionValidationError as exc:
                    consecutive_rejections += 1
                    rejected_decisions += 1
                    rejection = {
                        **exc.feedback(),
                        "call": calls,
                        "status": "rejected",
                        "consecutive_rejections": consecutive_rejections,
                        "environment_advanced": False,
                        "memory_updated": False,
                    }
                    AuditJournal.write(
                        audit.root / f"call_{calls - 1:04d}" / "decision_rejection.json", rejection
                    )
                    log({"event": "decision_rejected", "step": steps, **rejection})
                    audit.finish(state(), [], error={"type": type(exc).__name__, "message": str(exc)})
                    if calls >= self.max_calls or consecutive_rejections > self.max_decision_corrections:
                        stop = (
                            "budget_exhausted" if calls >= self.max_calls
                            else "decision_recovery_exhausted"
                        )
                        session.expire(stop)
                        break
                    continue
                if rejection is not None:
                    recovered_decisions += 1
                    recovery = {"call": calls, "rejected_calls": consecutive_rejections}
                    AuditJournal.write(
                        audit.root / f"call_{calls - 1:04d}" / "decision_recovery.json", recovery
                    )
                    log({"event": "decision_recovered", **recovery})
                rejection, consecutive_rejections = None, 0
                if decision.fds_contract is not None:
                    last_fds_contract = deepcopy(decision.fds_contract)
                    AuditJournal.write(
                        audit.root / f"call_{calls - 1:04d}" / "fds_contract.json",
                        decision.fds_contract,
                    )
                memory.update(decision, pending, calls, log)
                AuditJournal.write(
                    output / "memory_versions" / f"call_{calls - 1:04d}.json", memory.state()
                )
                log({"event": "decision", "call": calls, "step": steps, **value})
                batch = session.execute(decision, calls, memory.frames)
                if decision.fds_contract is not None:
                    last_fds_verdict = execution_verdict(
                        decision.fds_contract, batch.receipts
                    )
                    AuditJournal.write(
                        audit.root / f"call_{calls - 1:04d}" / "fds_verdict.json",
                        last_fds_verdict,
                    )
                    log(
                        {
                            "event": "fds_verdict",
                            "call": calls,
                            "step": steps,
                            **last_fds_verdict,
                        }
                    )
                stop = batch.stop_reason or stop
                if batch.stop_reason or session.remaining() <= 0 or calls >= self.max_calls:
                    session.expire("episode_end" if batch.stop_reason else "budget_exhausted")
                audit.finish(state(), batch.receipts)
                if batch.stop_reason:
                    break
            summary = {
                "method": "task_harness",
                "contract_mode": self.contract_mode,
                "episode": episode,
                "task_goal": session.observation.goal,
                "steps": session.steps if session else 0,
                "planner_calls": calls,
                "environment_batches": session.environment_batches if session else 0,
                "tool_batches": deepcopy(session.batch_counts) if session else {},
                "official_status": session.status if session else "unknown",
                "success": stop == "environment_done" and session.status == "success",
                "stop_reason": stop,
                "memory_enabled": True,
            }
        except BaseException as exc:
            summary = {
                "method": "task_harness",
                "contract_mode": self.contract_mode,
                "episode": episode,
                "steps": session.steps if session else 0,
                "planner_calls": calls,
                "environment_batches": session.environment_batches if session else 0,
                "tool_batches": deepcopy(session.batch_counts) if session else {},
                "official_status": session.status if session else "unknown",
                "success": False,
                "stop_reason": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "memory_enabled": True,
            }
            audit.finish(state(), [], error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            try:
                window.close()
                if hasattr(self.env, "set_run_outcome"):
                    error = (
                        {k: summary[k] for k in ("error_type", "error")}
                        if summary and summary["stop_reason"] == "error"
                        else None
                    )
                    self.env.set_run_outcome(summary["stop_reason"] if summary else stop, error)
                self.env.close()
            except Exception as exc:
                log({"event": "cleanup_error", "error": str(exc)})
                if summary is not None and summary["stop_reason"] != "error":
                    summary.update(
                        success=False,
                        stop_reason="error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    raise
            finally:
                if summary is not None:
                    summary["decision_recovery"] = {
                        "max_corrections": self.max_decision_corrections,
                        "rejected_decisions": rejected_decisions,
                        "correction_calls": correction_calls,
                        "recovered_decisions": recovered_decisions,
                        "last_rejection": rejection,
                    }
                    AuditJournal.write(output / "summary.json", summary)
        return summary
