"""Sys2-owned decisions; host-owned bounded memory, observations and mechanical execution."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from fds.common.audit import AuditJournal
from fds.common.episode import EpisodeIO
from fds.support.coordinates import robot_reference
from fds.task_harness.actions import TaskActions
from fds.task_harness.contracts import definitions, execution_contract
from fds.task_harness.fds import execution_verdict
from fds.task_harness.memory import EpisodicMemory
from fds.task_harness.observation import (
    ObservationWindow,
    deduplicate_image_inputs,
    group_images_by_camera,
    image_inputs,
    sample_every_third,
)
from fds.task_harness.postcheck import OutcomeTracker
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
        # Legacy caps no longer truncate fixed-stride fresh/demo inputs.
        opts.pop("demo_frames", None)
        opts.pop("initial_live_frames", None)
        self.max_tool_calls = opts.pop("max_tool_calls", 8)
        self.max_pose_targets = opts.pop("max_pose_targets", 32)
        self.max_review_frames = opts.pop("max_review_frames", opts.pop("max_observation_frames", 12))
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
        self.postcheck_enabled = fds_options.pop("postcheck_enabled", True)
        if type(self.postcheck_enabled) is not bool:
            raise ValueError("fds.postcheck_enabled must be a boolean")
        self.postcheck_enabled = self.postcheck_enabled and self.contract_mode == "full"
        if fds_options:
            raise ValueError(f"Unknown fds options: {sorted(fds_options)}")
        for name, value, low, high in (
            ("max_keyframes", self.max_keyframes, 0, 8),
            ("max_tool_calls", self.max_tool_calls, 1, 16),
            ("max_pose_targets", self.max_pose_targets, 1, 64),
            ("max_review_frames", self.max_review_frames, 1, 32),
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
        outcomes = OutcomeTracker() if self.postcheck_enabled else None
        skills = deepcopy(self.skills)
        AuditJournal.write(
            output / "skills.json", {"skills": skills, "mutable_during_episode": False}
        )
        calls, stop = 0, "budget_exhausted"
        last_fds_contract, last_fds_verdict = None, None
        last_fds_call, last_environment_fds = None, None
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
                **({"outcome_tracking": outcomes.summary(),
                    "event_memory_versions": outcomes.context(memory)["event_memory_versions"]}
                   if outcomes else {}),
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
            pending = sample_every_third(
                [f for f in observation.frames if f.cue]
            ) + sample_every_third(
                [f for f in observation.frames if not f.cue]
            )
            # Include the initial live state once; subsequent inputs contain only new pixels.
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
                max_review_frames=self.max_review_frames,
                geometry=geometry,
            )
            while calls < self.max_calls and session.remaining() > 0:
                observation, pending, steps = session.observation, session.pending, session.steps
                budget = min(self.max_tool_steps, session.remaining())
                paths, labels = image_inputs(pending, output / "frames")
                input_window_id = (
                    session.review_result["requested_window_id"]
                    if session.phase == "review" and session.review_result else window.window_id
                )
                input_window_frames = window.windows.get(input_window_id, {}).get("frames", {})
                for label in labels:
                    fid = label["frame"]
                    if session.phase == "review":
                        label["source"] = "history_review"
                        label["window_id"] = input_window_id
                    elif isinstance(fid, str):
                        label["source"] = "training_example"
                    else:
                        label["source"] = "fresh"
                session.delivered.update(f.index for f in pending)
                saved_paths, saved_labels = image_inputs(
                    list(memory.frames.values()), output / "frames"
                )
                for label in saved_labels:
                    label["source"] = "retained_memory"
                paths.extend(saved_paths)
                labels.extend(saved_labels)
                paths, labels = deduplicate_image_inputs(paths, labels)
                paths, labels = group_images_by_camera(paths, labels)
                # Memory changes in this decision do not retroactively hide supplied images.
                session.input_frames = {label["frame"] for label in labels}
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
                        self.max_review_frames,
                    ),
                    "max_tool_steps": budget,
                    "max_tool_calls": self.max_tool_calls,
                    "max_review_frames": self.max_review_frames,
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
                                        "window_step_offset": input_window_frames.get(f.index, {}).get(
                                            "step_offset"
                                        ),
                                    }
                                ),
                            }
                            for f in pending
                        ],
                        "sampling": session.sampling,
                        "review_window": window.describe(),
                        "review_windows": window.history(),
                        "review_result": deepcopy(session.review_result),
                        "review_history": {
                            "window_id": window.window_id,
                            "requests": deepcopy(session.review_history),
                            "meaning": "Metadata of review requests since the latest environment batch. "
                            "These entries do not restore images or make frame IDs valid "
                            "for task-harness memory citations; the existing provenance rules apply. "
                            "Accepted receipts count requested IDs not previously supplied "
                            "at request time. Reviews return exactly the requested frames. "
                            "Same-set and accepted-review counts include this request and only "
                            "accepted reviews for the requested window. Previously supplied frames may "
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
                    context["available_evidence_frames"] = list(dict.fromkeys(
                        [f.index for f in pending] + list(memory.frames)
                    ))
                    context["fds_protocol"] = {
                        "version": "fds_v2_postcheck" if outcomes is not None else "fds_v1",
                        "contract_required": True,
                        "semantic_success": "never inferred from command execution alone",
                        "statuses": ["supported", "contradicted", "uncertain"],
                    }
                    context["last_fds_contract"] = deepcopy(last_fds_contract)
                    context["last_fds_verdict"] = deepcopy(last_fds_verdict)
                    if (last_environment_fds is not None
                            and last_environment_fds["request_call"] != last_fds_call):
                        context["last_environment_fds"] = deepcopy(last_environment_fds)
                        context["fds_protocol"]["environment_prediction"] = (
                            "last_environment_fds preserves the prediction made before the "
                            "last physical action through intervening auxiliary decisions. "
                            "Compare it with supplied observations and execution feedback; "
                            "the contract itself is not outcome evidence. Its host verdict "
                            "classifies mechanical execution only, not visual/semantic truth."
                        )
                    if outcomes is not None:
                        context.update(outcomes.context(memory))
                conventions = getattr(self.env, "task_language_conventions", None)
                if conventions:
                    context["task_language_conventions"] = conventions()
                if rejection is not None:
                    context["decision_recovery"] = {
                        "previous_rejection": deepcopy(rejection),
                        "instruction": "The previous response was rejected before any tool or "
                        "memory update. Return a complete corrected decision using valid memory "
                        "and supplied images. Previous unretained images are not automatically "
                        "replayed; request review_frames if their pixels are needed again. "
                        "Do not invent evidence or mark unresolved "
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
                    staged_memory, memory_logs = None, []
                    if outcomes is not None:
                        # Validate the complete review against a preview, so a rejected
                        # reconciliation cannot mutate memory or execute any tool.
                        staged_memory = deepcopy(memory)
                        staged_memory.update(decision, pending, calls, memory_logs.append)
                        try:
                            outcomes.validate(
                                decision, memory, staged_memory,
                                [f.index for f in pending] + list(memory.frames),
                            )
                        except ValueError as exc:
                            raise DecisionValidationError(
                                str(exc), code="outcome_validation", path=["fds_contract"],
                                response=json.dumps(value, ensure_ascii=False),
                            ) from exc
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
                    session.pending, session.phase = [], "correction"
                    session.sampling = session.review_result = None
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
                    last_fds_call = calls
                    AuditJournal.write(
                        audit.root / f"call_{calls - 1:04d}" / "fds_contract.json",
                        decision.fds_contract,
                    )
                if outcomes is not None:
                    review = outcomes.accept_review(decision, memory, staged_memory, calls)
                    memory = staged_memory
                    for event in memory_logs:
                        log(event)
                    AuditJournal.write(
                        audit.root / f"call_{calls - 1:04d}" / "outcome_review.json", review
                    )
                    log({"event": "outcome_review", **review})
                else:
                    memory.update(decision, pending, calls, log)
                AuditJournal.write(
                    output / "memory_versions" / f"call_{calls - 1:04d}.json", memory.state()
                )
                log({"event": "decision", "call": calls, "step": steps, **value})
                batch = session.execute(decision, calls, memory.frames)
                if outcomes is not None:
                    binding = outcomes.register_action(decision, batch, session.observation.current.index)
                    if binding is not None:
                        AuditJournal.write(
                            audit.root / f"call_{calls - 1:04d}" / "action_outcomes.json", binding
                        )
                if decision.fds_contract is not None:
                    last_fds_verdict = execution_verdict(
                        decision.fds_contract, batch.receipts
                    )
                    if batch.kind == "environment" and session.steps > batch.start_step:
                        last_environment_fds = {
                            "request_call": calls,
                            "environment_steps_before": batch.start_step,
                            "environment_steps_after": session.steps,
                            "start_frame": batch.start_frame,
                            "end_frame": session.observation.current.index,
                            "contract": deepcopy(decision.fds_contract),
                            "mechanical_verdict": deepcopy(last_fds_verdict),
                        }
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
                    summary["postcheck_enabled"] = self.postcheck_enabled
                    if outcomes is not None:
                        summary["outcome_tracking"] = outcomes.summary()
                    summary["decision_recovery"] = {
                        "max_corrections": self.max_decision_corrections,
                        "rejected_decisions": rejected_decisions,
                        "correction_calls": correction_calls,
                        "recovered_decisions": recovered_decisions,
                        "last_rejection": rejection,
                    }
                    AuditJournal.write(output / "summary.json", summary)
        return summary
