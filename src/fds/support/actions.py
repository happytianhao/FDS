"""Mechanical VLA and absolute-pose adapters without task monitors or memory."""

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from fds.common.grounding import compile_subgoal
from fds.common.kinematics import rotation_error
from fds.common.vla_language import compile_training_subgoal, validate_wire


def quaternion_matrix(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError("quaternion_wxyz must be finite, unit length within 0.01")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class PoseLimits:
    max_translation_m: float = 0.12
    max_rotation_rad: float = 0.6
    max_joint_delta_rad: float = 1.2
    max_joint_step_rad: float = 0.08


class Actions:
    def __init__(self, env, policy, kinematics=None, limits=None):
        self.env, self.policy, self.kinematics = env, policy, kinematics
        self.last_gripper = None
        self.limits = PoseLimits(**(limits or {}))
        values = list(vars(self.limits).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Pose limits must be positive and finite")

    def applied(self, action):
        if self.env.action_spec.dim == 8:
            self.last_gripper = float(action[7])

    def prepare(self, call, observation, visible, budget, log):
        name, args = call["name"], call["arguments"]
        spec, state = self.env.action_spec, observation.current.state
        if name == "finish":
            return [], {}
        if name == "vla":
            if self.policy is None:
                raise ValueError("VLA policy is unavailable")
            if observation.current.index not in visible:
                raise ValueError(
                    "VLA requires a fresh current image; wait to obtain new observations"
                )
            fmt = getattr(self.policy, "grounding_format", "plain")
            if fmt == "training_yx":
                subgoal = compile_training_subgoal(args, observation.current.index, visible)
                validate_wire(subgoal, observation.current.images["front"].shape)
            else:
                if args["grounding"]:
                    raise ValueError("Plain VLA requires grounding=[]")
                subgoal = compile_subgoal(
                    SimpleNamespace(**args), observation.current.index, visible, [], False
                )
            started = time.perf_counter()
            try:
                raw = np.asarray(self.policy.infer(observation, subgoal), dtype=float)
            finally:
                log(
                    {
                        "event": "policy_inference",
                        "subgoal": subgoal,
                        "seconds": time.perf_counter() - started,
                    }
                )
            if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] not in (spec.dim, 8):
                raise ValueError("VLA must return a nonempty [time, action_dim] chunk")
            actions = raw[: min(args["steps"], budget), : spec.dim].copy()
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite VLA actions")
            log({"event": "policy_output", "audience": "human_audit_only", "actions": raw.tolist()})
            if spec.dim == 8:
                original = actions[:, 7].copy()
                actions[:, 7] = np.clip(original, spec.low[7], spec.high[7])
                if not np.array_equal(original, actions[:, 7]):
                    log(
                        {
                            "event": "gripper_saturated",
                            "raw": original.tolist(),
                            "applied": actions[:, 7].tolist(),
                        }
                    )
            details = {"compiled_subgoal": subgoal, "native_chunk_length": len(raw)}
        else:
            grip = (
                (
                    self.last_gripper
                    if self.last_gripper is not None
                    else self.env.gripper_command(state[7])
                )
                if spec.dim == 8
                else None
            )
            if name == "wait":
                action = np.r_[state[:7], grip] if spec.dim == 8 else state[:7]
                actions = np.repeat(action[None], args["steps"], axis=0)
                details = {}
            elif name == "move_ee":
                if self.kinematics is None:
                    raise ValueError("No declared FK/IK adapter")
                if sum(t["steps"] for t in args["targets"] if t["steps"] != "auto") > budget:
                    raise ValueError("Total pose target steps exceed the call budget")
                q, chunks, residuals, resolved = state[:7].astype(float), [], [], []
                nominal_pose = None
                for target_index, t in enumerate(args["targets"]):
                    pose, _ = self.kinematics.evaluate(q)
                    target_position = np.asarray(t["position_m"], dtype=float)
                    target_rotation = quaternion_matrix(t["quaternion_wxyz"])
                    translation = target_position - pose[:3, 3]
                    rotvec = rotation_error(target_rotation, pose[:3, :3])
                    # Validate the commanded path, not previous IK endpoint approximation.
                    # IK and joint speed/displacement checks still use the solved joints.
                    reference = pose if nominal_pose is None else nominal_pose
                    requested_translation = target_position - reference[:3, 3]
                    requested_rotation = rotation_error(target_rotation, reference[:3, :3])
                    if (
                        np.linalg.norm(requested_translation) > self.limits.max_translation_m + 1e-9
                        or np.linalg.norm(requested_rotation) > self.limits.max_rotation_rad + 1e-9
                    ):
                        raise ValueError(
                            "Absolute target exceeds local translation/rotation limits; "
                            f"target {target_index + 1}: translation norm {np.linalg.norm(requested_translation):.6f} m "
                            f"(max {self.limits.max_translation_m}), rotation {np.linalg.norm(requested_rotation):.6f} rad "
                            f"(max {self.limits.max_rotation_rad}); "
                            f"path reference position_m={reference[:3, 3].tolist()}. Request a shorter intermediate pose."
                        )
                    target, residual = self.kinematics.solve_delta(
                        q, translation, rotvec, "robot_base"
                    )
                    delta = float(np.max(np.abs(target - q)))
                    settle = 4 if t["steps"] == "auto" else 0
                    motion_steps = (
                        max(8, math.ceil(delta / (0.8 * self.limits.max_joint_step_rad)))
                        if t["steps"] == "auto"
                        else t["steps"]
                    )
                    if (
                        delta > self.limits.max_joint_delta_rad
                        or delta / motion_steps > self.limits.max_joint_step_rad
                    ):
                        raise ValueError(
                            f"IK target exceeds joint displacement/speed limits at target {target_index + 1}: "
                            f"joint delta {delta:.6f} rad (max {self.limits.max_joint_delta_rad}), "
                            f"per-step delta {delta / motion_steps:.6f} rad (max {self.limits.max_joint_step_rad}). "
                            "Use an intermediate pose for displacement; more steps for speed."
                        )
                    if spec.dim == 8:
                        grip = {"hold": grip, "open": 1.0, "close": -1.0}[t["gripper"]]
                    elif t["gripper"] != "hold":
                        raise ValueError("This robot has no gripper")
                    joints = np.linspace(q, target, motion_steps + 1)[1:]
                    if settle:
                        joints = np.concatenate([joints, np.repeat(target[None], settle, axis=0)])
                    resolved.append(
                        {
                            **t,
                            "steps": len(joints),
                            "motion_steps": motion_steps,
                            "settle_steps": settle,
                        }
                    )
                    chunks.append(
                        np.c_[joints, np.full(len(joints), grip)] if spec.dim == 8 else joints
                    )
                    residuals.append(residual)
                    q = target
                    nominal_pose = np.eye(4)
                    nominal_pose[:3, 3], nominal_pose[:3, :3] = target_position, target_rotation
                actions = np.concatenate(chunks)
                details = {
                    "ik_solutions": residuals,
                    "requested_targets": args["targets"],
                    "resolved_targets": resolved,
                    "collision_checked": False,
                }
            else:
                raise ValueError(f"Unknown action: {name}")
        if len(actions) > budget:
            raise ValueError(
                f"Action chunk requires {len(actions)} steps, exceeds remaining budget {budget}; "
                "no actions executed. Use fewer targets or explicit durations within speed limits."
            )
        return [spec.validate(a) for a in actions], details

    def feedback(self, details, observation, executed_steps):
        result = dict(details)
        if details.get("ik_solutions") and self.kinematics is not None and observation is not None:
            actual, _ = self.kinematics.evaluate(observation.current.state[:7])
            # On early termination compare with the last target actually commanded,
            # not an unexecuted later waypoint.
            elapsed = 0
            for index, t in enumerate(
                details.get("resolved_targets", details["requested_targets"])
            ):
                solution = details["ik_solutions"][index]
                elapsed += t["steps"]
                if elapsed >= executed_steps:
                    break
            target = np.asarray(solution["target_pose"])
            result["tracking"] = {
                "measured_position_error_m": float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
                "measured_rotation_error_rad": float(
                    np.linalg.norm(rotation_error(target[:3, :3], actual[:3, :3]))
                ),
                "target_fully_commanded": executed_steps >= elapsed,
                "source": "FK_from_proprioception",
                "semantic_success": "unverified",
            }
        return result
