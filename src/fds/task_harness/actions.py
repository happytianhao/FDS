"""Continuous long-range pose moves with bounded IK and original-target timing."""

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from fds.common.kinematics import rotation, rotation_error
from fds.support.actions import Actions, quaternion_matrix
from fds.support.coordinates import matrix_quaternion


@dataclass
class PoseLimits:
    # Rotation subdivision keeps the local solver away from its pi singularity.
    # Translation has no distance cap or fixed subdivision interval.
    max_rotation_rad: float = 0.6
    max_joint_delta_rad: float = 1.2
    max_joint_step_rad: float = 0.08


def intermediate_pose(start, end, fraction):
    relative = np.asarray(matrix_quaternion(end[:3, :3] @ start[:3, :3].T))
    sine = float(np.linalg.norm(relative[1:]))
    angle = 2 * math.atan2(sine, relative[0])
    vector = relative[1:] * (angle / sine) if sine > 0 else np.zeros(3)
    pose = np.eye(4)
    pose[:3, 3] = start[:3, 3] + fraction * (end[:3, 3] - start[:3, 3])
    pose[:3, :3] = rotation(vector * fraction) @ start[:3, :3]
    return pose, angle


class TaskActions(Actions):
    def __init__(self, env, policy, kinematics=None, limits=None):
        super().__init__(env, policy, kinematics)
        if "max_translation_m" in (limits or {}):
            raise ValueError(
                "task_harness no longer has max_translation_m; remove this obsolete setting. "
                "Provide the desired reachable target directly."
            )
        self.limits = PoseLimits(**(limits or {}))
        values = list(vars(self.limits).values())
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Pose limits must be positive and finite")

    def prepare(self, call, observation, visible, budget, log):
        if call["name"] != "move_ee" or self.kinematics is None:
            return super().prepare(call, observation, visible, budget, log)
        targets = call["arguments"]["targets"]
        if sum(t["steps"] for t in targets if t["steps"] != "auto") > budget:
            raise ValueError("Total pose target steps exceed the call budget")
        state, spec = observation.current.state, self.env.action_spec
        q = np.asarray(state[:7], dtype=float)
        reference, _ = self.kinematics.evaluate(q)
        grip = (
            self.last_gripper
            if self.last_gripper is not None
            else self.env.gripper_command(state[7])
            if spec.dim == 8
            else None
        )
        chunks, resolved, residuals, subdivisions = [], [], [], []
        attempts = 0
        # Bounded preflight search; never spend environment steps to probe IK.
        max_depth = math.ceil(math.log2(max(1, budget)))

        for index, original in enumerate(targets):
            position = np.asarray(original["position_m"], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("position_m must contain three finite coordinates")
            endpoint = np.eye(4)
            endpoint[:3, 3] = position
            endpoint[:3, :3] = quaternion_matrix(original["quaternion_wxyz"])
            _, angle = intermediate_pose(reference, endpoint, 0)
            capacity = budget - len(resolved)
            if original["steps"] != "auto":
                capacity = min(capacity, original["steps"])
            effective_angle = max(0.0, angle - 1e-9)
            if capacity < 1 or effective_angle > capacity * self.limits.max_rotation_rad:
                raise ValueError(
                    "Rotation subdivision exceeds the step budget; no actions executed"
                )
            count = max(1, math.ceil(effective_angle / self.limits.max_rotation_rad))
            knots = []
            reasons = {"rotation"} if count > 1 else set()

            def solve_segment(
                start_q,
                start_pose,
                end_pose,
                depth,
                *,
                knots=knots,
                capacity=capacity,
                reasons=reasons,
                index=index,
            ):
                nonlocal attempts
                attempts += 1
                if attempts > 2 * max(1, budget) or len(knots) >= capacity:
                    raise ValueError(
                        "Adaptive IK subdivision exceeds preflight/step budget; no actions executed"
                    )
                actual, _ = self.kinematics.evaluate(start_q)
                failure = None
                try:
                    solved, residual = self.kinematics.solve_delta(
                        start_q,
                        end_pose[:3, 3] - actual[:3, 3],
                        rotation_error(end_pose[:3, :3], actual[:3, :3]),
                        "robot_base",
                    )
                except ValueError as exc:
                    failure = str(exc)
                    reasons.add("ik_retry")
                else:
                    solved = np.asarray(solved, dtype=float)
                    if solved.shape != start_q.shape or not np.isfinite(solved).all():
                        raise ValueError("IK returned invalid joints; no actions executed")
                    if np.any(solved < spec.low[:7]) or np.any(solved > spec.high[:7]):
                        raise ValueError("Action exceeds robot joint limits; no actions executed")
                    probe = np.r_[solved, 0.0] if spec.dim == 8 else solved
                    spec.validate(probe)
                    delta = float(np.max(np.abs(solved - start_q)))
                    if delta <= self.limits.max_joint_delta_rad:
                        knots.append((start_q, solved, end_pose, residual, delta))
                        return solved
                    failure = (
                        f"joint displacement {delta:.6f} exceeds {self.limits.max_joint_delta_rad}"
                    )
                    reasons.add("joint_delta")
                if depth >= max_depth:
                    raise ValueError(
                        f"Adaptive IK subdivision could not resolve target {index + 1}: "
                        f"{failure}; no actions executed"
                    )
                middle, _ = intermediate_pose(start_pose, end_pose, 0.5)
                middle_q = solve_segment(start_q, start_pose, middle, depth + 1)
                return solve_segment(middle_q, middle, end_pose, depth + 1)

            current_pose, current_q = reference, q
            for part in range(count):
                target_pose, _ = intermediate_pose(reference, endpoint, (part + 1) / count)
                if part == count - 1:
                    target_pose = endpoint
                current_q = solve_segment(current_q, current_pose, target_pose, 0)
                current_pose = target_pose
            size = len(knots)
            if original["steps"] == "auto":
                planned_joint_step = 0.8 * self.limits.max_joint_step_rad
                if any(knot[4] > budget * planned_joint_step for knot in knots):
                    raise ValueError("Auto motion exceeds the step budget; no actions executed")
                durations = [max(1, math.ceil(knot[4] / planned_joint_step)) for knot in knots]
                # One minimum per ORIGINAL target, shared over moving support points.
                missing = max(0, 8 - sum(durations))
                base = durations.copy()
                total = sum(base)
                previous = cumulative = 0
                for i, length in enumerate(base):
                    cumulative += length
                    allocated = cumulative * missing // total
                    durations[i] += allocated - previous
                    previous = allocated
                settling = [0] * (size - 1) + [4]
            else:
                if original["steps"] < size:
                    raise ValueError(
                        "Adaptive subdivision needs more explicit steps; no actions executed"
                    )
                durations = [
                    (i + 1) * original["steps"] // size - i * original["steps"] // size
                    for i in range(size)
                ]
                settling = [0] * size
            if sum(durations) + sum(settling) + sum(t["steps"] for t in resolved) > budget:
                raise ValueError(
                    "Total resolved motion exceeds the step budget; no actions executed"
                )
            if spec.dim == 8:
                grip = {"hold": grip, "open": 1.0, "close": -1.0}[original["gripper"]]
            elif original["gripper"] != "hold":
                raise ValueError("This robot has no gripper")
            first_resolved = len(resolved)
            for part, ((start_q, solved, pose, residual, delta), motion, settle) in enumerate(
                zip(knots, durations, settling, strict=True)
            ):
                if delta / motion > self.limits.max_joint_step_rad:
                    raise ValueError(
                        f"IK target exceeds joint displacement/speed limits at target {index + 1}: "
                        f"per-step delta {delta / motion:.6f} rad "
                        f"(max {self.limits.max_joint_step_rad}); "
                        "increase explicit steps. No actions executed."
                    )
                joints = np.linspace(start_q, solved, motion + 1)[1:]
                if settle:
                    joints = np.concatenate([joints, np.repeat(solved[None], settle, axis=0)])
                resolved.append(
                    {
                        **deepcopy(original),
                        "position_m": pose[:3, 3].tolist(),
                        "quaternion_wxyz": (
                            deepcopy(original["quaternion_wxyz"])
                            if part == size - 1
                            else matrix_quaternion(pose[:3, :3])
                        ),
                        "steps": len(joints),
                        "motion_steps": motion,
                        "settle_steps": settle,
                        "requested_target_index": index,
                        "subtarget_index": part,
                        "subtarget_count": size,
                    }
                )
                chunks.append(
                    np.c_[joints, np.full(len(joints), grip)] if spec.dim == 8 else joints
                )
                residuals.append(residual)
            if size > 1:
                subdivisions.append(
                    {
                        "requested_target_index": index,
                        "translation_m": math.hypot(*(position - reference[:3, 3])),
                        "rotation_rad": angle,
                        "reference_position_m": reference[:3, 3].tolist(),
                        "resolved_target_indices": list(range(first_resolved, len(resolved))),
                        "requested_steps": original["steps"],
                        "resolved_steps": [t["steps"] for t in resolved[first_resolved:]],
                        "reasons": sorted(reasons),
                    }
                )
            q, reference = current_q, endpoint

        actions = np.concatenate(chunks)
        if len(actions) > budget:
            raise ValueError(
                f"Action chunk requires {len(actions)} steps, exceeds remaining budget {budget}; "
                "no actions executed"
            )
        validated = [spec.validate(action) for action in actions]
        details = {
            "requested_targets": deepcopy(targets),
            "resolved_targets": resolved,
            "ik_solutions": residuals,
            "collision_checked": False,
            "timing_policy": "original_target_auto_v1",
            "ik_attempts": attempts,
        }
        if subdivisions:
            details["pose_subdivisions"] = subdivisions
            for item in subdivisions:
                log({"event": "pose_subdivision", **item})
        return validated, details

    def feedback(self, details, observation, executed_steps):
        result = super().feedback(details, observation, executed_steps)
        if details.get("resolved_targets") and "tracking" in result:
            elapsed = 0
            for _index, target in enumerate(details["resolved_targets"]):
                elapsed += target["steps"]
                if elapsed >= executed_steps:
                    break
            result["tracking"].update(
                resolved_target_index=_index,
                requested_target_index=target["requested_target_index"],
                target_fully_commanded=(
                    executed_steps >= elapsed
                    and target["subtarget_index"] == target["subtarget_count"] - 1
                ),
            )
        return result
