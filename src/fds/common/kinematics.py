"""URDF-only FK and local damped-least-squares IK, with no scene-state access."""

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def prismatic_joint_spec(urdf, joint_name):
    """Public robot structure, independent of controller targets and scene state."""
    data = Path(urdf).read_bytes()
    joints = [j for j in ET.fromstring(data).findall("joint") if j.get("name") == joint_name]
    if len(joints) != 1 or joints[0].get("type") != "prismatic":
        raise ValueError("Measured gripper joint must be a unique prismatic URDF joint")
    limit = joints[0].find("limit")
    if limit is None:
        raise ValueError("Measured gripper joint has no URDF travel limits")
    low, high = float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Invalid gripper URDF travel limits")
    return {
        "joint_name": joint_name,
        "lower_m": low,
        "upper_m": high,
        "source": "own_robot_URDF",
        "urdf_sha256": hashlib.sha256(data).hexdigest(),
    }


def rotation(vector):
    vector = np.asarray(vector, dtype=float)
    theta = np.linalg.norm(vector)
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = vector / theta
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)


def rotation_error(target, current):
    r = target @ current.T
    theta = math.acos(float(np.clip((np.trace(r) - 1) / 2, -1, 1)))
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    if theta < 1e-6:
        return skew
    if math.pi - theta < 1e-5:
        raise ValueError("IK rotation near pi is outside the local solver domain")
    return skew * theta / math.sin(theta)


class URDFKinematics:
    def __init__(self, urdf, joint_names, tcp_link):
        data = Path(urdf).read_bytes()
        root = ET.fromstring(data)
        by_child = {j.find("child").attrib["link"]: j for j in root.findall("joint")}
        chain, link = [], tcp_link
        visited = set()
        while link in by_child:
            if link in visited:
                raise ValueError("Cyclic URDF")
            visited.add(link)
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        if not chain:
            raise ValueError("TCP not found in URDF chain")
        self.joint_names, self.chain = list(joint_names), []
        self.low, self.high = np.full(len(joint_names), -np.inf), np.full(len(joint_names), np.inf)
        used = set()
        for joint in reversed(chain):
            kind, name = joint.attrib["type"], joint.attrib["name"]
            if kind not in ("fixed", "revolute") or (kind != "fixed" and name not in joint_names):
                raise ValueError(f"Unsupported or unmapped URDF joint: {name} ({kind})")
            origin = joint.find("origin")
            attrs = origin.attrib if origin is not None else {}
            xyz = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (
                rotation([0, 0, rpy[2]]) @ rotation([0, rpy[1], 0]) @ rotation([rpy[0], 0, 0])
            )
            index, axis = None, None
            if kind == "revolute":
                index = joint_names.index(name)
                used.add(index)
                axis_xml = joint.find("axis")
                axis = np.fromstring(
                    axis_xml.attrib.get("xyz", "1 0 0") if axis_xml is not None else "1 0 0",
                    sep=" ",
                )
                axis /= np.linalg.norm(axis)
                limit = joint.find("limit").attrib
                self.low[index], self.high[index] = float(limit["lower"]), float(limit["upper"])
            self.chain.append((transform, index, axis))
        if used != set(range(len(joint_names))) or not np.isfinite([self.low, self.high]).all():
            raise ValueError("Incomplete bounded joint chain")
        self.metadata = {
            "base_link": link,
            "tcp_link": tcp_link,
            "joint_names": self.joint_names,
            "urdf_file": Path(urdf).name,
            "urdf_sha256": hashlib.sha256(data).hexdigest(),
            "source": "own_joint_state_and_URDF",
            "collision_checked": False,
        }

    def evaluate(self, joints):
        q = np.asarray(joints, dtype=float)
        if q.shape != self.low.shape or not np.isfinite(q).all():
            raise ValueError("Invalid FK joint vector")
        pose, axes = np.eye(4), []
        for origin, index, axis in self.chain:
            pose = pose @ origin
            if index is not None:
                axes.append((index, pose[:3, :3] @ axis, pose[:3, 3].copy()))
                turn = np.eye(4)
                turn[:3, :3] = rotation(axis * q[index])
                pose = pose @ turn
        jacobian = np.zeros((6, len(q)))
        for index, axis, position in axes:
            jacobian[:3, index] = np.cross(axis, pose[:3, 3] - position)
            jacobian[3:, index] = axis
        return pose, jacobian

    def pose(self, joints):
        transform, _ = self.evaluate(joints)
        return {
            "frame": self.metadata["base_link"],
            "tcp_link": self.metadata["tcp_link"],
            "position_m": transform[:3, 3].tolist(),
            "rotation_matrix": transform[:3, :3].tolist(),
            "source": "FK_from_proprioception",
        }

    def geometry(self):
        """Static robot geometry for a direct policy; never scene geometry."""
        return {
            **self.metadata,
            "joint_lower_rad": self.low.tolist(),
            "joint_upper_rad": self.high.tolist(),
            "chain": [
                {
                    "parent_to_joint_origin": origin.tolist(),
                    "joint": self.joint_names[index] if index is not None else "fixed",
                    "axis_in_joint_frame": axis.tolist() if axis is not None else None,
                }
                for origin, index, axis in self.chain
            ],
            "convention": "URDF right-handed coordinates, meters/radians. "
            "Compose origin then joint rotation in chain order to reach TCP.",
        }

    def solve_delta(self, joints, translation, rotvec, frame):
        q = np.asarray(joints, dtype=float).copy()
        target, _ = self.evaluate(q)
        basis = target[:3, :3].copy() if frame == "tool" else np.eye(3)
        target[:3, 3] += basis @ np.asarray(translation)
        target[:3, :3] = rotation(basis @ np.asarray(rotvec)) @ target[:3, :3]
        position_tolerance, rotation_tolerance = 0.001, 0.01
        first_pass_iterations = 120
        max_iterations = 2 * first_pass_iterations
        for iteration in range(max_iterations + 1):
            actual, jacobian = self.evaluate(q)
            error = np.r_[
                target[:3, 3] - actual[:3, 3], rotation_error(target[:3, :3], actual[:3, :3])
            ]
            position_error = float(np.linalg.norm(error[:3]))
            angular_error = float(np.linalg.norm(error[3:]))
            if position_error < position_tolerance and angular_error < rotation_tolerance:
                return q, {
                    "position_error_m": position_error,
                    "rotation_error_rad": angular_error,
                    "target_pose": target.tolist(),
                    "collision_checked": False,
                    "iterations": iteration,
                    "solver_pass": (
                        "unscaled" if iteration <= first_pass_iterations else "tolerance_scaled"
                    ),
                }
            if iteration == max_iterations:
                break
            if iteration >= first_pass_iterations:
                # Balance meters and radians by their existing acceptance tolerances.
                # Retry only after the original solve; both raw limits still apply.
                scale = position_tolerance / rotation_tolerance
                jacobian[3:] *= scale
                error[3:] *= scale
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.015**2 * np.eye(6), error
            )
            delta *= min(1, 0.08 / max(1e-12, float(np.max(np.abs(delta)))))
            q = np.clip(q + delta, self.low + 1e-5, self.high - 1e-5)
        raise ValueError(
            "IK did not converge within local pose tolerances; no action executed. "
            f"After {max_iterations} iterations: position error {position_error:.6f} m "
            f"(required < {position_tolerance}), rotation error {angular_error:.6f} rad "
            f"(required < {rotation_tolerance})."
        )
