"""Current-frame robot/camera arithmetic; no object state or retained observations."""

import numpy as np


def matrix_quaternion(rotation):
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Expected finite rotation matrix")
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return (q if q[0] >= 0 else -q).tolist()


def robot_reference(frame, kinematics, geometry, gripper_command):
    """Project only the robot TCP and virtual local offsets using public calibration."""
    pose = kinematics.pose(frame.state[:7])
    position = np.asarray(pose["position_m"])
    result = {
        "frame": frame.index,
        "source": "current_proprioception_FK_and_camera_calibration",
        "position_m": position.tolist(),
        "quaternion_wxyz": matrix_quaternion(pose["rotation_matrix"]),
        "gripper_last_command": gripper_command,
        "cameras": {},
    }
    spec = (geometry or {}).get("gripper")
    if spec:
        measured = float(frame.state[spec["state_index"]])
        result["finger"] = {
            "position_m": measured,
            "travel_fraction": (measured - spec["lower_m"]) / (spec["upper_m"] - spec["lower_m"]),
            "meaning": "0=closed limit, 1=open limit; not a grasp/success sensor",
        }
    if not geometry:
        return result
    base = np.asarray(geometry["base_to_world"])
    for camera, cal in frame.calibration.items():
        k = np.asarray(cal["K"])
        ext = np.asarray(cal["world_to_camera"])
        if k.shape != (3, 3) or ext.shape not in ((3, 4), (4, 4)):
            raise ValueError("Projection requires K[3,3] and world_to_camera[3,4 or 4,4]")
        transform = ext[:3] @ base

        def project(p, transform=transform, k=k):
            v = transform @ np.r_[p, 1.0]
            if v[2] <= 1e-6:
                return None
            uv = k @ v
            return (uv[:2] / uv[2]).tolist()

        uv = project(position)
        if uv is None:
            continue
        height, width = frame.images[camera].shape[:2]
        result["cameras"][camera] = {
            "image_size_wh": [width, height],
            "tcp_xy_pixels": uv,
            "tcp_inside_image": 0 <= uv[0] < width and 0 <= uv[1] < height,
            "tcp_plus_10mm_base_axis_xy_pixels": {
                axis: project(position + np.eye(3)[index] * 0.01)
                for index, axis in enumerate(("x", "y", "z"))
            },
            "note": "Projected TCP reference, not detected object coordinates. Pixel x right / y down. "
            "Offsets show local image motion from +0.01m robot-base translation.",
        }
    return result
