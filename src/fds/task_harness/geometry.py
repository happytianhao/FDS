"""Pure arithmetic on public calibration and model-specified geometric assumptions."""

from itertools import product

import numpy as np


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (4, 4):
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
            raise ValueError(f"{name} must be an affine transform")
        matrix = matrix[:3]
    matrix = _array(matrix, (3, 4), name)
    if np.linalg.cond(matrix[:, :3]) > 1e12:
        raise ValueError(f"{name} is singular or ill-conditioned")
    return matrix


def _camera(calibration, image_sizes, name):
    if name not in calibration or name not in image_sizes:
        raise ValueError("Camera needs supplied RGB and public calibration")
    try:
        camera = calibration[name]
        intrinsic = _array(camera["K"], (3, 3), "camera K")
        extrinsic = _transform(camera["world_to_camera"], "world_to_camera")
    except (KeyError, TypeError) as exc:
        raise ValueError("Camera calibration is incomplete") from exc
    if np.linalg.cond(intrinsic) > 1e12:
        raise ValueError("Camera K is singular or ill-conditioned")
    return intrinsic, extrinsic


def _pixel(value, size):
    pixel = _array(value, (2,), "pixel")
    width, height = _array(size, (2,), "image size")
    if min(width, height) < 1 or not (0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1):
        raise ValueError("Pixel is outside its supplied RGB image")
    return np.r_[pixel, 1.0]


def _line(matrix, pixel):
    line = matrix @ pixel
    norm = np.linalg.norm(line[:2])
    tolerance = np.finfo(float).eps * np.linalg.norm(matrix) * np.linalg.norm(pixel)
    if not np.isfinite(line).all() or norm <= tolerance:
        raise ValueError("Point produces a degenerate epipolar line")
    return line / norm


def check_image_correspondence(calibration, image_sizes, arguments):
    """Symmetric pixel-to-epipolar-line distances; never an identity verdict."""
    source, target = arguments["source_camera"], arguments["target_camera"]
    if source == target:
        raise ValueError("Choose two different cameras from the same frame")
    k1, e1 = _camera(calibration, image_sizes, source)
    k2, e2 = _camera(calibration, image_sizes, target)
    p1 = _pixel(arguments["source_pixel"], image_sizes[source])
    p2 = _pixel(arguments["target_pixel"], image_sizes[target])
    rotation = e2[:, :3] @ np.linalg.inv(e1[:, :3])
    translation = e2[:, 3] - rotation @ e1[:, 3]
    scale = np.linalg.norm(translation)
    tolerance = np.finfo(float).eps * max(1.0, np.linalg.norm(e1[:, 3]), np.linalg.norm(e2[:, 3]))
    if not np.isfinite(scale) or scale <= tolerance:
        raise ValueError("Camera baseline is degenerate")
    x, y, z = translation / scale
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    fundamental = np.linalg.inv(k2).T @ cross @ rotation @ np.linalg.inv(k1)
    target_line, source_line = _line(fundamental, p1), _line(fundamental.T, p2)
    distances = np.abs([target_line @ p2, source_line @ p1])
    if not np.isfinite(distances).all():
        raise ValueError("Epipolar residual is numerically invalid")
    return {
        "frame_id": arguments["frame_id"],
        "source_camera": source,
        "target_camera": target,
        "source_pixel": p1[:2].tolist(),
        "target_pixel": p2[:2].tolist(),
        "target_epipolar_line": target_line.tolist(),
        "source_epipolar_line": source_line.tolist(),
        "target_point_to_line_distance_px": float(distances[0]),
        "source_point_to_line_distance_px": float(distances[1]),
        "line_convention": "a*u + b*v + c = 0; sqrt(a*a+b*b)=1",
        "semantics": "Geometric consistency only, not identity, depth, target choice or success. "
        "Apparent centers may lie on different surfaces; multiple objects can satisfy a line. "
        "No acceptance threshold is applied and no image evidence is restored.",
    }


def project_pixel_to_plane(calibration, image_sizes, base_to_world, query):
    """Intersect a pixel ray with an explicitly assumed plane in robot-base coordinates."""
    camera = query["camera"]
    intrinsic, extrinsic = _camera(calibration, image_sizes, camera)
    base = _array(base_to_world, (4, 4), "base_to_world")
    _transform(base, "base_to_world")
    transform = extrinsic @ base
    normal = _array(query["plane_normal_base"], (3,), "plane normal")
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Plane normal must be a unit vector in robot-base coordinates")
    offset, pixel_error, plane_error = _array(
        [query["plane_offset_m"], query["pixel_error_px"], query["plane_offset_error_m"]],
        (3,),
        "plane offset and sensitivity assumptions",
    )
    if min(pixel_error, plane_error) < 0:
        raise ValueError("Sensitivity magnitudes must be nonnegative")
    pixel = _pixel(query["pixel_xy"], image_sizes[camera])[:2]
    origin = -np.linalg.solve(transform[:, :3], transform[:, 3])

    def intersect(uv, d):
        ray = np.linalg.solve(
            transform[:, :3], np.linalg.solve(intrinsic, _pixel(uv, image_sizes[camera]))
        )
        ray /= np.linalg.norm(ray)
        incidence = float(normal @ ray)
        if abs(incidence) < 1e-8:
            raise ValueError("Ray is parallel or nearly parallel to the assumed plane")
        distance = float((d - normal @ origin) / incidence)
        point = origin + distance * ray
        depth = float((transform @ np.r_[point, 1.0])[2])
        if not np.isfinite(point).all() or not np.isfinite(depth):
            raise ValueError("Intersection is numerically invalid")
        if distance <= 0 or depth <= 0:
            raise ValueError("Intersection is behind the camera or at its origin")
        return point, abs(incidence), depth

    point, incidence, depth = intersect(pixel, offset)
    samples, rejected = [], []
    for dx, dy, dd in product(
        sorted({-pixel_error, pixel_error}),
        sorted({-pixel_error, pixel_error}),
        sorted({-plane_error, plane_error}),
    ):
        perturbation = {"pixel_delta_xy": [float(dx), float(dy)], "plane_offset_delta_m": float(dd)}
        try:
            perturbed, _, _ = intersect(pixel + [dx, dy], offset + dd)
            samples.append(perturbed)
        except ValueError as exc:
            rejected.append({**perturbation, "error": str(exc)})
    envelope = np.asarray([point, *samples])
    displacement = float(np.linalg.norm(envelope - point, axis=1).max())
    if not np.isfinite(displacement):
        raise ValueError("Sensitivity calculation is numerically invalid")
    return {
        "frame_id": query["frame_id"],
        "camera": camera,
        "point_robot_base_m": point.tolist(),
        "assumptions": {
            k: query[k]
            for k in (
                "pixel_xy",
                "plane_normal_base",
                "plane_offset_m",
                "pixel_error_px",
                "plane_offset_error_m",
            )
        },
        "conditioning": {"absolute_ray_dot_plane_normal": incidence, "camera_depth_m": depth},
        "sensitivity": {
            "method": "Joint corner perturbations of pixel x/y and plane offset, plus nominal point; sampled envelope, not a confidence interval or a certified bound.",
            "min_robot_base_m": envelope.min(axis=0).tolist(),
            "max_robot_base_m": envelope.max(axis=0).tolist(),
            "max_sampled_displacement_m": displacement,
            "valid_perturbations": len(samples),
            "rejected_perturbations": rejected,
            "complete": not rejected,
        },
        "semantics": "Conditional on the model's pixel and assumed plane. No object detection, true depth, contact, support, identity or task success is inferred. This result does not restore image evidence.",
    }
