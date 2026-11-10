"""Persistent public observations, with explicit history review and fresh-only delivery."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from fds.types import Frame


def sample_every_third(items):
    """Select positions 0, 3, 6, ... and the final item, without a duplicate endpoint."""
    if not items:
        return []
    selected = list(items[::3])
    if (len(items) - 1) % 3:
        selected.append(items[-1])
    return selected


def group_images_by_camera(paths, labels):
    """Keep image/label pairs and each camera's original order; change presentation only."""
    cameras = list(dict.fromkeys(label["camera"] for label in labels))
    pairs = list(zip(paths, labels, strict=True))
    grouped = [(p, label) for camera in cameras for p, label in pairs if label["camera"] == camera]
    return [p for p, _ in grouped], [label for _, label in grouped]


def deduplicate_image_inputs(paths, labels):
    """Send each supplied frame/camera once, preserving all delivery provenance."""
    unique = {}
    for path, label in zip(paths, labels, strict=True):
        key = (label["origin"], label["frame"], label["camera"])
        if key not in unique:
            unique[key] = (path, deepcopy(label))
            continue
        old_path, old_label = unique[key]
        # A frame identity must not silently collapse different image files.
        if old_path != path:
            raise ValueError("Conflicting image paths for one frame/camera identity")
        sources = list(dict.fromkeys([
            *old_label.get("sources", [old_label.get("source")]),
            *label.get("sources", [label.get("source")]),
        ]))
        old_label["sources"] = [source for source in sources if source is not None]
    return [p for p, _ in unique.values()], [label for _, label in unique.values()]


def image_inputs(frames, directory):
    """Materialize only supplied images, preserving separate training frame IDs."""
    paths, labels = [], []
    for frame in frames:
        name = (
            f"frame_{frame.index:06d}"
            if isinstance(frame.index, int)
            else frame.index.replace(":", "_")
        )
        for camera, rgb in frame.images.items():
            path = directory / f"{name}_{camera}.png"
            if not path.exists():
                Image.fromarray(rgb).save(path)
            paths.append(path)
            labels.append(
                {
                    "frame": frame.index,
                    "camera": camera,
                    "cue": frame.cue,
                    "origin": "training_example" if isinstance(frame.index, str) else "episode",
                }
            )
    return group_images_by_camera(paths, labels)


class ObservationWindow:
    """Persistent RGB archive with an active batch and addressable historical windows.

    Only this public-sensor archive is a retrieval source. Audit files and previous
    model inputs remain separate. Closing a window or episode never deletes pixels.
    """

    def __init__(self, output):
        self.directory = Path(output) / "observations"
        (self.directory / "frames").mkdir(parents=True)
        self.frames = {}
        self.windows = {}
        self.archive = {}
        self.example = None
        self.window_id = self.kind = None

    def begin(self, window_id, kind):
        if window_id in self.windows:
            raise ValueError("Observation window IDs must be unique")
        self.expire()
        self.window_id, self.kind = window_id, kind
        self.windows[window_id] = {"kind": kind, "frames": self.frames}
        self._record("windows", {"window_id": window_id, "kind": kind})

    def expire(self):
        # End active selection only; historical windows and files remain available.
        self.frames = {}
        self.window_id = self.kind = None

    def close(self):
        from fds.common.audit import AuditJournal

        AuditJournal.write(self.directory / "index.json", {"windows": self.history()})
        self.expire()
        if self.example is not None:
            self.example.close()

    def _record(self, name, value):
        with (self.directory / f"{name}.jsonl").open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def add(self, frame, step_offset=None):
        if self.window_id is None:
            raise ValueError("Begin an observation window before adding frames")
        if frame.index in self.frames:
            return False
        fresh = frame.index not in self.archive
        if fresh:
            paths = {}
            for index, (camera, rgb) in enumerate(frame.images.items()):
                path = self.directory / "frames" / f"frame_{frame.index:06d}_{index}.png"
                Image.fromarray(rgb).save(path)
                paths[camera] = path
            self.archive[frame.index] = {
                "paths": paths,
                "image_sizes": {
                    camera: [rgb.shape[1], rgb.shape[0]] for camera, rgb in frame.images.items()
                },
                "state": frame.state.copy(),
                "cue": frame.cue,
                "calibration": deepcopy(frame.calibration),
            }
        row = {**self.archive[frame.index], "step_offset": step_offset}
        self.frames[frame.index] = row
        self._record("frames", {
            "window_id": self.window_id, "frame": frame.index, "step_offset": step_offset,
            "images": {camera: str(path.relative_to(self.directory))
                       for camera, path in row["paths"].items()},
            "image_sizes": row["image_sizes"], "state": row["state"].tolist(),
            "cue": row["cue"], "calibration": row["calibration"],
        })
        return fresh

    def geometry_metadata(self, frame_id, delivered, window_id=None):
        """Only frames visible in this request; historical pixels require explicit review."""
        frames = self.archive if window_id is None else self.windows.get(window_id, {}).get("frames", {})
        if frame_id not in frames:
            raise ValueError("Geometry frame is outside the specified observation archive/window")
        if frame_id not in delivered:
            raise ValueError("Geometry frame is not supplied in this request; review it first")
        row = frames[frame_id]
        return deepcopy(row["calibration"]), deepcopy(row["image_sizes"])

    def describe(self, window_id=None):
        window_id = self.window_id if window_id is None else window_id
        window = self.windows.get(window_id, {})
        entries = [(fid, row["cue"], row["step_offset"]) for fid, row in window.get("frames", {}).items()]
        if window_id == "initial" and self.example is not None:
            entries.extend((fid, row["cue"], None) for fid, row in self.example.catalog.items())
        ranges, previous = [], None
        for fid, cue, offset in entries:
            prefix, index = fid.rsplit(":", 1) if isinstance(fid, str) else ("episode", fid)
            index = int(index)
            if previous != (prefix, index - 1, cue):
                ranges.append(
                    {
                        "first_frame": fid,
                        "last_frame": fid,
                        "cue": cue,
                        "origin": "episode" if prefix == "episode" else "training_example",
                        "first_step_offset": offset,
                        "last_step_offset": offset,
                    }
                )
            else:
                ranges[-1].update(last_frame=fid, last_step_offset=offset)
            previous = (prefix, index, cue)
        return {
            "window_id": window_id,
            "kind": window.get("kind"),
            "frame_count": len(entries),
            "frame_ranges": ranges,
        }

    def history(self):
        return [self.describe(window_id) for window_id in self.windows]

    def review(self, window_id, frame_ids, budget):
        if window_id not in self.windows:
            raise ValueError("Unknown observation window; use observation.review_windows")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Review frame IDs must be unique")
        if len(frame_ids) > budget:
            raise ValueError("Review exceeds max_review_frames")
        frames = self.windows[window_id]["frames"]
        examples = self.example.catalog if window_id == "initial" and self.example is not None else {}
        missing = [fid for fid in frame_ids if fid not in frames and fid not in examples]
        if missing:
            raise ValueError(f"Frames are outside the specified observation window: {missing}")
        selected = []
        requested = set(frame_ids)
        for fid, row in frames.items():
            if fid not in requested:
                continue
            images = {}
            for camera, path in row["paths"].items():
                with Image.open(path) as image:
                    images[camera] = np.asarray(image).copy()
            selected.append(
                Frame(fid, images, row["state"].copy(), row["cue"], deepcopy(row["calibration"]))
            )
        if examples:
            selected.extend(self.example.read([fid for fid in frame_ids if fid in examples]))
        return selected
