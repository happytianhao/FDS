"""One expiring public-observation window, separate from retained memory and audit."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from fds.types import Frame


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
    """Disk-backed RGB snapshots; never reads prior input files or human audit."""

    def __init__(self, output):
        self.storage = TemporaryDirectory(prefix="observation-window-", dir=output)
        self.directory = Path(self.storage.name)
        self.frames = {}
        self.example = None
        self.window_id = self.kind = None

    def begin(self, window_id, kind):
        self.expire()
        self.window_id, self.kind = window_id, kind

    def expire(self):
        # Delete only files created in this private, episode-scoped directory.
        for path in self.directory.iterdir():
            path.unlink()
        self.frames.clear()
        if self.example is not None:
            self.example.close()
        self.example = None
        self.window_id = self.kind = None

    def close(self):
        self.frames.clear()
        if self.example is not None:
            self.example.close()
        self.example = None
        self.window_id = self.kind = None
        self.storage.cleanup()

    def add(self, frame, step_offset=None):
        if frame.index in self.frames:
            return
        paths = {}
        for index, (camera, rgb) in enumerate(frame.images.items()):
            path = self.directory / f"{len(self.frames)}_{index}.png"
            Image.fromarray(rgb).save(path)
            paths[camera] = path
        self.frames[frame.index] = {
            "paths": paths,
            "image_sizes": {
                camera: [rgb.shape[1], rgb.shape[0]] for camera, rgb in frame.images.items()
            },
            "state": frame.state.copy(),
            "cue": frame.cue,
            "calibration": deepcopy(frame.calibration),
            "step_offset": step_offset,
        }

    def geometry_metadata(self, frame_id, delivered):
        """Only already-delivered, live-window metadata; no image retrieval or new evidence."""
        if self.window_id is None or frame_id not in self.frames:
            raise ValueError("Geometry frame is outside the active observation window")
        if frame_id not in delivered:
            raise ValueError("Geometry frame has not been supplied; review it first")
        row = self.frames[frame_id]
        return deepcopy(row["calibration"]), deepcopy(row["image_sizes"])

    def describe(self):
        entries = [(fid, row["cue"], row["step_offset"]) for fid, row in self.frames.items()]
        if self.example is not None:
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
            "window_id": self.window_id,
            "kind": self.kind,
            "frame_count": len(entries),
            "frame_ranges": ranges,
        }

    def review(self, window_id, frame_ids, current, budget):
        if window_id != self.window_id or self.window_id is None:
            raise ValueError("Observation window expired or unknown; use the current window_id")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Review frame IDs must be unique")
        if len(set(frame_ids) | {current.index}) > budget:
            raise ValueError("Review exceeds the frame budget including the mandatory final frame")
        examples = self.example.catalog if self.example is not None else {}
        missing = [fid for fid in frame_ids if fid not in self.frames and fid not in examples]
        if missing:
            raise ValueError(f"Frames are outside the active observation window: {missing}")
        selected = []
        for fid, row in self.frames.items():
            if fid not in frame_ids or fid == current.index:
                continue
            images = {}
            for camera, path in row["paths"].items():
                with Image.open(path) as image:
                    images[camera] = np.asarray(image).copy()
            selected.append(
                Frame(fid, images, row["state"].copy(), row["cue"], deepcopy(row["calibration"]))
            )
        if self.example is not None:
            selected.extend(self.example.read([fid for fid in frame_ids if fid in examples]))
        selected.append(current)
        return selected
