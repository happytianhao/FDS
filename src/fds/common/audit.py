"""Offline-only evidence journal. Nothing here is a Sys2 retrieval source."""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image


@dataclass
class AuditConfig:
    enabled: bool = False
    capture_all_frames: bool = True


class AuditJournal:
    def __init__(self, output, config=None):
        self.config = AuditConfig(**(config or {}))
        self.output = Path(output)
        self.root = self.output / "audit"
        self.seen = set()
        self.active = None
        if self.config.enabled:
            (self.root / "frames").mkdir(parents=True)

    @staticmethod
    def write(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
        temporary.replace(path)

    def capture(self, frames, step):
        if not self.config.enabled or not self.config.capture_all_frames:
            return
        for frame in frames:
            if frame.index in self.seen:
                continue
            paths = {}
            for index, (camera, rgb) in enumerate(frame.images.items()):
                path = self.root / "frames" / f"frame_{frame.index:06d}_camera_{index}.png"
                Image.fromarray(rgb).save(path)
                paths[camera] = str(path.relative_to(self.output))
            item = {
                "frame": frame.index,
                "step": step,
                "cue": frame.cue,
                "images": paths,
                "robot_state": frame.state.tolist(),
                "audience": "human_audit_only",
            }
            with (self.root / "frames.jsonl").open("a") as stream:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            self.seen.add(frame.index)

    def begin(self, call, context, images, state, *, logical_images=None):
        if not self.config.enabled:
            return
        from fds.decision import decision_schema
        from fds.planners import make_prompt

        directory = self.root / f"call_{call - 1:04d}"
        directory.mkdir()
        self.active = (directory, time.monotonic())

        def manifest(paths, labels):
            return [
                {
                    "path": str(Path(path).relative_to(self.output)),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "label": label,
                }
                for path, label in zip(paths, labels, strict=True)
            ]

        entry = {
            "call": call,
            "utc": datetime.now(UTC).isoformat(),
            "context": context,
            "images": manifest(images, context.get("image_presentations", context["image_labels"])),
        }
        if logical_images is not None:
            # Human provenance only: these originals occur as pixels in the actual
            # attachments above; they are NOT additional model attachments.
            entry["logical_images"] = manifest(logical_images, context["image_labels"])
        self.write(directory / "input.json", entry)
        self.write(directory / "before.json", state)
        self.write(directory / "schema.json", decision_schema(context))
        (directory / "prompt.txt").write_text(make_prompt(context))

    def decision(self, value):
        if self.active:
            self.write(self.active[0] / "decision.json", value)

    def finish(self, state, feedback, error=None):
        if not self.active:
            return
        directory, started = self.active
        self.write(directory / "after.json", state)
        outcome = {
            "seconds_including_execution": time.monotonic() - started,
            "feedback": feedback,
            "error": error,
        }
        if error and not (directory / "decision.json").exists() and feedback:
            # decide() failed before the runtime could start this call's feedback list.
            outcome.update(feedback=[], feedback_carried_from_prior_call=feedback)
        self.write(directory / "outcome.json", outcome)
        self.active = None
