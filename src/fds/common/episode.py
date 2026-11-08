"""Mechanical episode facilities; no decisions, memory, retries or task rules."""

import json

from PIL import Image


class EpisodeIO:
    tool_step_limit = 200

    def __init__(
        self, benchmark, planner, policy=None, *, max_steps=1300, max_calls=100, max_tool_steps=20
    ):
        if min(max_steps, max_calls, max_tool_steps) < 1 or max_tool_steps > self.tool_step_limit:
            raise ValueError("Invalid execution budgets")
        self.env, self.planner, self.policy = benchmark, planner, policy
        self.max_steps, self.max_calls, self.max_tool_steps = max_steps, max_calls, max_tool_steps

    @staticmethod
    def _log(directory, event):
        with (directory / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _images(frames, directory):
        paths, labels = [], []
        for frame in frames:
            for camera, rgb in frame.images.items():
                path = directory / f"frame_{frame.index:06d}_{camera}.png"
                if not path.exists():
                    Image.fromarray(rgb).save(path)
                paths.append(path)
                labels.append({"frame": frame.index, "camera": camera, "cue": frame.cue})
        return paths, labels
