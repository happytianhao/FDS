"""Optional extra training video, distinct from the current episode's cue/demo."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fds.benchmarks import benchmark_tasks
from fds.common.audit import AuditJournal
from fds.common.paths import workspace_root
from fds.task_harness.observation import sample_every_third


@dataclass
class TrainingFrame:
    index: str
    images: dict
    cue: bool


class TrainingVideo:
    """Verified training RGB, copied losslessly for persistent history review."""

    def __init__(self, filename, metadata, record, benchmark, task, episode, max_frames, output):
        import h5py

        self.filename, self.record = filename, record
        self.task, self.episode, self.max_frames = task, episode, max_frames
        self.closed = False
        self.file_identity = self._file_identity()
        with h5py.File(filename, "r") as data:
            group = self._verified_group(data)
            self.goal = self._string(group["setup"]["task_goal"][()])
            indices = sorted(
                int(k.removeprefix("timestep_")) for k in group if k.startswith("timestep_")
            )
            if not indices:
                raise ValueError("Training example contains no video frames")
            self.catalog = {
                f"training:{episode}:{index}": {
                    "index": index,
                    "cue": bool(
                        np.asarray(group[f"timestep_{index}"]["info/is_video_demo"][()]).reshape(
                            -1
                        )[0]
                    ),
                }
                for index in indices
            }
        self.context = {
            "benchmark": benchmark,
            "split": "train",
            "task": task,
            "episode": episode,
            "goal": self.goal,
            "frame_count": len(self.catalog),
            "contract": "Extra training video, never current state or current progress. Frame IDs use training:<episode>:<index>. RGB and cue flags only, no oracle actions, subgoals or hidden state. All video frames are archived. The first request receives every third frame from the first, plus the final frame. Later requests receive training images only through retained keyframes or explicit review_frames of the initial window. Training references always describe demonstration evidence, not this episode's execution.",
        }
        directory = output / "training_example"
        directory.mkdir()
        AuditJournal.write(
            directory / "source.json",
            {"file": str(filename), "training_metadata": str(metadata), **self.context},
        )
        # Preserve the complete public video, including frames never sent to Sys2.
        # Retrieval must not depend on the source dataset still being mounted.
        (directory / "frames").mkdir()
        with h5py.File(self.filename, "r") as data:
            group = self._verified_group(data)
            for fid, entry in self.catalog.items():
                step = group[f"timestep_{entry['index']}"]
                entry["paths"] = {}
                for camera in ("front", "wrist"):
                    rgb = np.asarray(step[f"obs/{camera}_rgb"][()])
                    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                        raise ValueError("Example RGB must be uint8 [height,width,3]")
                    path = directory / "frames" / f"{fid.replace(':', '_')}_{camera}.png"
                    Image.fromarray(rgb).save(path)
                    entry["paths"][camera] = path
        if self.file_identity != self._file_identity():
            raise ValueError("Training example changed while archiving")
        AuditJournal.write(directory / "frames.json", {
            fid: {"index": row["index"], "cue": row["cue"],
                  "images": {camera: str(path.relative_to(directory))
                             for camera, path in row["paths"].items()}}
            for fid, row in self.catalog.items()
        })

    @staticmethod
    def _string(value):
        value = np.asarray(value).reshape(-1)[0]
        return value.decode() if isinstance(value, bytes) else str(value)

    def _file_identity(self):
        stat = self.filename.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def close(self):
        self.closed = True

    def _verified_group(self, data):
        group = data[f"episode_{self.episode}"]
        setup = group["setup"]
        if int(np.asarray(setup["seed"][()]).reshape(-1)[0]) != self.record["seed"]:
            raise ValueError("Example seed does not match the official training manifest")
        if self._string(setup["task"][()]) != self.task:
            raise ValueError("Example task does not match the official training manifest")
        if hasattr(self, "goal") and self._string(setup["task_goal"][()]) != self.goal:
            raise ValueError("Training example changed during the episode")
        return group

    def initial(self):
        return self.read(sample_every_third(list(self.catalog)))

    def read(self, frame_ids):
        if self.closed:
            raise ValueError("Training example reader is closed")
        if not frame_ids:
            return []
        if any(fid not in self.catalog for fid in frame_ids):
            raise ValueError("Frame is outside the verified training example")
        frames = []
        requested = set(frame_ids)
        for fid, entry in self.catalog.items():
            if fid not in requested:
                continue
            images = {}
            for camera, path in entry["paths"].items():
                with Image.open(path) as image:
                    images[camera] = np.asarray(image).copy()
            frames.append(TrainingFrame(fid, images, entry["cue"]))
        return frames


def example_loader(options, benchmark, task):
    options = dict(options)
    enabled = options.pop("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("examples.enabled must be boolean")
    root = options.pop("training_root", None)
    episode = options.pop("episode", 0)
    max_frames = options.pop("max_frames", 16)
    if (
        options
        or type(episode) is not int
        or episode < 0
        or type(max_frames) is not int
        or not 1 <= max_frames <= 64
    ):
        raise ValueError("Invalid training example options")
    if not enabled:
        return None
    if (
        not root
        or benchmark not in ("robomme", "robomme_plus")
        or task not in benchmark_tasks(benchmark)
    ):
        raise ValueError("Training examples require a supported benchmark, task and training_root")
    directory = Path(root).expanduser()
    if not directory.is_absolute():
        directory = workspace_root() / directory
    filename = directory / "h5" / f"record_dataset_{task}.h5"
    if not filename.is_file():
        filename = directory / f"record_dataset_{task}.h5"
    # Read only the official TRAIN manifest. Never infer a split from an H5 filename.
    metadata = (
        workspace_root()
        / f"third_party/{benchmark}/src/{benchmark}/env_metadata/train/record_dataset_{task}_metadata.json"
    )
    records = json.loads(metadata.read_text())["records"]
    if episode >= len(records) or records[episode]["episode"] != episode:
        raise ValueError("Training example is outside the official training split")
    record = records[episode]

    def load(output):
        return TrainingVideo(
            filename, metadata, record, benchmark, task, episode, max_frames, output
        )

    return load
