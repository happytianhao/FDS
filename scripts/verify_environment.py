"""Reset one official training episode to verify the local simulator stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FDS_WORKSPACE", str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def benchmark_source(name: str) -> Path:
    return ROOT / "third_party" / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["robomme_plus", "robomme"])
    parser.add_argument("--task", default="BinFill")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    if args.benchmark == "robomme_plus":
        from fds.benchmarks.robomme_plus import RoboMMEPlus

        adapter = RoboMMEPlus(args.task, args.split, max_steps=10000, calibration=True)
    else:
        from fds.benchmarks.robomme import RoboMME

        adapter = RoboMME(args.task, args.split, max_steps=1300, calibration=True)
    try:
        observation = adapter.reset(args.episode)
        frame = observation.frames[-1]
        result = {
            "status": "ok",
            "benchmark": args.benchmark,
            "task": args.task,
            "split": args.split,
            "episode": args.episode,
            "python": sys.executable,
            "fds_source": str(Path(__import__("fds").__file__).resolve()),
            "benchmark_source": str(benchmark_source(args.benchmark).resolve()),
            "benchmark_commit": subprocess.check_output(
                ["git", "-C", str(benchmark_source(args.benchmark)), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "goal": observation.goal,
            "frames": len(observation.frames),
            "front_rgb_shape": list(frame.images["front"].shape),
            "wrist_rgb_shape": list(frame.images["wrist"].shape),
            "robot_state_shape": list(frame.state.shape),
            "action_dim": adapter.action_spec.dim,
        }
    finally:
        adapter.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
