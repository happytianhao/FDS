"""Trusted episode entry point. Invoked by absolute path with Python -I."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


class ScoredBenchmark:
    """Record official step status independently of the candidate's summary."""

    def __init__(self, benchmark, max_steps):
        self.benchmark = benchmark
        self.max_steps = max_steps
        self.steps = 0
        self.status = "ongoing"
        self.done = False
        self.reset_called = False

    def __getattr__(self, name):
        return getattr(self.benchmark, name)

    def reset(self, episode):
        if self.reset_called:
            raise RuntimeError("Only one reset is allowed per scored episode")
        self.reset_called = True
        return self.benchmark.reset(episode)

    def step(self, action):
        if self.done or self.steps >= self.max_steps:
            raise RuntimeError("No step is allowed after termination or the fixed step budget")
        result = self.benchmark.step(action)
        self.steps += 1
        self.status, self.done = result.status, result.done
        if self.steps >= self.max_steps and not result.done:
            result.truncated = True
            self.done = True
        return result


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    tree, output = Path(job["tree"]), Path(job["output"])
    if job.get("workspace"):
        os.environ["FDS_WORKSPACE"] = job["workspace"]
    config = job["config"]
    validation = bool(config["run"].get("validation_only", False))
    if not validation and not job["allow_experiments"]:
        raise ValueError("Real evaluation requires explicit experiment authorization")
    sys.path.insert(0, str(tree / "src"))
    from fds import cli

    if not Path(cli.__file__).resolve().is_relative_to(tree.resolve()):
        raise RuntimeError("Worker imported the wrong executor source")
    original_make = cli.make_benchmark
    observed = []

    def make_benchmark(name, **kwargs):
        wrapped = ScoredBenchmark(
            original_make(name, **kwargs), config["benchmark"].get("max_steps", 1300)
        )
        observed.append(wrapped)
        return wrapped

    cli.make_benchmark = make_benchmark
    from fds.common.audit import AuditJournal

    provenance = cli.manifest(config)
    provenance["validation_only"] = validation
    provenance["rsi"] = {
        key: job.get(key)
        for key in (
            "version",
            "source_sha256",
            "config_sha256",
            "controller_sha256",
            "task",
            "episode",
        )
    }
    # The runner owns creating a fresh episode directory. Preserve preflight
    # provenance beside the job even if a native crash prevents finalization.
    AuditJournal.write(output.with_suffix(".manifest.json"), provenance)
    start = time.monotonic()
    error, summary = None, {}
    try:
        summary = cli.execute_episode(config, job["task"], job["episode"], output)
    except Exception as exc:
        traceback.print_exc()
        error = {"type": type(exc).__name__, "message": str(exc)}
    output.mkdir(parents=True, exist_ok=True)
    AuditJournal.write(output / "manifest.json", provenance)
    official = observed[0].status if len(observed) == 1 and error is None else "error"
    if official == "error" and error is None:
        error = {"type": "ScoringError", "message": "Missing/ambiguous benchmark or official error"}
    receipt = {
        "version": job["version"],
        "source_sha256": job["source_sha256"],
        "task": job["task"],
        "episode": job["episode"],
        "benchmark": config["benchmark"]["name"],
        "split": config["benchmark"].get("split"),
        "controller_sha256": job.get("controller_sha256"),
        "config_sha256": job["config_sha256"],
        "validation_only": validation,
        "official_status": official,
        "success": official == "success" and error is None,
        "steps": sum(b.steps for b in observed),
        "error": error,
        "planner_calls": summary.get("planner_calls", 0),
        "actor_reported_success": summary.get("success"),
        "stop_reason": summary.get("stop_reason", "error"),
        "timing": cli.episode_timing(output, time.monotonic() - start),
    }
    # Export scorer-corrected summaries for development evidence; actor claims remain explicit.
    summary.update({k: receipt[k] for k in ("success", "official_status", "steps", "timing")})
    summary.update(
        task=job["task"],
        episode=job["episode"],
        method=config["method"],
        benchmark=config["benchmark"]["name"],
        split=config["benchmark"].get("split"),
        actor_reported_success=receipt["actor_reported_success"],
    )
    if error:
        summary.update(stop_reason="error", error=error["message"], error_type=error["type"])
    AuditJournal.write(output / "summary.json", summary)
    AuditJournal.write(output / "receipt.json", receipt)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
