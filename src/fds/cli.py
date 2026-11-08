from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fds.benchmarks import benchmark_tasks, make_benchmark, validate_executor
from fds.common.recovery import (
    RecoveringPlanner,
    effective_recovery,
    planner_recovery_options,
    request_resume,
)
from fds.planners import (
    ChatPlanner,
    CodexPlanner,
    MockPlanner,
    ResponsesPlanner,
)
from fds.task_harness.executor import EXECUTORS


def git_state(path):
    try:

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()

        git("rev-parse", "--show-toplevel")
        try:
            commit = git("rev-parse", "--verify", "HEAD")
        except subprocess.CalledProcessError:
            commit = None
        return {"commit": commit, "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest(config):
    modules = {}
    for name in (
        "fds",
        "robomme",
        "robomme_plus",
        "mani_skill",
        "sapien",
        "torch",
    ):
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            modules[name] = {"path": spec.origin, "git": git_state(Path(spec.origin).parent)}
    packages = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions()}
    package_root = Path(__file__).resolve().parent
    source_hashes = {
        str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package_root.rglob("*.py"))
    }
    return {
        "config": config,
        "python": platform.python_version(),
        "packages": packages,
        "fds_source_sha256": source_hashes,
        "modules": modules,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed_note": "run.seed seeds Python/NumPy; benchmark scenes use official instance seeds",
        "planner_deterministic": config["planner"]["backend"] == "mock",
    }


def doctor():
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "PIL", "jsonschema", "robomme", "robomme_plus", "openai")
    }
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "modules": modules,
                "codex_available": shutil.which("codex") is not None,
                "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
                "gpu_tool_available": shutil.which("nvidia-smi") is not None,
            },
            indent=2,
        )
    )


def resolve_executor_defaults(config):
    """Resolve the single task-harness configuration before recording a run."""
    config = copy.deepcopy(config)
    method = config.get("method", "task_harness")
    if method not in EXECUTORS:
        raise ValueError(f"Unknown executor: {method}")
    if method == "task_harness":
        if "baseline" in config:
            raise ValueError("task_harness does not accept baseline options")
        config.setdefault("task_harness", {})
        fds = config.setdefault("fds", {})
        if not isinstance(fds, dict):
            raise ValueError("fds must be a table")
        unknown = set(fds) - {"enabled", "contract_mode"}
        if unknown:
            raise ValueError(f"Unknown fds options: {sorted(unknown)}")
        enabled = fds.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("fds.enabled must be a boolean")
        mode = fds.get("contract_mode", "full" if enabled else "off")
        if mode not in ("off", "full"):
            raise ValueError("fds.contract_mode must be 'off' or 'full'")
        if not enabled:
            mode = "off"
        fds["enabled"] = mode != "off"
        fds["contract_mode"] = mode
        from fds.task_harness.skills import validate_skills

        validate_skills(config["task_harness"].get("skills", []))
        config.setdefault("run", {}).setdefault("max_tool_steps", 400)
        return config


def execute_episode(config, task, episode, output):
    config = resolve_executor_defaults(config)
    method = config.get("method", "task_harness")
    if "memory" in config or "execution" in config:
        raise ValueError("Legacy memory/execution configurations have been removed")
    if "harness" in config or "baseline" in config:
        raise ValueError("FDS only accepts task_harness options")
    if config.get("policy", {}).get("enabled", False):
        raise ValueError("FDS task_harness does not use a VLA policy")
    benchmark_config = dict(config["benchmark"])
    name = benchmark_config.pop("name")
    validate_executor(name, method, policy_enabled=config.get("policy", {}).get("enabled", False))
    benchmark_config["task"] = task
    planner_config = dict(config["planner"])
    backend = planner_config.pop("backend")
    recovery = planner_recovery_options(config)
    # The shared wrapper owns attempts. Never multiply these by backend retries.
    for key in ("transport_retries", "retry_delay_seconds", "retry_max_delay_seconds"):
        planner_config.pop(key, None)
    constructors = {
        "chat": ChatPlanner,
        "codex": CodexPlanner,
        "responses": ResponsesPlanner,
        "mock": MockPlanner,
    }
    if backend not in constructors:
        raise ValueError(f"Unknown planner backend: {backend}")
    run_config = config["run"]
    random.seed(run_config.get("seed", 7))
    np.random.seed(run_config.get("seed", 7))
    planner = None
    try:
        planner = constructors[backend](**planner_config)
        if backend != "mock" and recovery.transport_retries:
            planner = RecoveringPlanner(planner, recovery)
        from fds.task_harness.examples import example_loader
        from fds.task_harness.runtime import TaskHarness

        runner_type, runner_options = (
            TaskHarness,
            {
                "options": config["task_harness"],
                "fds_options": config["fds"],
                "examples": example_loader(config.get("examples", {}), name, task),
            },
        )
        benchmark = make_benchmark(name, **benchmark_config)
        if hasattr(benchmark, "set_executor"):
            benchmark.set_executor(method)
        if hasattr(benchmark, "set_output"):
            benchmark.set_output(output)
        if hasattr(benchmark, "set_validation_only"):
            benchmark.set_validation_only(
                run_config.get("validation_only", False) or backend == "mock"
            )
        try:
            runner = runner_type(
                benchmark,
                planner,
                None,
                max_steps=benchmark_config.get("max_steps", 1300),
                max_calls=run_config.get("max_calls", 100),
                max_tool_steps=run_config.get("max_tool_steps", 20),
                **runner_options,
            )
        except BaseException:
            benchmark.close()
            raise
        summary = runner.run(episode, output)
        if hasattr(benchmark, "benchmark_metrics"):
            summary["benchmark_metrics"] = benchmark.benchmark_metrics()
        summary.update(
            method=method,
            benchmark=name,
            split=benchmark_config.get("split"),
            task=task,
            episode=episode,
        )
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if planner is not None and hasattr(planner, "close"):
            planner.close()


def isolated_episode(
    manifest_path,
    task,
    episode,
    output,
    timeout,
    *,
    gpu=None,
    log_directory=None,
    cancel_event=None,
):
    """Contain native simulator crashes and reclaim GPU resources per episode."""
    log_directory = log_directory or manifest_path.parent / "workers"
    log_directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "fds.cli",
        "_episode",
        "--manifest",
        str(manifest_path),
        "--task",
        task,
        "--episode",
        str(episode),
        "--output",
        str(output),
    ]
    with (
        (log_directory / f"{task}_{episode}.stdout.log").open("w") as out,
        (log_directory / f"{task}_{episode}.stderr.log").open("w") as err,
    ):
        environment = os.environ.copy()
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            command, stdout=out, stderr=err, start_new_session=True, env=environment
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Experiment batch interrupted")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            # SIGTERM lets the worker close a running planner child before force-kill.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if process.returncode:
        raise RuntimeError(
            f"Episode worker exited with code {process.returncode}; see {log_directory}"
        )
    return json.loads((output / "summary.json").read_text())


def episode_failure(output, episode, exc):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"episode": episode}
    if summary.get("stop_reason") == "error":
        summary["runner_error"] = str(exc)
    else:
        summary.update(
            success=False, stop_reason="error", error_type=type(exc).__name__, error=str(exc)
        )
    path.write_text(json.dumps(summary, indent=2))
    return summary


def episode_timing(output, wall_seconds):
    model_seconds = 0.0
    prompt_tokens, completion_tokens, timed_calls = 0, 0, 0
    request_attempts, transport_retries, retry_wait_seconds = 0, 0, 0.0
    recovery_pause_seconds, unknown_attempt_usage = 0.0, 0
    for path in output.glob("call_*/stats.json"):
        stats = json.loads(path.read_text())
        model_seconds += stats.get("seconds", 0.0)
        usage = stats.get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        completion_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        timed_calls += 1
        request_attempts += stats.get("request_attempts", 1)
        transport_retries += stats.get("transport_retries", 0)
        retry_wait_seconds += stats.get("retry_wait_seconds", 0.0)
        recovery_pause_seconds += stats.get("recovery_pause_seconds", 0.0)
        unknown_attempt_usage += stats.get("failed_attempt_usage_unknown", 0)
    policy_seconds, policy_calls = 0.0, 0
    trace = output / "trace.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "policy_inference":
                policy_seconds += event["seconds"]
                policy_calls += 1
    return {
        "wall_seconds": wall_seconds,
        "model_seconds": model_seconds,
        "policy_seconds": policy_seconds,
        "policy_calls": policy_calls,
        "other_seconds": max(0.0, wall_seconds - model_seconds - policy_seconds),
        "model_calls_timed": timed_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_request_attempts_recorded": request_attempts,
        "transport_retries": transport_retries,
        "retry_wait_seconds": retry_wait_seconds,
        "recovery_pause_seconds": recovery_pause_seconds,
        "failed_attempt_usage_unknown": unknown_attempt_usage,
    }


def save_results(root, summaries, tasks, name, validation_only, planned_episodes, wall_seconds):
    metrics = {
        "benchmark": name,
        "validation_only": validation_only,
        "status": "completed" if len(summaries) == planned_episodes else "running",
        "planned_episodes": planned_episodes,
        "episodes": len(summaries),
        "successes": sum(s["success"] for s in summaries),
        "errors": sum(
            s["stop_reason"] == "error" or s.get("official_status") == "error" for s in summaries
        ),
        "success_rate": sum(s["success"] for s in summaries) / len(summaries)
        if summaries
        else None,
        "wall_seconds": wall_seconds,
        "episode_wall_seconds": sum(s["timing"]["wall_seconds"] for s in summaries),
        "model_seconds": sum(s["timing"]["model_seconds"] for s in summaries),
        "policy_seconds": sum(s["timing"]["policy_seconds"] for s in summaries),
        "policy_calls": sum(s["timing"]["policy_calls"] for s in summaries),
        "environment_steps": sum(s.get("steps", 0) for s in summaries),
        "planner_calls": sum(s.get("planner_calls", 0) for s in summaries),
        "prompt_tokens": sum(s["timing"]["prompt_tokens"] for s in summaries),
        "completion_tokens": sum(s["timing"]["completion_tokens"] for s in summaries),
        "model_request_attempts_recorded": sum(
            s["timing"].get("model_request_attempts_recorded", 0) for s in summaries
        ),
        "transport_retries": sum(s["timing"].get("transport_retries", 0) for s in summaries),
        "retry_wait_seconds": sum(s["timing"].get("retry_wait_seconds", 0) for s in summaries),
        "recovery_pause_seconds": sum(
            s["timing"].get("recovery_pause_seconds", 0) for s in summaries
        ),
        "failed_attempt_usage_unknown": sum(
            s["timing"].get("failed_attempt_usage_unknown", 0) for s in summaries
        ),
        "per_task": {
            task: {
                "episodes": sum(s["task"] == task for s in summaries),
                "successes": sum(s["success"] for s in summaries if s["task"] == task),
                "wall_seconds": sum(
                    s["timing"]["wall_seconds"] for s in summaries if s["task"] == task
                ),
            }
            for task in tasks
        },
    }
    # Atomic replacement keeps partial-batch monitoring readable during long runs.
    for filename, data in (("metrics.json", metrics), ("episodes.json", summaries)):
        temporary = root / (filename + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        temporary.replace(root / filename)
    columns = [
        "task",
        "episode",
        "success",
        "official_status",
        "stop_reason",
        "steps",
        "planner_calls",
        "wall_seconds",
        "model_seconds",
        "policy_seconds",
        "policy_calls",
        "other_seconds",
        "prompt_tokens",
        "completion_tokens",
    ]
    with (root / "episodes.csv.tmp").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **summary,
                    **summary["timing"],
                    **{
                        key: value
                        for key, value in summary.get("benchmark_metrics", {}).items()
                        if key in ()
                    },
                }
            )
    (root / "episodes.csv.tmp").replace(root / "episodes.csv")
    return metrics


def run(args):
    batch_start = time.perf_counter()
    config_path = args.config.resolve()
    config = resolve_executor_defaults(tomllib.loads(config_path.read_text()))
    config["recovery"] = effective_recovery(config)
    benchmark_config = dict(config["benchmark"])
    name = benchmark_config.pop("name")
    if args.task:
        benchmark_config["task"] = args.task
    available_tasks = benchmark_tasks(name)
    tasks = (
        list(available_tasks)
        if benchmark_config.get("task") == "All"
        else [benchmark_config.get("task", "mock")]
    )
    run_config = config["run"]
    if not args.task and "tasks" in run_config:
        tasks = run_config["tasks"]
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(
            isinstance(t, str) and t and t not in (".", "..") and "/" not in t and "\\" not in t
            for t in tasks
        )
        or len(set(tasks)) != len(tasks)
        or (name != "mock" and any(t not in available_tasks for t in tasks))
    ):
        raise ValueError("run.tasks must contain unique valid task names")
    episodes = args.episodes if args.episodes is not None else run_config.get("episodes", [0])
    if (
        not episodes
        or any(type(ep) is not int or ep < 0 for ep in episodes)
        or len(set(episodes)) != len(episodes)
    ):
        raise ValueError("Episode IDs must be unique nonnegative integers")
    random.seed(run_config.get("seed", 7))
    np.random.seed(run_config.get("seed", 7))
    root = args.output or Path(run_config.get("output", "runs")) / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S_%f"
    )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    effective = {
        **config,
        "benchmark": {"name": name, **benchmark_config},
        "run": {**run_config, "episodes": episodes, "output": str(root)},
    }
    metadata = manifest(effective)
    metadata["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    metadata["validation_only"] = (
        name == "mock"
        or config["planner"]["backend"] == "mock"
        or run_config.get("validation_only", False)
    )
    isolate = run_config.get("isolate_episodes", name in ("robomme", "robomme_plus"))
    metadata["episode_process_isolation"] = isolate
    workers = run_config.get("workers", [])
    if workers:
        if not isolate:
            raise ValueError("GPU workers require isolated episodes")
        if config["planner"]["backend"] != "chat" and any("base_url" in w for w in workers):
            raise ValueError("Worker base_url overrides require the chat planner")
        if len({str(w["gpu"]) for w in workers}) != len(workers):
            raise ValueError("Assign each parallel worker a different GPU")
        if config.get("policy", {}).get("enabled") and not config["policy"].get(
            "require_session_isolation"
        ):
            addresses = [(w.get("policy_host", "127.0.0.1"), w.get("policy_port")) for w in workers]
            if any(port is None for _, port in addresses) or len(set(addresses)) != len(addresses):
                raise ValueError(
                    "Stateful policy servers require a distinct policy_port per worker"
                )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2))
    summaries = []
    planned = len(tasks) * len(episodes)
    save_results(
        root,
        summaries,
        tasks,
        name,
        metadata["validation_only"],
        planned,
        time.perf_counter() - batch_start,
    )

    def emit(event):
        event = {"utc": datetime.now(UTC).isoformat(), **event}
        text = json.dumps(event, ensure_ascii=False)
        with (root / "experiment.log").open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
        print(text, flush=True)

    cancel_event = threading.Event()

    def job(task, episode, worker=None):
        output = root / task / f"episode_{episode:04d}"
        episode_start = time.perf_counter()
        try:
            if worker is not None:
                job_metadata = copy.deepcopy(metadata)
                job_metadata["config"]["benchmark"]["task"] = task
                job_metadata["config"]["run"]["episodes"] = [episode]
                if "base_url" in worker:
                    job_metadata["config"]["planner"]["base_url"] = worker["base_url"]
                if (
                    job_metadata["config"].get("policy", {}).get("enabled")
                    and "policy_port" in worker
                ):
                    job_metadata["config"]["policy"].update(
                        host=worker.get("policy_host", "127.0.0.1"), port=worker["policy_port"]
                    )
                job_metadata["worker"] = worker
                job_manifest = root / "workers" / f"{task}_{episode}.manifest.json"
                job_manifest.parent.mkdir(exist_ok=True)
                job_manifest.write_text(json.dumps(job_metadata, indent=2))
                summary = isolated_episode(
                    job_manifest,
                    task,
                    episode,
                    output,
                    run_config.get("episode_timeout", 21600),
                    gpu=worker["gpu"],
                    log_directory=root / "workers",
                    cancel_event=cancel_event,
                )
            else:
                summary = (
                    isolated_episode(
                        manifest_path,
                        task,
                        episode,
                        output,
                        run_config.get("episode_timeout", 21600),
                    )
                    if isolate
                    else execute_episode(effective, task, episode, output)
                )
        except Exception as exc:
            summary = episode_failure(output, episode, exc)
        summary["timing"] = episode_timing(output, time.perf_counter() - episode_start)
        if worker is not None:
            summary["worker"] = worker
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return {"task": task, **summary}

    def record(summary):
        summaries.append(summary)
        save_results(
            root,
            summaries,
            tasks,
            name,
            metadata["validation_only"],
            planned,
            time.perf_counter() - batch_start,
        )
        event = {
            "event": "episode_finished",
            "task": summary["task"],
            "episode": summary["episode"],
            "success": summary["success"],
            "stop_reason": summary["stop_reason"],
            "wall_seconds": summary["timing"]["wall_seconds"],
            "worker": summary.get("worker"),
        }
        if summary["stop_reason"] == "error":
            event.update(
                {
                    key: summary[key]
                    for key in ("error_type", "error", "runner_error")
                    if key in summary
                }
            )
            event["summary_path"] = str(
                root / summary["task"] / f"episode_{summary['episode']:04d}" / "summary.json"
            )
        emit(event)

    emit(
        {
            "event": "batch_started",
            "planned_episodes": planned,
            "parallel_workers": len(workers) or 1,
        }
    )
    jobs = iter((task, episode) for task in tasks for episode in episodes)
    try:
        if workers:
            with ThreadPoolExecutor(max_workers=len(workers)) as executor:
                pending = {}

                def submit(worker):
                    item = next(jobs, None)
                    if item is not None:
                        task, episode = item
                        emit(
                            {
                                "event": "episode_started",
                                "task": task,
                                "episode": episode,
                                "worker": worker,
                            }
                        )
                        pending[executor.submit(job, task, episode, worker)] = worker

                try:
                    for worker in workers:
                        submit(worker)
                    while pending:
                        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in finished:
                            worker = pending.pop(future)
                            record(future.result())
                            submit(worker)
                except BaseException:
                    cancel_event.set()
                    raise
        else:
            for task, episode in jobs:
                emit({"event": "episode_started", "task": task, "episode": episode})
                record(job(task, episode))
    except BaseException:
        metrics = save_results(
            root,
            summaries,
            tasks,
            name,
            metadata["validation_only"],
            planned,
            time.perf_counter() - batch_start,
        )
        metrics["status"] = "interrupted"
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        emit({"event": "batch_interrupted", "completed_episodes": len(summaries)})
        raise
    metrics = save_results(
        root,
        summaries,
        tasks,
        name,
        metadata["validation_only"],
        planned,
        time.perf_counter() - batch_start,
    )
    emit(
        {
            "event": "batch_finished",
            "wall_seconds": metrics["wall_seconds"],
            "successes": metrics["successes"],
            "errors": metrics["errors"],
        }
    )
    print(json.dumps(metrics, indent=2))
    return 1 if metrics["errors"] else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rsi":
        from fds.meta_harness.cli import main as rsi_main

        return rsi_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description="FDS: falsifiable decision-state experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check dependencies without reading credentials")
    resume_parser = sub.add_parser(
        "resume", help="Retry live requests waiting after transport failures; no simulator restart"
    )
    resume_parser.add_argument(
        "directory", type=Path, help="Run, episode, or call directory of a still-running worker"
    )
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--task", help="Task name for the selected benchmark, or All")
    run_parser.add_argument("--episodes", nargs="+", type=int)
    run_parser.add_argument(
        "--output", type=Path, help="Fresh output directory; existing data is never overwritten"
    )
    worker_parser = sub.add_parser("_episode", help=argparse.SUPPRESS)
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--task", required=True)
    worker_parser.add_argument("--episode", type=int, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
        return 0
    if args.command == "resume":
        result = request_resume(args.directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["requested"] else 1
    if args.command == "_episode":

        def terminated(signum, frame):
            raise TimeoutError("Episode worker terminated by parent")

        signal.signal(signal.SIGTERM, terminated)
        config = json.loads(args.manifest.read_text())["config"]
        try:
            execute_episode(config, args.task, args.episode, args.output)
        except Exception as exc:
            traceback.print_exc()
            episode_failure(args.output, args.episode, exc)
            return 1
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
