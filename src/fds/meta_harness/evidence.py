"""Offline improvement evidence, kept separate from executor observations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from fds.meta_harness.store import (
    digest,
    file_hash,
    read,
    require_training_splits,
    suite_cases,
    write,
)
from fds.task_harness.executor import recorded_executor


def outcome(summary):
    if summary.get("stop_reason") == "error":
        return "error"
    if summary.get("official_status") == "success" and summary.get("success"):
        return "success"
    if summary.get("official_status") == "fail":
        return "fail"
    if summary.get("official_status") == "timeout":
        return "budget_exhausted"
    return summary.get("stop_reason", "error")


def summarize_episode(directory, method=None):
    directory = Path(directory).resolve()
    summary = read(directory / "summary.json")
    manifest = next(
        (
            p / "manifest.json"
            for p in (directory, *directory.parents)
            if (p / "manifest.json").is_file()
        ),
        None,
    )
    config = read(manifest).get("config", {}) if manifest else {}
    actual_method = recorded_executor(
        summary.get("method", config.get("method", method)),
        summary.get("memory_enabled", config.get("baseline", {}).get("memory_enabled", False)),
    )
    if method and actual_method != method:
        raise ValueError(f"Method mismatch in {directory}")
    calls, errors = [], Counter()
    paths = sorted((directory / "audit").glob("call_*/decision.json"))
    for p in paths:
        feedback_path = p.with_name("outcome.json")
        feedback = read(feedback_path) if feedback_path.exists() else {}
        for item in (
            feedback.get("feedback")
            if isinstance(feedback.get("feedback"), list)
            else [feedback.get("feedback")]
        ):
            if isinstance(item, dict) and item.get("error"):
                errors[str(item["error"])] += 1
    # Initial intent and the last four decisions expose forgetting/repetition without a full dump.
    selected = list(dict.fromkeys(paths[:1] + paths[-4:]))
    for p in selected:
        decision = read(p)
        feedback = read(p.with_name("outcome.json")) if p.with_name("outcome.json").exists() else {}
        calls.append({"audit": str(p.parent), "decision": decision, "outcome": feedback})
    record = {
        "role": "development",
        "source": str(directory),
        "method": actual_method,
        "benchmark": config.get("benchmark", {}).get("name"),
        "split": summary.get("split", config.get("benchmark", {}).get("split")),
        "task": summary.get("task", directory.parent.name),
        "episode": summary["episode"],
        "summary_sha256": file_hash(directory / "summary.json"),
        "outcome": outcome(summary),
        "summary": {
            k: summary.get(k)
            for k in (
                "success",
                "official_status",
                "stop_reason",
                "steps",
                "planner_calls",
                "task_goal",
                "error_type",
                "error",
                "timing",
                "benchmark_metrics",
            )
        },
        "repeated_errors": errors.most_common(8),
        "audit_excerpt": calls,
        "note": "Offline improver evidence only. Model memory claims are not official truth.",
    }
    record["id"] = "ev_" + digest(record)[:24]
    return record


def collect(store, inputs):
    require_training_splits(store.config)
    candidates = []
    for item in inputs:
        path = Path(item).resolve()
        if path.is_file() and path.name != "summary.json":
            values = read(path)
            if not isinstance(values, list):
                raise ValueError("Selection JSON must be a list of source/method records")
            for value in values:
                if value["method"] == store.config["executor"]:
                    source = Path(value["source"])
                    if not source.is_absolute():
                        source = Path(store.session["repo"]) / source
                    candidates.append((source, value["method"]))
        elif path.name == "summary.json":
            candidates.append((path.parent, None))
        elif (path / "summary.json").is_file():
            candidates.append((path, None))
        elif path.is_dir():
            candidates.extend((p.parent, None) for p in sorted(path.rglob("summary.json")))
        else:
            raise ValueError(f"Evidence input not found: {path}")
    allowed = set(suite_cases(store.config, "development"))
    records, seen = [], set()
    for directory, hint in candidates:
        record = summarize_episode(directory, hint)
        if record["method"] != store.config["executor"]:
            continue
        key = (record["task"], record["episode"])
        if key not in allowed:
            raise ValueError(f"{key} is not in development; validation traces cannot be imported")
        if record["benchmark"] != store.session["benchmark"]:
            raise ValueError("Evidence benchmark differs from the frozen session")
        expected_split = store.config["development"].get("split")
        if expected_split is not None and record["split"] != expected_split:
            raise ValueError(
                "Evidence split is not development; held-out traces cannot be imported"
            )
        if key in seen:
            raise ValueError(f"Duplicate {key}; provide an explicit selection, not best-of retries")
        seen.add(key)
        records.append(record)
    if not records:
        raise ValueError("No matching development evidence found")
    for record in records:
        write(store.root / "evidence" / (record["id"] + ".json"), record)
    store.event("evidence_collected", evidence_ids=[r["id"] for r in records])
    return [r["id"] for r in records]
