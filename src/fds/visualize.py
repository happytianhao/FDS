"""Offline research viewer: exact model inputs, memory changes and human-only replay.

Usage: python -m fds.visualize RUN_DIRECTORY [--watch]
One browser viewer per run; reuse original images and raw audit files.
No planner, simulator, retrieval tool or network service is called by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from fds.replay_video import build_replay
from fds.task_harness.executor import recorded_executor

RENDERER_SOURCE = Path(__file__).read_bytes()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_lines(path):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    values = []
    for line in lines:
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # An in-progress writer may not yet have finished its last line.
    return values


def link(path, directory):
    return quote(os.path.relpath(path, directory), safe="/")


def text_file(path):
    return path.read_text() if path.exists() else ""


def manifest_path(directory):
    """CLI batches and RSI workers store provenance at different nesting levels."""
    return next(
        (
            parent / "manifest.json"
            for parent in (directory, *directory.parents)
            if (parent / "manifest.json").is_file()
        ),
        None,
    )


def episode_data(episode, destination):
    """Read immutable evidence. Missing measurements stay unknown, never synthetic zero."""
    episode, destination = Path(episode), Path(destination)
    frames = read_lines(episode / "audit/frames.jsonl")
    for frame in frames:
        frame["images"] = {
            camera: link(episode / path, destination) for camera, path in frame["images"].items()
        }
    events, by_call = read_lines(episode / "trace.jsonl"), {}
    current = 0
    for event in events:
        if event["event"] in ("observation_delivery", "decision"):
            current = event["call"]
        by_call.setdefault(current, []).append(event)
    calls, issues, hashes, fresh_seen, segment_seen = [], [], {}, {}, {}
    for directory in sorted((episode / "audit").glob("call_*")):
        inp = read_json(directory / "input.json", {})
        if not inp:
            continue
        number = inp["call"]
        segment = inp.get("context", {}).get("latest_execution")
        if segment:
            key = segment["segment_id"]
            if key in segment_seen:
                issues.append(
                    {
                        "call": number,
                        "issue": "latest_execution_replayed",
                        "segment": key,
                        "previous_call": segment_seen[key],
                    }
                )
            segment_seen[key] = number
        for image in inp.get("images", []):
            path = episode / image["path"]
            if path not in hashes:
                hashes[path] = (
                    hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                )
            if hashes[path] != image["sha256"]:
                issues.append(
                    {"call": number, "issue": "input_image_hash_mismatch", "path": image["path"]}
                )
            image["path"] = link(path, destination)
        for image in inp.get("logical_images", []):
            path = episode / image["path"]
            if path not in hashes:
                hashes[path] = (
                    hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                )
            if hashes[path] != image["sha256"]:
                issues.append(
                    {"call": number, "issue": "logical_image_hash_mismatch", "path": image["path"]}
                )
            image["path"] = link(path, destination)
        fresh = {
            frame
            for im in inp.get("images", [])
            if im["label"].get("source") == "fresh"
            for frame in im["label"].get("frames", [im["label"].get("frame")])
        }
        for frame in fresh:
            if frame in fresh_seen:
                issues.append(
                    {
                        "call": number,
                        "issue": "fresh_frame_replayed",
                        "frame": frame,
                        "previous_call": fresh_seen[frame],
                    }
                )
            fresh_seen[frame] = number
        provider = episode / directory.name
        stats = read_json(provider / "stats.json", {})
        retry = read_json(provider / "retry.json", {})
        provider_prompt = text_file(provider / "prompt.txt")
        prompt = text_file(directory / "prompt.txt")
        if provider_prompt and provider_prompt != prompt:
            issues.append({"call": number, "issue": "provider_prompt_mismatch"})
        decision = read_json(directory / "decision.json")
        outcome = read_json(directory / "outcome.json", {})
        if outcome.get("error") and decision is None and outcome.get("feedback"):
            # Frozen v2 runtime still holds the preceding feedback if decide() fails.
            # Keep that evidence visible, but never attribute it to the failed call.
            outcome = {
                **outcome,
                "feedback_carried_from_prior_call": outcome["feedback"],
                "feedback": [],
            }
        calls.append(
            {
                "number": number,
                "input": inp,
                "before": read_json(directory / "before.json", {}),
                "after": read_json(directory / "after.json", {}),
                "decision": decision,
                "outcome": outcome,
                "prompt": prompt,
                "response": text_file(provider / "response.json"),
                "stats": stats,
                "events": by_call.get(number, []),
                "transport_retry": retry,
                "recovery": read_json(provider / "recovery.json"),
                "provider_events": read_lines(provider / "events.jsonl"),
                "files": {
                    name: link(path, destination)
                    for name, path in {
                        "input.json": directory / "input.json",
                        "before.json": directory / "before.json",
                        "after.json": directory / "after.json",
                        "decision.json": directory / "decision.json",
                        "prompt.txt": directory / "prompt.txt",
                        "schema.json": directory / "schema.json",
                        "provider events": provider / "events.jsonl",
                        "response.json": provider / "response.json",
                        "stats.json": provider / "stats.json",
                        "stderr.log": provider / "stderr.log",
                        "retry.json": provider / "retry.json",
                        "recovery.json": provider / "recovery.json",
                        **{
                            f"attempt {attempt['attempt']} {name}": provider
                            / attempt["directory"]
                            / name
                            for attempt in retry.get("attempts", [])
                            for name in (
                                "events.jsonl",
                                "stderr.log",
                                "response.json",
                                "stats.json",
                            )
                        },
                    }.items()
                    if path.exists()
                },
            }
        )
    summary = read_json(episode / "summary.json")
    frame_scope = "all_recorded_frames"
    if not frames and calls:
        # Some benchmarks expose RGB only at command boundaries. Reuse the
        # exact audited inputs, never invent intermediate camera observations.
        selected = {}
        for call in calls:
            context = call["input"].get("context", {})
            states = {
                item["frame"]: item for item in context.get("observation", {}).get("frames", [])
            }
            for item in call["input"].get("images", []):
                label = item["label"]
                fid = label.get("frame")
                if label.get("origin") == "training_example" or str(fid).startswith("training:"):
                    # Training images remain visible in the call input, never on
                    # the current episode's physical timeline or robot state.
                    continue
                if fid not in states or not label.get("camera"):
                    continue
                record = selected.setdefault(
                    fid,
                    {
                        "frame": fid,
                        "step": states[fid].get("environment_step"),
                        "cue": label.get("cue", False),
                        "robot_state": states[fid].get("robot_state"),
                        "images": {},
                    },
                )
                record["images"][label["camera"]] = item["path"]
        frames = [selected[key] for key in sorted(selected)]
        frame_scope = "model_observation_boundaries_only"
    provenance = manifest_path(episode)
    manifest = read_json(provenance, {}) if provenance else {}
    validation_only = bool(
        manifest.get("validation_only")
        or manifest.get("config", {}).get("run", {}).get("validation_only")
    )
    if summary and "planner_calls" in summary and summary["planner_calls"] != len(calls):
        issues.append(
            {
                "issue": "call_count_mismatch",
                "summary": summary.get("planner_calls"),
                "audit": len(calls),
            }
        )
    if summary:
        for call in calls:
            if not call["after"]:
                issues.append({"issue": "missing_call_after", "call": call["number"]})
    actions = [event for event in events if event["event"] == "step"]
    if summary and "steps" in summary and summary["steps"] != len(actions):
        issues.append(
            {"issue": "action_count_mismatch", "summary": summary["steps"], "trace": len(actions)}
        )
    rejections = Counter()
    for call in calls:
        for item in call["outcome"].get("feedback", []):
            if item.get("error"):
                rejections[item["error"]] += 1
    retention = [e for e in events if e["event"] == "retention_operation"]
    operations = Counter(e["operation"]["operation"] for e in retention if not e.get("rejection"))
    reads = [e for e in events if e["event"] == "retained_read" and not e.get("error")]
    semantic = [e for e in events if e["event"] in ("memory_operation", "automatic_demo_memory")]
    summary_status = (
        (
            "success"
            if summary.get("success")
            else "error"
            if summary.get("stop_reason") == "error" or summary.get("official_status") == "error"
            else "failure"
        )
        if summary
        else "running"
    )
    if (episode / "not_started.json").exists():
        summary_status = "not_started"
    terminal = next(
        (event for event in reversed(actions) if event.get("terminated") or event.get("truncated")),
        None,
    )
    environment_result = (
        {
            "step": terminal["step"],
            "official_status": terminal.get("scorer_status"),
            "success": terminal.get("scorer_status") == "success",
            "terminated": terminal.get("terminated"),
            "truncated": terminal.get("truncated"),
        }
        if terminal
        else None
    )
    files = {path.name: link(path, destination) for path in episode.glob("*.json")}
    for path in [
        episode / "simulator.log",
        episode / "official/result.json",
        episode / "official/task-progress.json",
        episode / "official/final-proprio.json",
        *sorted((episode / "official/terminal_rgb").glob("*.png")),
    ]:
        if path.exists():
            files[str(path.relative_to(episode))] = link(path, destination)
    episode_id = int(episode.name.removeprefix("episode_"))
    for suffix in ("stdout.log", "stderr.log", "manifest.json"):
        worker = episode.parent.parent / "workers" / f"{episode.parent.name}_{episode_id}.{suffix}"
        if worker.exists():
            files["worker " + suffix] = link(worker, destination)
    return {
        "task": episode.parent.name,
        "episode": episode.name,
        "method": recorded_executor(
            (summary or {}).get("method", manifest.get("config", {}).get("method")),
            (summary or {}).get(
                "memory_enabled",
                manifest.get("config", {}).get("baseline", {}).get("memory_enabled", False),
            ),
        ),
        "benchmark": manifest.get("config", {}).get("benchmark", {}).get("name"),
        "split": manifest.get("config", {}).get("benchmark", {}).get("split"),
        "benchmark_identity": {
            key: value
            for key, value in read_json(episode / "benchmark_metadata.json", {}).items()
            if key
            in (
                "benchmark",
                "task",
                "split",
                "episode",
                "seed",
                "difficulty",
                "variant",
                "official_max_steps",
                "source_revision",
                "episode_metadata_sha256",
            )
        },
        "summary": summary,
        "status": summary_status,
        "validation_only": validation_only,
        "environment_result": environment_result,
        "frame_scope": frame_scope,
        "calls": calls,
        "frames": frames,
        "actions": actions,
        "rejections": dict(rejections),
        "retention_counts": {**dict(operations), "read": len(reads)},
        "diagnostics": {
            "selected_tools": dict(
                Counter(
                    tool["name"]
                    for c in calls
                    if c["decision"]
                    for tool in c["decision"]
                    .get("tool_batch", c["decision"])
                    .get(
                        "tool_calls",
                        [
                            {
                                "name": c["decision"].get(
                                    "tool", c["decision"].get("tool_call", {}).get("name")
                                )
                            }
                        ],
                    )
                )
            ),
            "executed_chunks_by_tool": dict(
                Counter(
                    e["tool"]
                    for e in events
                    if e["event"] == "execution_outcome" and e["executed_steps"] > 0
                )
            ),
            "policy_inference_calls": sum(e["event"] == "policy_inference" for e in events),
            "executed_steps_by_tool": dict(
                Counter(
                    {
                        tool: sum(
                            e["executed_steps"]
                            for e in events
                            if e["event"] == "execution_outcome" and e["tool"] == tool
                        )
                        for tool in {e["tool"] for e in events if e["event"] == "execution_outcome"}
                    }
                )
            ),
            "semantic_operations": len(semantic),
            "semantic_operations_accepted": sum(not e.get("rejection") for e in semantic),
            "retention_operations_rejected": sum(bool(e.get("rejection")) for e in retention),
            "model_verdicts": dict(
                Counter(
                    e["verdict"]["status"] for e in events if e["event"] == "subtask_assessment"
                )
            ),
        },
        "integrity": {
            "issues": issues,
            "checked_images": len(hashes),
            "fresh_frames": len(fresh_seen),
            "audit_frames": len(frames),
            "calls": len(calls),
            "latest_segments": len(segment_seen),
        },
        "files": files,
        "retained_files": {
            path.parent.name: link(path, destination)
            for path in (episode / "retained").glob("selection_*/content.json")
        },
        "trace": link(episode / "trace.jsonl", destination)
        if (episode / "trace.jsonl").exists()
        else None,
    }


def write_json(path, data):
    write_text(path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2))


def write_text(path, content):
    # Stable URLs and unchanged mtimes: repeated generation creates no new copy.
    if path.is_file() and path.read_bytes() == content.encode("utf-8"):
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


ASSET_DIRECTORY = Path(__file__).with_name("viewer")
STYLE = (ASSET_DIRECTORY / "style.css").read_text()
VIEWER = (ASSET_DIRECTORY / "episode.html").read_text()
SCRIPT = "\n".join((ASSET_DIRECTORY / name).read_text() for name in ("resources.js", "viewer.js"))


def scene_label(identity):
    parts = [str(identity["split"])] if identity.get("split") else []
    if identity.get("episode") is not None:
        parts.append(f"episode {identity['episode']}")
    parts.extend(str(identity[k]) for k in ("difficulty", "variant") if identity.get(k))
    return " · ".join(parts)


def scene_panel(data):
    identity = data["benchmark_identity"]
    if not identity:
        return ""
    parts = [scene_label(identity)]
    for key, label in (("seed", "场景 seed"), ("official_max_steps", "官方步数预算")):
        if identity.get(key) is not None:
            parts.append(f"{label}：{identity[key]}")
    if identity.get("source_revision"):
        parts.append(f"源码版本：{identity['source_revision'][:12]}")
    source = data["files"].get("benchmark_metadata.json")
    source_link = (
        f'<a href="{html.escape(source, quote=True)}">场景元数据与完整版本记录</a>'
        if source
        else ""
    )
    return (
        '<div class="panel"><h2>基准与场景来源</h2><p>'
        + html.escape(" · ".join(part for part in parts if part))
        + f'</p><p>{source_link}</p><p class="muted">此处为人工核验信息，不作为模型输入。</p></div>'
    )


def page(
    title,
    body,
    data=None,
    method="task_harness",
    benchmark=None,
    memory_enabled=False,
    split=None,
):
    benchmark_name = {
        "robomme": "RoboMME",
        "robomme_plus": "RoboMME-Plus",
    }.get(benchmark, benchmark)
    if benchmark_name:
        title = f"{benchmark_name} · {title}"
    # Inline data stays inert even when a model emits HTML or a closing script tag.
    encoded = json.dumps(
        data or {}, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    # Bootstrap travels with HTML: a proxy rejecting a separate CSS/JS request must
    # not leave the audit page unstyled or prevent its controls from initializing.
    script = f"<script>{SCRIPT}</script>" if data is not None else ""
    description = {
        "task_harness": "FDS task_harness · Sys2 → 顺序工具 → TCP/IK · 契约与增量记忆",
    }.get(method, method)
    if split:
        description = f"数据划分：{split} · {description}"
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
        f"<header><small>HARNESS / RESEARCH AUDIT</small><h1>{html.escape(title)}</h1>"
        f"<div>{html.escape(description)}</div></header>"
        f'<script type="application/json" id="data">{encoded}</script>{body}{script}</body></html>'
    )


def diagnostic_panel(run, rows):
    """Curated, evidence-linked observations kept separate from benchmark scoring."""
    report = read_json(run / "diagnostics/analysis.json", {})
    if not report:
        return ""
    pages = {(row["task"], row["episode"]): row["page"] for row in rows}
    cards = []
    for finding in report.get("findings", []):
        evidence = []
        for item in finding.get("evidence", []):
            target = pages.get((item.get("task"), item.get("episode", 0)))
            number = item.get("call")
            if target and isinstance(number, int) and number > 0:
                evidence.append(
                    f'<a href="{quote(target)}#call-{number}">'
                    f"{html.escape(item['task'])} · 第 {number} 轮</a>"
                )
        cards.append(
            f'<div class="panel"><h3>{html.escape(finding["title"])}</h3>'
            f"<p>{html.escape(finding['observation'])}</p>"
            f'<p class="muted">{html.escape(finding.get("interpretation", ""))}</p>'
            f"<p>{' · '.join(evidence)}</p></div>"
        )
    probes = []
    for path in sorted((run / "diagnostics").glob("*_sys1_chunks/summary.json")):
        if (path.parent / "proposal_comparison.png").exists():
            probes.append(
                f'<p><a href="{link(path.parent / "proposal_comparison.png", run / "visualization")}">'
                f"{html.escape(path.parent.name)} · 离线动作提案对比</a> · "
                f'<a href="{link(path, run / "visualization")}">原始诊断数据</a></p>'
            )
    return (
        '<div class="panel"><h2>本轮诊断 · 点击证据定位到具体调用</h2>'
        "<p>以下是已记录现象与待验证解释；未在本轮中修改控制逻辑或追加 episode。</p>"
        f'<div class="grid">{"".join(cards)}</div>{"".join(probes)}'
        '<a href="../diagnostics/analysis.json">诊断记录 JSON</a></div>'
    )


def collection_roots(run):
    """Keep paired RSI arms and development/validation results separate."""
    return sorted(
        {
            path
            for prefix in ("", "eval_*/", "candidates/*/eval_*/")
            for arm in ("parent", "candidate")
            for suite in ("development", "validation")
            for path in run.glob(f"{prefix}{arm}/{suite}")
            if path.is_dir()
        }
    )


def build_collection(run, roots, *, videos=True):
    output = run / "visualization"
    (output / "assets").mkdir(parents=True, exist_ok=True)
    write_text(output / "assets/style.css", STYLE)
    results, entries = [], []
    for root in roots:
        result = build(root, videos=videos)
        result["suite"] = root.relative_to(run).as_posix()
        results.append(result)
        entries.append(
            f'<li><a href="{link(Path(result["index"]), output)}">'
            f"{html.escape(result['suite'])}</a> · {result['episodes']} episodes</li>"
        )
    write_text(
        output / "index.html",
        page(
            "RSI · 实验审计",
            '<main><div class="panel"><h2>执行器版本与数据划分</h2>'
            "<p>父版本、候选版本及开发／验证集分别查看，原始审计与图片直接复用。</p>"
            "<ul>" + "".join(entries) + "</ul></div></main>",
        ),
    )
    write_json(
        output / "results.json", {"format": "standard_browser_collection_v1", "suites": results}
    )
    return {
        "index": str(output / "index.html"),
        "suites": len(results),
        **{
            key: sum(r[key] for r in results)
            for key in ("episodes", "completed", "integrity_issues")
        },
    }


def build(run, *, videos=True):
    run = Path(run).resolve()
    if not run.is_dir():
        raise ValueError(f"Run directory does not exist: {run}")
    roots = collection_roots(run)
    if roots:
        return build_collection(run, roots, videos=videos)
    output = run / "visualization"
    output.mkdir(exist_ok=True)
    renderer = RENDERER_SOURCE + STYLE.encode() + VIEWER.encode() + SCRIPT.encode()
    previous = read_json(output / "results.json", {})
    fingerprint = hashlib.sha256(renderer)
    (output / "assets").mkdir(exist_ok=True)
    write_text(output / "assets/style.css", STYLE)
    write_text(output / "assets/viewer.js", SCRIPT)
    batch = read_json(run / "batch.json", {})
    user_stop = read_json(run / "job/user_stop.json", {})
    interrupted = {
        (item["task"], item["episode"]) for item in user_stop.get("interrupted_episodes", [])
    }
    unstarted = {
        (item["task"], item["episode"]) for item in user_stop.get("unstarted_episodes", [])
    }
    experiment = run / "experiment" if (run / "experiment").is_dir() else run
    source_manifest = experiment / "manifest.json"
    run_manifest = read_json(source_manifest, {})
    single_path = run if run.name.startswith("episode_") and run.name[8:].isdigit() else None
    discovered = (
        {run.parent.name: [int(run.name[8:])]}
        if single_path
        else {
            p.name: sorted(
                int(ep.name[8:])
                for ep in p.glob("episode_*")
                if ep.is_dir() and ep.name[8:].isdigit()
            )
            for p in sorted(experiment.iterdir())
            if p.is_dir()
        }
    )
    discovered = {task: ids for task, ids in discovered.items() if ids}
    if not run_manifest and discovered:
        task = next(iter(discovered))
        source_manifest = manifest_path(
            single_path or experiment / task / f"episode_{discovered[task][0]:04d}"
        )
        run_manifest = read_json(source_manifest, {}) if source_manifest else {}
    method = recorded_executor(
        run_manifest.get("config", {}).get("method", "task_harness"),
        run_manifest.get("config", {}).get("task_harness", {}).get("memory_enabled", True),
    )
    benchmark = run_manifest.get("config", {}).get("benchmark", {}).get("name")
    memory_enabled = method == "task_harness"
    tasks = batch.get("tasks") or list(discovered)
    declared_ids = batch.get("episodes")
    if declared_ids is None and (experiment / "manifest.json").is_file() and not single_path:
        declared_ids = run_manifest.get("config", {}).get("run", {}).get("episodes")
    cases = {
        task: declared_ids if declared_ids is not None else discovered.get(task, [0])
        for task in tasks
    }
    intervention = read_json(run / "intervention.json", {})
    single_episode = sum(len(ids) for ids in cases.values()) == 1
    rows, integrity, pages, diagnostics = [], {}, [], {}
    for task in tasks:
        for episode_id in cases[task]:
            episode = single_path or experiment / task / f"episode_{episode_id:04d}"
            data = episode_data(episode, output)
            data["replay_video"] = (
                build_replay(episode, output, data["frames"])
                if videos
                else {"status": "disabled", "reason": "本次生成未启用视频。"}
            )
            if data["replay_video"].get("path"):
                data["replay_video"]["url"] = (
                    quote(data["replay_video"]["path"], safe="/")
                    + "?v="
                    + data["replay_video"]["fingerprint"][:16]
                )
            data["task"] = task
            data["method"] = data["method"] or method
            data["benchmark"] = data["benchmark"] or benchmark
            if not episode.exists():
                data["status"] = "pending"
            if (task, episode_id) in interrupted:
                data["status"] = "interrupted"
            elif (task, episode_id) in unstarted:
                data["status"] = "not_run"
            elif (
                intervention.get("type") == "diagnostic_early_stop"
                and intervention.get("full_episode_result") is False
                and (
                    (intervention.get("task"), intervention.get("episode")) == (task, episode_id)
                    or single_episode
                    and "task" not in intervention
                    and "episode" not in intervention
                )
            ):
                data["status"] = "diagnostic_early_stop"
                data["intervention"] = intervention
                data["files"]["intervention.json"] = link(run / "intervention.json", output)
            elif (
                intervention.get("type") == "post_terminal_review_interruption"
                and (intervention.get("task"), intervention.get("episode")) == (task, episode_id)
                and data["environment_result"]
                and data["calls"]
                and data["calls"][-1]["input"].get("context", {}).get("review_only")
            ):
                # Annotation alone cannot manufacture a terminal environment result.
                data["status"] = "post_terminal_review_interruption"
                data["intervention"] = intervention
                data["files"]["intervention.json"] = link(run / "intervention.json", output)
            filename = f"{task}_{episode_id:04d}.html"
            # Raw text is fetched from the original audit only when its tab is opened.
            for call in data["calls"]:
                call.pop("prompt")
                call.pop("response")
            rendered = page(
                f"{task} · episode {episode_id}",
                VIEWER.replace("<!-- benchmark-identity -->", scene_panel(data)),
                data,
                method=data["method"],
                benchmark=data["benchmark"],
                memory_enabled=memory_enabled,
                split=data["split"],
            )
            fingerprint.update(rendered.encode())
            write_text(output / filename, rendered)
            pages.append(filename)
            s = data["summary"] or {}
            last = data["calls"][-1] if data["calls"] else {}
            rows.append(
                {
                    "task": task,
                    "episode": episode_id,
                    "status": data["status"],
                    "method": data["method"],
                    "benchmark": data["benchmark"],
                    "split": data["split"],
                    "benchmark_identity": data["benchmark_identity"],
                    "benchmark_metadata_file": data["files"].get("benchmark_metadata.json"),
                    "success": None
                    if data["status"] in ("interrupted", "not_run", "diagnostic_early_stop")
                    else s.get("success"),
                    "intervention": data.get("intervention"),
                    "environment_success": (
                        data["environment_result"]["success"]
                        if data["environment_result"]
                        else None
                    ),
                    "stop_reason": s.get("stop_reason"),
                    "steps": s.get(
                        "steps",
                        last.get("after", {}).get("step", last.get("before", {}).get("step")),
                    ),
                    "calls": len(data["calls"]),
                    "retention": data["retention_counts"],
                    "retry_audited": any(c["transport_retry"] for c in data["calls"]),
                    "request_attempts": sum(
                        len(c["transport_retry"]["attempts"])
                        if c["transport_retry"]
                        else c["stats"].get("request_attempts", 1)
                        for c in data["calls"]
                    ),
                    "transport_retries": sum(
                        max(0, len(c["transport_retry"]["attempts"]) - 1)
                        if c["transport_retry"]
                        else c["stats"].get("transport_retries", 0)
                        for c in data["calls"]
                    ),
                    "vla_selected": data["diagnostics"]["selected_tools"].get("vla", 0),
                    "vla_executed": data["diagnostics"]["executed_chunks_by_tool"].get("vla", 0),
                    "motion_selected": data["diagnostics"]["selected_tools"].get("move_ee", 0),
                    "motion_executed": data["diagnostics"]["executed_chunks_by_tool"].get(
                        "move_ee", 0
                    ),
                    "native_selected": sum(
                        data["diagnostics"]["selected_tools"].get(k, 0)
                        for k in ("arm", "base", "wait")
                    ),
                    "native_executed": sum(
                        data["diagnostics"]["executed_chunks_by_tool"].get(k, 0)
                        for k in ("arm", "base", "wait")
                    ),
                    "benchmark_metrics": s.get("benchmark_metrics", {}),
                    "rejections": data["rejections"],
                    "wall_seconds": s.get("timing", {}).get("wall_seconds"),
                    "page": filename,
                    "error": s.get("error"),
                }
            )
            integrity[f"{task}/{episode_id}"] = data["integrity"]
            diagnostics[f"{task}/{episode_id}"] = {
                **data["diagnostics"],
                "rejections": data["rejections"],
                "retention": data["retention_counts"],
            }
    metrics = read_json(experiment / "metrics.json", {})
    counts = Counter(row["status"] for row in rows)
    complete = sum(counts[k] for k in ("success", "failure", "error"))
    table = []
    names = {
        "success": "官方成功",
        "failure": "未成功",
        "error": "运行错误",
        "running": "执行中",
        "not_started": "未启动",
        "pending": "待运行",
        "interrupted": "用户中断",
        "not_run": "未运行（用户停止）",
        "diagnostic_early_stop": "诊断性截断",
        "post_terminal_review_interruption": "环境已结束，复核中断",
    }
    for row in rows:
        reason = row["error"] or row["stop_reason"] or ""
        # The full error and file paths remain in results.json and the episode audit.
        reason = reason.split("; inspect ", 1)[0]
        if row["status"] in ("interrupted", "not_run"):
            reason = "用户要求停止；不纳入完整 episode 成绩。" + (
                "已有过程完整保留。" if row["status"] == "interrupted" else ""
            )
        elif row["status"] == "diagnostic_early_stop":
            reason = "诊断性截断；不纳入完整 episode 成绩。" + row["intervention"].get("reason", "")
        elif row["status"] == "post_terminal_review_interruption":
            reason = (
                "环境官方成功："
                + str(row["environment_success"])
                + "；最终复核中断，原始整条运行仍为错误。"
                + row["intervention"].get("reason", "")
            )
        rejections = ", ".join(
            f"{k} ×{v}" for k, v in sorted(row["rejections"].items(), key=lambda x: -x[1])[:3]
        )
        r = row["retention"]
        task_cell = (
            f'<a href="{quote(row["page"])}"><strong>{html.escape(row["task"])}</strong> ↗</a>'
        )
        if row["benchmark_identity"]:
            task_cell += f"<br><small>{html.escape(scene_label(row['benchmark_identity']))}"
            if row["benchmark_metadata_file"]:
                task_cell += (
                    f' · <a href="{html.escape(row["benchmark_metadata_file"], quote=True)}">'
                    "场景记录</a>"
                )
            task_cell += "</small>"
        cells = [
            task_cell,
            f'<span class="badge {row["status"]}">{names[row["status"]]}</span>',
            html.escape(str(row["steps"] if row["steps"] is not None else "—")),
            (
                str(row["calls"])
                + (
                    f"<br><small>请求 {row['request_attempts']} / 重试 {row['transport_retries']}</small>"
                    if row["retry_audited"]
                    else ""
                )
            ),
            (
                f"{row['motion_executed']} / {row['motion_selected']}"
            ),
            (
                f"{r.get('save', 0)} / {r.get('read', 0)} / {r.get('delete', 0)}"
                if memory_enabled
                else "不启用"
            ),
            f"{row['wall_seconds'] / 60:.1f} min" if row["wall_seconds"] is not None else "—",
            html.escape(reason)
            + ("<br><small>" + html.escape(rejections) + "</small>" if rejections else ""),
        ]
        table.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    stamp = "__GENERATED_UTC__"
    issues = sum(len(item["issues"]) for item in integrity.values())
    stop_note = (
        "<p><strong>本批已按用户要求停止。中途停止与未运行单独列出，"
        "不计为完整失败任务；原始工作进程日志保留实际终止异常。</strong></p>"
        if user_stop
        else ""
    )
    if batch.get("retry_of"):
        stop_note += (
            f"<p>本批仅补跑原先通信错误的 {len(tasks)} 个任务，重新 reset 相同 episode；"
            "运行中允许有界请求重试，环境在重试期间不动。"
            f'<a href="{link(Path(batch["retry_of"]) / "visualization/index.html", output)}">'
            "查看原始实验（结果保留）</a>。补跑成绩单独统计。</p>"
        )
    models = read_json(run / "provenance/models.json", {})
    if not models and run_manifest.get("config"):
        configured = run_manifest["config"]
        policy, planner = configured.get("policy", {}), configured.get("planner", {})
        models = {
            "summary": [
                {
                    "role": "Sys1 / VLA",
                    "identity": policy.get("model_id", "未启用 VLA"),
                    "settings": policy.get("protocol", "基础动作工具"),
                },
                {
                    "role": "Sys2",
                    "identity": planner.get("model", planner.get("backend", "未知")),
                    "settings": planner.get("backend", "未知"),
                },
            ]
        }
    model_panel = ""
    if models:
        model_source = run / "provenance/models.json"
        if not model_source.exists():
            model_source = source_manifest
        model_panel = (
            '<div class="panel"><h2>本轮模型配置</h2><table><thead><tr>'
            "<th>职责</th><th>模型 / 版本</th><th>运行设置</th></tr></thead><tbody>"
            + "".join(
                "<tr><td>"
                + "</td><td>".join(
                    html.escape(str(row[k])) for k in ("role", "identity", "settings")
                )
                + "</td></tr>"
                for row in models["summary"]
            )
            + f'</tbody></table><p><a href="{link(model_source, output)}">模型与运行配置原始记录</a></p></div>'
        )
    artifacts = {
        "冻结配置": run / "config.toml",
        "源码校验值": run / "source_sha256.json",
        "环境与有效配置 Manifest": source_manifest or experiment / "manifest.json",
        "已结束结果 CSV": experiment / "episodes.csv",
        "汇总 JSON": experiment / "metrics.json",
        "实验日志": experiment / "experiment.log",
        "GPU 记录": run / "job/gpu_usage.csv",
        "接口预检": run / "preflight/sys2_status.json",
    }
    artifact_links = " · ".join(
        f'<a href="{link(path, output)}">{html.escape(label)}</a>'
        for label, path in artifacts.items()
        if path.exists()
    )
    action_heading = "IK 动作块执行 / 选择"
    analysis_guide = (
        "<p>2. 查看该轮 Sys2 输入：目标是否入镜、是否遗漏关键变化、读取了哪些历史片段。</p>"
        "<p>3. 查看输出与 Harness 反馈：子目标、定位、验证是否合理，是否发生接口拒绝。</p>"
        "<p>4. 在记忆变化页核对新增、更新、删除和证据有效期，区分未保存、读错与模型误判。</p>"
        if memory_enabled
        else "<p>2. 查看该轮 Sys2 输入：公开总目标、本轮新画面、机器人状态与上一轮执行反馈。</p>"
        "<p>3. 对照模型输出、接口拒绝和实际动作；IK 收敛、运动跟踪与官方成功分别判断。</p>"
        "<p>4. 此方法无外部记忆、无历史回看；记忆页用于确认未启用，历史回放仅供人工分析。</p>"
    )
    body = f"""<main><div class="note">点击任务进入逐轮分析；模型、数据集与采样预算以该次运行 Manifest 为准。
    原始录像仅供人类审计；页面中“本轮真正输入的画面”才是 Sys2 实际所见。接口验证与 mock 结果不是模型任务成绩。{stop_note}</div>
    <div class="stats"><div class="stat">完整结束 / 计划<b>{complete} / {len(rows)}</b></div>
    <div class="stat">官方成功<b>{counts["success"]}</b></div><div class="stat">执行后未成功<b>{counts["failure"]}</b></div>
    <div class="stat">运行错误 / 未启动<b>{counts["error"]} / {counts["not_started"]}</b></div>
    <div class="stat">用户中断 / 未运行<b>{counts["interrupted"]} / {counts["not_run"]}</b></div>
    <div class="stat">诊断性截断（非完整成绩）<b>{counts["diagnostic_early_stop"]}</b></div>
    <div class="stat">输入证据一致性问题<b>{issues}</b></div></div>
    {model_panel}
    <div class="panel"><h2>{len(tasks)} 个任务 · 结果与问题定位</h2><div class="scroll"><table class="results-table"><thead><tr><th>任务</th><th>结果</th><th>环境步</th><th>Sys2 轮数</th><th>{action_heading}</th><th>原始记忆 存 / 读 / 删</th><th>耗时</th><th>停止原因 / 常见拒绝</th></tr></thead><tbody>{"".join(table)}</tbody></table></div></div>
    {diagnostic_panel(run, rows)}
    <div class="grid"><div class="panel"><h2>如何定位问题</h2><p>1. 先用执行回放确认真实行为，定位异常轮次。</p>{analysis_guide}</div>
    <div class="panel"><h2>可复核产物</h2><p>{artifact_links}</p><p><a href="results.csv">状态 CSV</a> · <a href="results.json">可视化索引</a> · <a href="integrity.json">输入证据检查</a></p><p class="muted">生成时间 {stamp}。离线页面为生成时快照；运行中的结果需重新生成后刷新。</p></div></div></main>"""
    # Both legacy batch layouts and direct CLI runs expose their actual artifacts.
    body = body.replace("../experiment/", link(experiment, output).rstrip("/") + "/")
    rendered = page(
        f"{len(tasks)} 任务 · 逐轮实验分析",
        body,
        method=method,
        benchmark=benchmark,
        memory_enabled=memory_enabled,
        split=run_manifest.get("config", {}).get("benchmark", {}).get("split"),
    )
    fingerprint.update(rendered.encode())
    fingerprint.update(json.dumps([rows, metrics, integrity, diagnostics], sort_keys=True).encode())
    content_hash = fingerprint.hexdigest()
    stamp = (
        previous.get("generated_utc") if previous.get("content_sha256") == content_hash else None
    )
    stamp = stamp or datetime.now(UTC).isoformat(timespec="seconds")
    write_text(output / "index.html", rendered.replace("__GENERATED_UTC__", stamp))
    write_json(
        output / "results.json",
        {
            "generated_utc": stamp,
            "metrics": metrics,
            "rows": rows,
            "counts": dict(counts),
            "renderer_sha256": hashlib.sha256(renderer).hexdigest(),
            "content_sha256": content_hash,
            "format": "standard_browser_v1",
        },
    )
    stream = io.StringIO(newline="")
    columns = (
        "task",
        "episode",
        "status",
        "success",
        "steps",
        "calls",
        "vla_selected",
        "vla_executed",
        "motion_selected",
        "motion_executed",
        "stop_reason",
        "error",
        "wall_seconds",
        "page",
    )
    writer = csv.DictWriter(stream, columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_text(output / "results.csv", "\ufeff" + stream.getvalue())
    for row in previous.get("rows", []):
        obsolete = row.get("page", "")
        if (
            obsolete
            and obsolete not in pages
            and Path(obsolete).name == obsolete
            and obsolete.endswith(".html")
            and obsolete != "index.html"
        ):
            (output / obsolete).unlink(missing_ok=True)
    write_json(output / "integrity.json", integrity)
    write_json(output / "diagnostics.json", diagnostics)
    return {
        "index": str(output / "index.html"),
        "episodes": len(rows),
        "completed": complete,
        "integrity_issues": issues,
        "pages": pages,
    }


def run_finished(run):
    if (run / "job/result.json").is_file() or (run / "report.json").is_file():
        return True
    if run.name.startswith("episode_") and (run / "summary.json").is_file():
        return True
    return any(
        read_json(root / "metrics.json", {}).get("status") in ("completed", "interrupted", "error")
        for root in (run, run / "experiment")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 encoding for a quick text/image-only refresh",
    )
    parser.add_argument(
        "--watch", action="store_true", help="Refresh every 30 s until the supervised run finishes"
    )
    args = parser.parse_args()
    while True:
        if args.watch and (
            Path(__file__).read_bytes() != RENDERER_SOURCE
            or (ASSET_DIRECTORY / "style.css").read_text() != STYLE
            or (ASSET_DIRECTORY / "episode.html").read_text() != VIEWER
            or "\n".join(
                (ASSET_DIRECTORY / name).read_text() for name in ("resources.js", "viewer.js")
            )
            != SCRIPT
        ):
            os.execv(sys.executable, [sys.executable, "-m", "fds.visualize", *sys.argv[1:]])
        print(json.dumps(build(args.run, videos=not args.no_video), ensure_ascii=False), flush=True)
        if not args.watch or run_finished(args.run):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
