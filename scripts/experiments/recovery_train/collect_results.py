"""Collect complete or failed train scenes; never fabricate missing measurements."""
import collections
import csv
import datetime
import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
collection_lock = (OUT / "collector.lock").open("a")
fcntl.flock(collection_lock, fcntl.LOCK_EX)
warnings = []


def read(path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        warnings.append({"path": str(path), "error": str(error)})
        return {} if default is None else default


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


campaign = read(OUT / "campaign.json")
keys = set()
for mode in ("f0", "f1"):
    for directory in (OUT / f"train_{mode}").glob("*_ep*"):
        task, episode = directory.name.rsplit("_ep", 1)
        keys.add((mode, task, int(episode)))
for path in (OUT / "logs").glob("*.failure.json"):
    failure = read(path)
    if all(k in failure for k in ("condition", "task", "episode")):
        keys.add((failure["condition"], failure["task"], failure["episode"]))

rows = []
for mode, task, episode in sorted(keys):
    directory = OUT / f"train_{mode}" / f"{task}_ep{episode}"
    ep = directory / task / f"episode_{episode:04d}"
    failure = read(OUT / "logs" / f"{mode}_{task}_ep{episode}.failure.json")
    runner = read(OUT / "logs" / f"{mode}_{task}_ep{episode}.status.json")
    summary = read(ep / "summary.json")
    metrics = read(directory / "metrics.json")
    terminal = bool(failure) or runner.get("status") in ("completed", "failed") or (bool(summary) and metrics.get("status") == "completed")
    if not terminal:
        continue
    metadata = read(ep / "benchmark_metadata.json")
    official = next(r for r in campaign["train_episode_metadata"][task] if r["episode"] == episode)
    calls = sorted(q for q in ep.glob("call_*") if q.is_dir())
    attempts = [q for call in calls for q in sorted(call.glob("attempt_*")) if q.is_dir()]
    contracts = [r["fds_contract"] for q in attempts if (r := read(q / "response.json")).get("fds_contract") is not None]
    accepted = list((ep / "audit").glob("call_*/fds_contract.json"))
    verdicts = collections.Counter(read(q).get("status", "unknown") for q in (ep / "audit").glob("call_*/fds_verdict.json"))
    rejected = []
    transport_retries = 0
    for call in calls:
        retry = read(call / "retry.json")
        transport_retries += max(0, len(retry.get("attempts", [])) - 1)
        for attempt in retry.get("attempts", []):
            error = attempt.get("error", "").lower()
            if attempt.get("status") in ("fatal_error", "decision_rejected") and any(word in error for word in ("contract", "assumption", "evidence requirement", "candidate")):
                rejected.append({"call": call.name, **attempt})
    for outcome_path in (ep / "audit").glob("call_*/outcome.json"):
        call_name = outcome_path.parent.name
        outcome_error = read(outcome_path).get("error") or {}
        message = outcome_error.get("message", "")
        if any(word in message.lower() for word in ("contract", "assumption", "evidence requirement", "candidate")) and not any(x["call"] == call_name for x in rejected):
            if read(ep / call_name / "response.json").get("fds_contract") is not None:
                rejected.append({"call": call_name, "status": "runtime_validation_error", "error_type": outcome_error.get("type"), "error": message})
    settings = collections.Counter()
    known_usage = {"input_tokens": 0, "output_tokens": 0}
    missing_usage = 0
    for attempt in attempts:
        stats = read(attempt / "stats.json")
        settings[tuple(stats.get(k) for k in ("model", "effort", "sandbox_mode", "approval_policy"))] += 1
        usage = stats.get("usage", {})
        if "input_tokens" not in usage or "output_tokens" not in usage:
            missing_usage += 1
        for key in known_usage:
            known_usage[key] += usage.get(key, 0) or 0
    expected_settings = ("gpt-6-astra", "medium", "workspace-write", "never")
    verified = settings[expected_settings]
    settings_issues = [{"settings": list(k), "attempts": v} for k, v in settings.items() if k != expected_settings]
    feedback = [receipt for q in (ep / "audit").glob("call_*/outcome.json") for receipt in read(q).get("feedback", [])]
    artifact_complete = bool(summary) and metrics.get("status") == "completed"
    infrastructure_error = runner.get("infrastructure_error") or failure.get("infrastructure_error")
    error = summary.get("error") or failure.get("error") or (infrastructure_error or {}).get("error")
    failed = bool(error) or not artifact_complete or bool(metrics.get("errors", 0)) or runner.get("status") == "failed"
    final_steps = summary.get("steps")
    if final_steps is None and artifact_complete:
        final_steps = metrics.get("environment_steps")
    observed_after = [read(q) for q in (ep / "audit").glob("call_*/after.json")]
    step_evidence = [d.get("step", d.get("steps")) for d in observed_after if isinstance(d.get("step", d.get("steps")), int)]
    reported_input = metrics.get("prompt_tokens")
    reported_output = metrics.get("completion_tokens")
    if not attempts and artifact_complete:
        known_usage = {"input_tokens": reported_input or 0, "output_tokens": reported_output or 0}
    tokens_disagree = bool(artifact_complete and (reported_input != known_usage["input_tokens"] or reported_output != known_usage["output_tokens"]))
    explanation = error or metadata.get("result", {}).get("failure_reason")
    if not explanation and not summary.get("success", False):
        if summary.get("official_status") == "timeout":
            explanation = f"Official environment timeout after {final_steps} steps; remaining_steps={metadata.get('result', {}).get('remaining_steps', 'unknown')}"
        else:
            explanation = f"No success recorded; official_status={summary.get('official_status', 'unknown')}, stop_reason={summary.get('stop_reason', 'missing_result')}"
    rows.append({
        "condition": mode, "task": task, "episode": episode,
        "decision_recovery": summary.get("decision_recovery", {}),
        "decision_rejections": summary.get("decision_recovery", {}).get("rejected_decisions", 0),
        "correction_calls": summary.get("decision_recovery", {}).get("correction_calls", 0),
        "recovered_decisions": summary.get("decision_recovery", {}).get("recovered_decisions", 0),
        "split": metadata.get("split", "train"), "difficulty": metadata.get("difficulty", official["difficulty"]), "official_seed": metadata.get("seed", official["seed"]),
        "success": bool(summary.get("success", False)), "official_status": summary.get("official_status", "unknown"), "metadata_official_status": metadata.get("result", {}).get("official_status"),
        "stop_reason": summary.get("stop_reason", failure.get("stop_reason", "missing_result")), "completed_without_error": not failed, "artifact_complete": artifact_complete,
        "request_attempts": len(attempts), "planner_calls": summary.get("planner_calls", metrics.get("planner_calls") if artifact_complete else len(calls)),
        "prompt_tokens": known_usage["input_tokens"], "completion_tokens": known_usage["output_tokens"], "reported_prompt_tokens": reported_input, "reported_completion_tokens": reported_output, "token_accounting_disagreement": tokens_disagree,
        "environment_steps": final_steps, "observed_environment_steps_lower_bound": max(step_evidence) if step_evidence else None,
        "contracts_proposed": len(contracts), "contracts_accepted": len(accepted), "contract_validity": len(accepted) / len(contracts) if contracts else None,
        "contracts_rejected": len(rejected), "contract_rejection_details": rejected, "contradicted": verdicts["contradicted"], "uncertain": verdicts["uncertain"],
        "tool_rejected_receipts": sum(r.get("status") == "rejected" for r in feedback),
        "public_error_receipts": sum(bool(r.get("environment_feedback", {}).get("action_error") or r.get("environment_feedback", {}).get("trial_reset")) for r in feedback),
        "transport_retries": transport_retries, "reported_failed_attempt_usage_unknown": metrics.get("failed_attempt_usage_unknown", 0), "attempts_without_recorded_token_usage": missing_usage,
        "settings_verified_for_attempts": verified, "settings_issues": settings_issues, "wall_seconds": metrics.get("wall_seconds"),
        "error_type": summary.get("error_type") or failure.get("error_type") or (infrastructure_error or {}).get("error_type"), "error": error, "infrastructure_error": infrastructure_error,
        "benchmark_failure_reason": metadata.get("result", {}).get("failure_reason"),
        "failure_explanation": explanation,
        "output_path": str(directory), "physical_output_path": str(directory.resolve()), "summary_path": str(ep / "summary.json"), "metrics_path": str(directory / "metrics.json"), "audit_path": str(ep / "audit"),
    })

totals = {}
sum_fields = ("request_attempts", "decision_rejections", "correction_calls", "recovered_decisions", "planner_calls", "prompt_tokens", "completion_tokens", "environment_steps", "contracts_proposed", "contracts_accepted", "contracts_rejected", "contradicted", "uncertain", "transport_retries", "reported_failed_attempt_usage_unknown", "attempts_without_recorded_token_usage")
for mode in ("f0", "f1", "all"):
    selected = [r for r in rows if mode == "all" or r["condition"] == mode]
    totals[mode] = {k: sum(r[k] or 0 for r in selected) for k in sum_fields}
    totals[mode].update(finished=len(selected), completed_without_error=sum(r["completed_without_error"] for r in selected), successes=sum(r["success"] for r in selected), task_failures=sum(not r["success"] and r["completed_without_error"] for r in selected), process_errors=sum(not r["completed_without_error"] for r in selected), missing_artifacts=sum(not r["artifact_complete"] for r in selected), unknown_environment_steps=sum(r["environment_steps"] is None for r in selected))
source_unchanged = all(hashlib.sha256((ROOT / "src/fds" / p).read_bytes()).hexdigest() == digest for p, digest in campaign["fds_source_sha256"].items())
result = {"generated_utc": datetime.datetime.now(datetime.UTC).isoformat(), "source_unchanged": source_unchanged, "rows": rows, "totals": totals, "warnings": warnings, "cost_formula": "prompt_tokens / 1e6 * input_price + completion_tokens / 1e6 * output_price", "counting_notes": ["Each attempt usage counted once; canonical copies excluded. Missing usage is reported explicitly, making known token totals lower bounds when relevant.", "Rejected contracts counted from fatal validation errors and preserved raw responses; absent verdicts do not imply zero rejection.", "Contract validity = accepted / proposed; F0 without contracts is null (N/A).", "Official metrics and independent usage sums retained separately; missing steps remain null.", "Scene errors do not halt the queue under the user's updated policy."]}
atomic_json(OUT / "results.json", result)
fields = [k for k in rows[0] if k not in ("contract_rejection_details", "settings_issues", "infrastructure_error")] if rows else []
with (OUT / "results.csv.tmp").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
(OUT / "results.csv.tmp").replace(OUT / "results.csv")
print(json.dumps({"completed_rows": len(rows), "totals": totals}, indent=2))
