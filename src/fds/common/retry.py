"""Opt-in transport retries while the caller holds the environment unchanged."""

import hashlib
import json
import random
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path


class TransportError(RuntimeError):
    """A failed inference transport, before any decision is returned for execution."""


def classify_request_failure(error):
    """Stable controller-facing classification; unknown failures are never retried."""
    result = {"failure_category": "unknown", "retryable": False, "global_blocker": False}
    if isinstance(error, BaseException) and not isinstance(error, Exception):
        return dict(result, failure_category="interrupted")
    value = str(error).lower()
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        value += " " + json.dumps(body).lower()
    status = getattr(error, "status_code", None)
    categories = (
        ("quota", ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "exceeded your current quota", "usage limit", "quota exceeded")),
        ("authentication", ("unauthorized", "authentication", "invalid_api_key", "incorrect api key", "requires an existing chatgpt login")),
        ("authorization", ("forbidden", "permission_denied")),
        ("context_limit", ("context_length_exceeded", "context window", "max_output_tokens")),
        ("schema", ("invalid_json_schema", "invalid schema")),
        ("certificate", ("certificate", "ssl verification")),
    )
    for category, markers in categories:
        if any(marker in value for marker in markers):
            return dict(result, failure_category=category, global_blocker=category in ("quota", "authentication", "authorization"))
    if status in (401, 403):
        return dict(result, failure_category="authentication" if status == 401 else "authorization", global_blocker=True)
    # These flags prevent replay even when a completed response mentions transport errors.
    if getattr(error, "request_replay_forbidden", False):
        return dict(result, failure_category="replay_forbidden")
    if status == 429 or any(x in value for x in ("429 too many requests", "rate_limit_exceeded", "rate limit reached")):
        return dict(result, failure_category="rate_limit", retryable=True)
    if status in (408, 409) or isinstance(status, int) and 500 <= status <= 599:
        return dict(result, failure_category="service_unavailable", retryable=True)
    if isinstance(status, int) and 400 <= status <= 499:
        return dict(result, failure_category="invalid_request")
    if isinstance(error, (subprocess.TimeoutExpired, TimeoutError)):
        return dict(result, failure_category="timeout", retryable=True)
    if isinstance(error, (TransportError, ConnectionError)):
        return dict(result, failure_category="transport", retryable=True)
    try:
        import openai
        if isinstance(error, openai.APIConnectionError):
            return dict(result, failure_category="transport", retryable=True)
    except ImportError:
        pass
    if isinstance(error, str) and any(x in value for x in (
        "connection failed: error sending request", "connection reset by peer",
        "stream disconnected before completion", "request timed out", "operation timed out",
        "error sending request for url", "connection refused", "temporary failure in name resolution",
        "429 too many requests", "500 internal server error", "502 bad gateway",
        "503 service unavailable", "504 gateway timeout",
        "currently experiencing high demand, which may cause temporary errors",
        "selected model is at capacity",
    )):
        return dict(result, failure_category="transport", retryable=True)
    return result


def retryable_transport_error(error):
    return classify_request_failure(error)["retryable"]


def transient_codex_failure(message, events):
    """Conservative allowlist: unknown errors, completed turns and tools never retry."""
    for event in events:
        if event.get("type") == "turn.completed":
            return False
        item = event.get("item")
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type and item_type not in ("reasoning", "agent_message", "error"):
            return False
    return classify_request_failure(message)["retryable"]


def retry_after_seconds(error):
    """Read standard Retry-After seconds/date without accepting unbounded delays."""
    import math
    from email.utils import parsedate_to_datetime

    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, seconds) if math.isfinite(seconds) else None


def retry_decision(
    decide, context, images, directory, *, retries, delay, max_delay, model, effort, pause_seconds=0,
    auto_resume_after_seconds=0, sandbox_mode=None, approval_policy=None
):
    """Return one accepted decision; preserve every failed request in its own folder."""
    from fds.common.audit import AuditJournal
    from fds.decision import decision_schema
    from fds.planners import make_prompt
    from fds.task_harness.protocol import DecisionValidationError

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    context, images = deepcopy(context), tuple(Path(p).resolve() for p in images)
    prompt = make_prompt(context)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    AuditJournal.write(directory / "schema.json", decision_schema(context))
    image_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in images]
    journal = {
        "max_retries": retries,
        "accepted_attempt": None,
        "attempts": [],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "image_sha256": image_hashes,
        "environment_advanced_during_retry": False,
    }
    from fds.common.recovery import RecoveryConfig
    RecoveryConfig(retries, delay, max_delay, pause_seconds, auto_resume_after_seconds)
    wall_started = time.monotonic()
    index, budget, resumed, window_index = 0, retries, False, 0
    while True:
        if [hashlib.sha256(p.read_bytes()).hexdigest() for p in images] != image_hashes:
            raise RuntimeError("Inference images changed during transport retry")
        attempt = directory / f"attempt_{index + 1:04d}"
        record = {"attempt": index + 1, "directory": attempt.name, "status": "running"}
        journal["attempts"].append(record)
        AuditJournal.write(directory / "retry.json", journal)
        started = time.monotonic()
        retry_wait, pending_error = None, None
        try:
            decision = decide(deepcopy(context), images, attempt)
        except BaseException as exc:
            classification = classify_request_failure(exc)
            transient = classification["retryable"]
            record.update(classification)
            record.update(
                status="transport_error" if transient else (
                    "decision_rejected" if isinstance(exc, DecisionValidationError) else "fatal_error"
                ),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if getattr(exc, "status_code", None) is not None:
                record["http_status"] = exc.status_code
            print(
                json.dumps(
                    {
                        "event": "model_request_error",
                        "directory": str(directory),
                        "attempt": index + 1,
                        "retryable": transient,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not transient:
                raise
            if index < budget:
                retry_wait = min(
                    max_delay, delay * 2**window_index * random.SystemRandom().uniform(0.8, 1.2)
                )
                server_wait = retry_after_seconds(exc)
                if server_wait is not None:
                    record["retry_after_seconds"] = server_wait
                    if server_wait > max_delay:
                        record["retry_after_exceeds_policy"] = True
                        raise
                    retry_wait = max(retry_wait, server_wait)
                record["retry_wait_seconds"] = retry_wait
            elif (pause_seconds or auto_resume_after_seconds) and not resumed:
                server_wait = retry_after_seconds(exc)
                if server_wait is not None:
                    record["retry_after_seconds"] = server_wait
                    # Operator resume can arrive immediately; never retry too early.
                    if not auto_resume_after_seconds or server_wait > auto_resume_after_seconds:
                        record["retry_after_exceeds_policy"] = True
                        raise
                pending_error = exc
            else:
                raise
        else:
            record["status"] = "accepted"
            journal["accepted_attempt"] = index + 1
            return decision
        finally:
            record["seconds"] = time.monotonic() - started
            # Canonical files show the last attempt; the separate directories retain all attempts.
            for name in ("events.jsonl", "stderr.log", "response.json"):
                target = directory / name
                if (attempt / name).exists():
                    shutil.copyfile(attempt / name, target)
                else:
                    target.unlink(missing_ok=True)
            stats_path = attempt / "stats.json"
            try:
                stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
                if not isinstance(stats, dict):
                    raise ValueError("Provider stats must be an object")
            except ValueError as exc:
                # A request can fail while its provider is writing statistics.
                # Keep the partial file and the real request error independently.
                if stats_path.exists():
                    shutil.copyfile(stats_path, attempt / "stats.partial.json")
                stats = {}
                record["stats_error"] = str(exc)
            stats.setdefault("usage", None)
            stats.setdefault("seconds", record["seconds"])
            stats.update(requested_model=model, requested_effort=effort,
                         requested_sandbox_mode=sandbox_mode, requested_approval_policy=approval_policy,
                         configuration_provenance="requested", configuration_verified=False)
            attempt.mkdir(parents=True, exist_ok=True)
            AuditJournal.write(stats_path, stats)
            usage = stats.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            record["usage_known"] = (
                usage.get("input_tokens", usage.get("prompt_tokens")) is not None
                and usage.get("output_tokens", usage.get("completion_tokens")) is not None
            )
            stats.update(
                model=model,
                effort=effort,
                seconds=sum(r.get("seconds", 0) for r in journal["attempts"]),
                total_wall_seconds=time.monotonic() - wall_started,
                request_attempts=len(journal["attempts"]),
                transport_retries=index,
                retry_wait_seconds=sum(r.get("retry_wait_seconds", 0) for r in journal["attempts"]),
                failed_attempt_usage_unknown=sum(
                    r["status"] != "accepted" and not r.get("usage_known", False)
                    for r in journal["attempts"]
                ),
            )
            if resumed:
                stats["recovery_pause_seconds"] = journal.get("recovery_pause_seconds", 0)
            AuditJournal.write(directory / "stats.json", stats)
            AuditJournal.write(directory / "retry.json", journal)
        if retry_wait is not None:
            time.sleep(retry_wait)
        if pending_error is not None:
            from fds.common.recovery import wait_for_resume

            if auto_resume_after_seconds:
                from fds.common.recovery import automatic_cooldown
                paused = automatic_cooldown(directory, auto_resume_after_seconds, pending_error, index + 1)
                resumed = True
                journal["automatic_resume"] = True
            else:
                resumed, paused = wait_for_resume(directory, pause_seconds, pending_error, index + 1)
            journal["recovery_pause_seconds"] = paused
            journal["manual_resume"] = resumed and not auto_resume_after_seconds
            stats["recovery_pause_seconds"] = paused
            stats["total_wall_seconds"] = time.monotonic() - wall_started
            AuditJournal.write(directory / "stats.json", stats)
            AuditJournal.write(directory / "retry.json", journal)
            if not resumed:
                raise pending_error
            # At most one extra recovery window, with the same bounded retry budget.
            budget = index + 1 + retries
            window_index = -1
        index += 1
        window_index += 1
