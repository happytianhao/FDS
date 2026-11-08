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


def retryable_transport_error(error):
    """Recognize transport failures only, without requiring the optional SDK."""
    if not isinstance(error, Exception):
        return False  # KeyboardInterrupt/SystemExit must never trigger a request.
    if isinstance(error, (TransportError, subprocess.TimeoutExpired)):
        return True
    try:
        import openai
    except ImportError:
        return False
    if isinstance(error, openai.APIConnectionError):
        return True
    if isinstance(error, openai.APIStatusError):
        # A depleted account is not a transient rate limit, even when HTTP is 429.
        body = error.body or {}
        detail = body.get("error", body) if isinstance(body, dict) else {}
        codes = (detail.get("code"), detail.get("type")) if isinstance(detail, dict) else ()
        if any(
            code in ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active")
            for code in codes
        ):
            return False
        return error.status_code in (408, 409, 429, 500, 502, 503, 504)
    return False


def transient_codex_failure(message, events):
    """Conservative allowlist: unknown errors, completed turns and tools never retry."""
    for event in events:
        if event.get("type") == "turn.completed":
            return False
        item_type = event.get("item", {}).get("type")
        if item_type and item_type not in ("reasoning", "agent_message", "error"):
            return False
    value = message.lower()
    if any(
        word in value
        for word in (
            "unauthorized",
            "authentication",
            "forbidden",
            "quota",
            "billing_hard_limit_reached",
            "billing_not_active",
            "certificate",
            "max_output_tokens",
            "context_length_exceeded",
            "invalid_json_schema",
        )
    ):
        return False
    return any(
        word in value
        for word in (
            "connection failed: error sending request",
            "connection reset by peer",
            "stream disconnected before completion",
            "request timed out",
            "429 too many requests",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
            "currently experiencing high demand, which may cause temporary errors",
            "selected model is at capacity",
        )
    )


def retry_decision(
    decide, context, images, directory, *, retries, delay, max_delay, model, effort, pause_seconds=0
):
    """Return one accepted decision; preserve every failed request in its own folder."""
    from fds.common.audit import AuditJournal
    from fds.decision import decision_schema
    from fds.planners import make_prompt

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
    index, budget, resumed = 0, retries, False
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
            transient = retryable_transport_error(exc)
            record.update(
                status="transport_error" if transient else "fatal_error",
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
                    max_delay, delay * 2**index * random.SystemRandom().uniform(0.8, 1.2)
                )
                record["retry_wait_seconds"] = retry_wait
            elif pause_seconds and not resumed:
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
                stats = {}
                record["stats_error"] = str(exc)
            stats.update(
                model=model,
                effort=effort,
                seconds=sum(r.get("seconds", 0) for r in journal["attempts"]),
                request_attempts=len(journal["attempts"]),
                transport_retries=index,
                retry_wait_seconds=sum(r.get("retry_wait_seconds", 0) for r in journal["attempts"]),
                failed_attempt_usage_unknown=sum(
                    r["status"] != "accepted" for r in journal["attempts"]
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

            resumed, paused = wait_for_resume(directory, pause_seconds, pending_error, index + 1)
            journal["recovery_pause_seconds"] = paused
            journal["manual_resume"] = resumed
            stats["recovery_pause_seconds"] = paused
            AuditJournal.write(directory / "stats.json", stats)
            AuditJournal.write(directory / "retry.json", journal)
            if not resumed:
                raise pending_error
            # At most one extra recovery window, with the same bounded retry budget.
            budget = index + 1 + retries
        index += 1
