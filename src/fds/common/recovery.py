"""Request recovery while the synchronous episode worker retains its live state.

These files are control messages, not simulator checkpoints or model memory.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from fds.common.audit import AuditJournal


@dataclass(frozen=True)
class RecoveryConfig:
    transport_retries: int = 3
    retry_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 20.0
    pause_seconds: float = 300.0
    auto_resume_after_seconds: float = 0.0

    def __post_init__(self):
        if type(self.transport_retries) is not int or not 0 <= self.transport_retries <= 5:
            raise ValueError("recovery.transport_retries must be an integer in 0..5")
        values = (self.retry_delay_seconds, self.retry_max_delay_seconds, self.pause_seconds, self.auto_resume_after_seconds)
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 0 <= self.retry_delay_seconds <= self.retry_max_delay_seconds <= 60
            or not 0 <= self.pause_seconds <= 3600
            or not 0 <= self.auto_resume_after_seconds <= 3600
        ):
            raise ValueError("Invalid recovery delays: 0 <= delay <= max <= 60; 0 <= pause <= 3600")


def planner_recovery_options(config):
    """Resolve defaults once, including the original Codex planner overrides."""
    options = dict(config.get("recovery", {}))
    planner = config["planner"]
    for key in ("transport_retries", "retry_delay_seconds", "retry_max_delay_seconds"):
        if key in planner:
            options[key] = planner[key]
    return RecoveryConfig(**options)


class RecoveringPlanner:
    def __init__(self, planner, options):
        self.planner, self.options = planner, options

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def decide(self, context, images, directory):
        from fds.common.retry import retry_decision

        options = self.options
        if not options.transport_retries and not options.auto_resume_after_seconds:
            return self.planner.decide(context, images, directory)
        return retry_decision(
            self.planner.decide,
            context,
            images,
            directory,
            retries=options.transport_retries,
            delay=options.retry_delay_seconds,
            max_delay=options.retry_max_delay_seconds,
            model=getattr(self.planner, "model", None),
            effort=getattr(self.planner, "effort", None),
            pause_seconds=options.pause_seconds,
            auto_resume_after_seconds=options.auto_resume_after_seconds,
            sandbox_mode=getattr(self.planner, "sandbox_mode", None),
            approval_policy=getattr(self.planner, "approval_policy", None),
        )

    def close(self):
        if hasattr(self.planner, "close"):
            self.planner.close()


def process_identity(pid):
    """Linux start time prevents a reused PID from accepting a stale resume file."""
    try:
        # comm may contain spaces and parentheses; starttime is field 22.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in ("Z", "X") else fields[19]
    except (OSError, IndexError):
        return None


def wait_for_resume(directory, seconds, error, attempt):
    """Allow one bounded operator recovery window after automatic retries fail."""
    directory = Path(directory)
    status_path, request_path = directory / "recovery.json", directory / "resume.json"
    request_path.unlink(missing_ok=True)
    started = time.monotonic()
    state = {
        "status": "waiting",
        "token": uuid.uuid4().hex,
        "pid": os.getpid(),
        "process_start": process_identity(os.getpid()),
        "host": socket.gethostname(),
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "started_utc": datetime.now(UTC).isoformat(),
        "deadline_unix": time.time() + seconds,
        "max_pause_seconds": seconds,
        "environment_held": True,
        "process_checkpoint": False,
    }
    AuditJournal.write(status_path, state)
    print(
        json.dumps(
            {
                "event": "request_recovery_wait",
                "directory": str(directory),
                "seconds": seconds,
                "resume": f"python -m fds.cli resume {directory}",
                "error": str(error),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        while time.monotonic() - started < seconds:
            if request_path.exists():
                try:
                    request = json.loads(request_path.read_text())
                except (OSError, json.JSONDecodeError):
                    request = {}
                if not isinstance(request, dict):
                    request = {}
                if request.get("token") == state["token"]:
                    state["status"] = "resumed"
                    return True, time.monotonic() - started
                request_path.unlink(missing_ok=True)
            time.sleep(min(0.25, max(0, seconds - (time.monotonic() - started))))
        state["status"] = "expired"
        return False, time.monotonic() - started
    except BaseException:
        state["status"] = "interrupted"
        raise
    finally:
        state["wait_seconds"] = time.monotonic() - started
        state["environment_held"] = False
        AuditJournal.write(status_path, state)
        request_path.unlink(missing_ok=True)


def request_resume(directory):
    """Signal live waiting workers only; never reset or rerun an exited episode."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Recovery directory does not exist: {directory}")
    paths = (
        [directory / "recovery.json"]
        if (directory / "recovery.json").is_file()
        else sorted(directory.glob("**/call_*/recovery.json"))
    )
    requested, inactive = [], []
    for path in paths:
        state = json.loads(path.read_text())
        if state.get("status") != "waiting":
            continue
        identity = process_identity(state.get("pid"))
        if (
            state.get("host") != socket.gethostname()
            or identity is None
            or identity != state.get("process_start")
            or time.time() >= state["deadline_unix"]
        ):
            inactive.append(str(path.parent))
            continue
        # Unique temp files make concurrent CLI invocations safe as well.
        target = path.parent / "resume.json"
        temporary = target.with_name(f".resume-{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps({"token": state["token"], "requested_utc": datetime.now(UTC).isoformat()})
        )
        temporary.replace(target)
        requested.append(str(path.parent))
    return {
        "requested": requested,
        "inactive": inactive,
        "note": "Only live waiting requests can resume; exited simulators are not restored.",
    }


def effective_recovery(config):
    return asdict(planner_recovery_options(config))


def automatic_cooldown(directory, seconds, error, attempt):
    """One synchronous cooldown: no environment, tools, or memory are advanced."""
    started = time.monotonic()
    state = {"status": "automatic_cooldown", "attempt": attempt, "error": str(error),
             "error_type": type(error).__name__, "max_pause_seconds": seconds,
             "environment_held": True, "process_checkpoint": False}
    path = Path(directory) / "recovery.json"
    AuditJournal.write(path, state)
    try:
        time.sleep(seconds)
        state["status"] = "automatically_resumed"
    except BaseException:
        state["status"] = "interrupted"
        raise
    finally:
        state["wait_seconds"] = time.monotonic() - started
        state["environment_held"] = False
        AuditJournal.write(path, state)
    return state["wait_seconds"]
