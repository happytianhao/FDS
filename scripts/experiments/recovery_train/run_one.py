"""Isolate one train episode; preserve errors and clean up before the next scene."""
import datetime
import json
import os
import signal
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = Path(__file__).resolve().parent
mode, task, episode_text = sys.argv[1:]
episode = int(episode_text)
assert mode == "f1"
assert task in json.loads((CAMPAIGN / "campaign.json").read_text())["tasks"]
assert episode == 0
output = CAMPAIGN / f"train_{mode}" / f"{task}_ep{episode}"
assert not output.exists(), f"Refusing duplicate output: {output}"
logs = CAMPAIGN / "logs"
logs.mkdir(exist_ok=True)
stem = f"{mode}_{task}_ep{episode}"
status_path = logs / f"{stem}.status.json"
state = {"mode": mode, "task": task, "episode": episode, "output": str(output), "runner_pid": os.getpid(), "runner_create_time": psutil.Process().create_time()}


def save(**updates):
    state.update(updates, updated_utc=datetime.datetime.now(datetime.UTC).isoformat())
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(status_path)


def stop_child(child):
    try:
        descendants = psutil.Process(child.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
    for process in descendants:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(descendants, timeout=3)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def on_termination(signum, frame):
    raise InterruptedError(f"Scene runner received signal {signum}")


signal.signal(signal.SIGTERM, on_termination)
env = os.environ.copy()
env["PATH"] = str(Path.home() / ".local/bin") + os.pathsep + env.get("PATH", "")
env["FDS_WORKSPACE"] = str(ROOT)
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    env[key] = "http://127.0.0.1:21084"
env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost,::1"
env["HF_ENDPOINT"] = "https://hf-mirror.com"
config = Path(json.loads((CAMPAIGN / "campaign.json").read_text())["configs"][mode])
wall_limit = tomllib.loads(config.read_text())["run"]["episode_timeout"] + 60
cmd = [str(ROOT / ".venv/bin/fds"), "run", "--config", str(config), "--task", task, "--episodes", str(episode), "--output", str(output)]
child = None
save(status="starting", command=cmd, started_utc=datetime.datetime.now(datetime.UTC).isoformat())
try:
    with (logs / f"{stem}.log").open("xb") as stream:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(status="running", worker_pid=child.pid, worker_create_time=psutil.Process(child.pid).create_time())
        infrastructure_error = None
        try:
            child.wait(timeout=wall_limit)
        except subprocess.TimeoutExpired:
            infrastructure_error = {"error_type": "SceneTimeout", "error": f"Scene exceeded {wall_limit}s"}
            save(status="stopping_after_timeout", infrastructure_error=infrastructure_error)
            stop_child(child)
        # The worker owns bounded request retries and writes its final metrics.
        # An intermediate model_request_error must not terminate the live environment.
        save(status="completed" if child.returncode == 0 and not infrastructure_error else "failed",
             returncode=child.returncode, infrastructure_error=infrastructure_error)

except BaseException as error:
    if child is not None:
        stop_child(child)
    save(status="failed", returncode=child.returncode if child else None, infrastructure_error={"error_type": type(error).__name__, "error": str(error)}, traceback=traceback.format_exc())
