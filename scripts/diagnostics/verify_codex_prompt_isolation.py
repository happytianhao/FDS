#!/usr/bin/env python3
"""Check Codex request construction against a rejecting localhost-only provider.

No model is called. A fresh CODEX_HOME has no real credentials, the provider
uses a disposable placeholder, and the HTTP listener never forwards requests.
Only booleans/counts are printed; request bodies and headers stay unreported.

Run with the project environment:
    .venv/bin/python scripts/diagnostics/verify_codex_prompt_isolation.py

Use --codex-binary for installations whose launcher cannot be safely resolved.
The known local Python launcher is inspected only for its literal binary path;
it is never executed because it overrides CODEX_HOME with an account profile.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_MARKER = "FDS_DIAGNOSTIC_SYSTEM_8A42"
USER_MARKER = "FDS_DIAGNOSTIC_USER_A39C"
SKILL_MARKERS = ("### Available skills", "openai-docs", "skill-creator", "skill-installer")
DISABLED_FEATURES = (
    "unbounded_connection_retries", "view_image", "shell_tool", "unified_exec",
    "sleep_tool", "collaboration_modes", "multi_agent", "apps", "plugins",
    "browser_use", "computer_use", "image_generation", "hooks", "memories",
    "skill_search",
)


def resolve_binary(explicit: str | None) -> Path:
    path = Path(explicit or shutil.which("codex") or "")
    if not path.is_file():
        raise ValueError("Codex executable unavailable; supply --codex-binary")
    path = path.resolve()
    with path.open("rb") as handle:
        prefix = handle.read(4)
    if prefix == b"\x7fELF":
        return path
    # Never run a launcher that may restore real account settings. Do not
    # interpret arbitrary code or read the credentials/config it references.
    try:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "binary"
                       for target in node.targets):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                candidate = Path(node.value.value)
                if candidate.is_file():
                    with candidate.open("rb") as handle:
                        if handle.read(4) == b"\x7fELF":
                            return candidate.resolve()
    except (SyntaxError, UnicodeError):
        pass
    raise ValueError("Launcher not safely resolvable; supply the native --codex-binary")


def fixture() -> tuple[str, str]:
    sys.path.insert(0, str(ROOT / "src"))
    from fds.task_harness.contracts import definitions
    from fds.task_harness.protocol import make_system_prompt, make_user_prompt

    context = {
        "decision_protocol": "task_harness", "contract_mode": "full",
        "tool_definitions": definitions(8, False, 4),
        "max_tool_steps": 8, "max_tool_calls": 8,
        "execution_contract": {"diagnostic_marker": SYSTEM_MARKER},
        "task_goal": USER_MARKER, "memory": {"events": []}, "skills": [],
    }
    return make_system_prompt(context), make_user_prompt(context)


def summarize_request(body: dict, fixed: str, user: str) -> dict:
    messages = [item for item in body.get("input", []) if isinstance(item, dict)]
    text_by_role: dict[str, list[str]] = {}
    for message in messages:
        content = message.get("content", [])
        text = ("".join(item.get("text", "") for item in content if isinstance(item, dict))
                if isinstance(content, list) else str(content))
        text_by_role.setdefault(str(message.get("role")), []).append(text)
    # Some model catalogs serialize base instructions as a developer input;
    # other catalogs use the Responses top-level instructions field.
    authority = "\n".join([str(body.get("instructions", "")),
                           *text_by_role.get("developer", []),
                           *text_by_role.get("system", [])])
    user_text = "\n".join(text_by_role.get("user", []))
    all_text = authority + "\n" + user_text
    return {
        "message_count": len(messages),
        "authority_characters": len(authority),
        "user_characters": len(user_text),
        "native_coding_base_present": "You are Codex" in authority,
        "fixed_instructions_in_authority": fixed in authority,
        "fixed_marker_in_authority": SYSTEM_MARKER in authority,
        "fixed_marker_in_user": SYSTEM_MARKER in user_text,
        "user_prompt_preserved": user in user_text,
        "user_marker_in_authority": USER_MARKER in authority,
        "skill_catalog_marker_count": sum(marker in all_text for marker in SKILL_MARKERS),
        "generic_skills_heading_present": "## Skills" in all_text,
        "requested_model_preserved": body.get("model") == "gpt-6-astra",
        "requested_effort_preserved": body.get("reasoning", {}).get("effort") == "medium",
    }


def invoke(command: list[str], environment: dict[str, str], timeout: float) -> dict:
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    return {"exit_code": process.returncode, "timed_out": timed_out,
            "stdout_bytes": len(stdout), "stderr_bytes": len(stderr)}


def run(args) -> dict:
    binary = resolve_binary(args.codex_binary)
    fixed, user = fixture()
    captured: list[dict] = []

    class RejectingProvider(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 16 * 1024 * 1024:
                self.send_error(413)
                return
            try:
                body = json.loads(self.rfile.read(length))
                if isinstance(body, dict):
                    captured.append(summarize_request(body, fixed, user))
            except (ValueError, UnicodeError):
                pass
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"offline diagnostic complete",'
                             b'"type":"invalid_request_error"}}')

        def do_CONNECT(self):
            self.send_error(403)

        def do_GET(self):
            self.send_error(403)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RejectingProvider)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="fds-codex-isolation-") as temporary:
            root = Path(temporary)
            codex_home, work = root / "codex-home", root / "work"
            codex_home.mkdir(mode=0o700)
            work.mkdir()
            instruction_file = root / "harness-instructions.txt"
            instruction_file.write_text(fixed, encoding="utf-8")
            # Allow-list a minimal environment instead of copying account keys,
            # gateway headers, auth-token variables, or the current CODEX_HOME.
            environment = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "TZ")
                           if name in os.environ}
            address = f"http://127.0.0.1:{server.server_port}"
            environment.update(CODEX_HOME=str(codex_home),
                               FDS_DIAGNOSTIC_TOKEN="local-placeholder",
                               HTTP_PROXY=address, HTTPS_PROXY=address, ALL_PROXY=address,
                               http_proxy=address, https_proxy=address, all_proxy=address,
                               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
            config = {
                "model": "gpt-6-astra", "model_reasoning_effort": "medium",
                "approval_policy": "never", "model_provider": "fds_diagnostic",
                "model_providers.fds_diagnostic.name": "Local rejecting diagnostic",
                "model_providers.fds_diagnostic.base_url": address + "/v1",
                "model_providers.fds_diagnostic.wire_api": "responses",
                "model_providers.fds_diagnostic.requires_openai_auth": False,
                "model_providers.fds_diagnostic.env_key": "FDS_DIAGNOSTIC_TOKEN",
                "model_providers.fds_diagnostic.request_max_retries": 0,
                "model_providers.fds_diagnostic.stream_max_retries": 0,
                "model_providers.fds_diagnostic.supports_websockets": False,
                "project_doc_max_bytes": 0, "web_search": "disabled",
            }
            base = [str(binary), "exec", "--ignore-user-config", "--ephemeral",
                    "--skip-git-repo-check", "--sandbox", "workspace-write", "--json",
                    "--cd", str(work)]
            for key, value in config.items():
                base.extend(["-c", key + "=" + json.dumps(value)])
            for feature in DISABLED_FEATURES:
                base.extend(["--disable", feature])
            results = {}
            cases = {
                "default_base_skills_disabled": {"skills.include_instructions": False},
                "harness_base_skills_default": {"model_instructions_file": str(instruction_file)},
                "harness_base_skills_disabled": {
                    "model_instructions_file": str(instruction_file),
                    "skills.include_instructions": False,
                },
            }
            for name, overrides in cases.items():
                captured.clear()
                command = list(base)
                for key, value in overrides.items():
                    command.extend(["-c", key + "=" + json.dumps(value)])
                process = invoke(command + ["--", user], environment, args.timeout)
                results[name] = {**process, "request_count": len(captured),
                                 "requests": list(captured)}
            baseline = results["default_base_skills_disabled"]["requests"]
            isolated = results["harness_base_skills_disabled"]["requests"]
            passed = bool(len(baseline) == len(isolated) == 1
                          and baseline[0]["native_coding_base_present"]
                          and isolated[0]["fixed_instructions_in_authority"]
                          and isolated[0]["fixed_marker_in_authority"]
                          and not isolated[0]["native_coding_base_present"]
                          and not isolated[0]["fixed_marker_in_user"]
                          and not isolated[0]["user_marker_in_authority"]
                          and isolated[0]["user_prompt_preserved"]
                          and isolated[0]["skill_catalog_marker_count"] == 0
                          and not isolated[0]["generic_skills_heading_present"]
                          and isolated[0]["requested_model_preserved"]
                          and isolated[0]["requested_effort_preserved"]
                          and not any(row["timed_out"] for row in results.values())
                          and not (codex_home / "auth.json").exists())
            return {"passed": passed, "localhost_only_provider": True,
                    "temporary_auth_file_present": (codex_home / "auth.json").exists(),
                    "fixed_instruction_characters": len(fixed),
                    "fixed_instruction_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
                    "cases": results}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-binary")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        report = run(args)
    except Exception as error:
        # Exceptions can contain command lines or third-party diagnostic text;
        # print only the type, never raw subprocess errors or request contents.
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
