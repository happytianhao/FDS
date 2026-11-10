from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from fds.decision import Decision, decision_schema


def make_prompt(context):
    if context.get("decision_protocol") == "task_harness":
        from fds.task_harness.protocol import make_prompt

        return make_prompt(context)
    raise ValueError("FDS planner received an unsupported decision protocol")


class MockPlanner:
    def __init__(self, **_kwargs):
        pass

    def decide(self, context, images, directory):
        if context.get("decision_protocol") == "task_harness":
            from fds.task_harness.protocol import mock_decision

            return mock_decision(context)
        raise ValueError("FDS mock planner received an unsupported decision protocol")


class CodexPlanner:
    """Stateless multimodal inference using an existing official CLI login.

    No credential values are read or copied into the repository. Each call runs
    without user/project instructions, shell, external tools, or shared memory.
    """

    sandbox_mode, approval_policy = "workspace-write", "never"
    prompt_protocol = "codex_base_instructions_v1"
    skills_include_instructions = False

    def __init__(
        self,
        model="gpt-6-astra",
        effort="medium",
        timeout=180,
        login_home=None,
        transport_retries=0,
        retry_delay_seconds=2.0,
        retry_max_delay_seconds=20.0,
    ):
        if (
            type(transport_retries) is not int
            or not 0 <= transport_retries <= 5
            or not all(
                isinstance(v, (int, float)) and math.isfinite(v)
                for v in (retry_delay_seconds, retry_max_delay_seconds)
            )
            or not 0 <= retry_delay_seconds <= retry_max_delay_seconds <= 60
        ):
            raise ValueError(
                "Transport retries must be 0..5; delays must satisfy 0 <= initial <= max <= 60"
            )
        self.transport_retries = transport_retries
        self.retry_delay_seconds, self.retry_max_delay_seconds = (
            retry_delay_seconds,
            retry_max_delay_seconds,
        )
        self.model, self.effort, self.timeout = model, effort, timeout
        self.environment = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
            self.environment.pop(key, None)
        if login_home is not None:
            self.environment["CODEX_HOME"] = str(Path(login_home).expanduser().resolve())
        result = subprocess.run(
            [
                "codex",
                "-c",
                'forced_login_method="chatgpt"',
                "-c",
                'model_provider="openai"',
                "login",
                "status",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env=self.environment,
        )
        if result.returncode or "Logged in using ChatGPT" not in result.stdout + result.stderr:
            raise RuntimeError("The codex backend requires an existing ChatGPT login: codex login")

    def decide(self, context, images, directory):
        if self.transport_retries:
            from fds.common.retry import retry_decision

            return retry_decision(
                self._decide_once,
                context,
                images,
                directory,
                retries=self.transport_retries,
                delay=self.retry_delay_seconds,
                max_delay=self.retry_max_delay_seconds,
                model=self.model,
                effort=self.effort,
                sandbox_mode=self.sandbox_mode, approval_policy=self.approval_policy,
            )
        return self._decide_once(context, images, directory)

    def _decide_once(self, context, images, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        schema_path, output = directory / "schema.json", directory / "response.json"
        schema = decision_schema(context)
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        from fds.task_harness.protocol import make_system_prompt, make_user_prompt

        prompt = make_user_prompt(context)
        instructions = make_system_prompt(context)
        instructions_path = directory / "system_prompt.txt"
        instructions_path.write_text(instructions, encoding="utf-8")
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        (directory / "prompt_manifest.json").write_text(json.dumps({
            "prompt_protocol": self.prompt_protocol,
            "base_instructions_source": "model_instructions_file",
            "base_instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "skills_include_instructions": False,
            "configuration_provenance": "requested",
        }), encoding="utf-8")
        _requested_stats(directory, self)
        provider = "fds_one_pass"
        with tempfile.TemporaryDirectory(prefix="fds-inference-") as work:
            command = [
                "codex",
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--json",
                "--model",
                self.model,
                "--cd",
                work,
                "--output-schema",
                str(schema_path),
                "-o",
                str(output),
                "-c",
                'forced_login_method="chatgpt"',
                "-c",
                f'model_provider="{provider}"',
                "-c",
                f"model_reasoning_effort={json.dumps(self.effort)}",
                "-c",
                'approval_policy="never"',
                "-c",
                'web_search="disabled"',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                f"model_instructions_file={json.dumps(str(instructions_path))}",
                "-c",
                "skills.include_instructions=false",
            ]
            # Built-in provider IDs cannot be overridden. Keep official OpenAI auth
            # and default routing, with a per-process provider that cannot replay.
            for key, value in {
                "name": '"OpenAI one-pass"',
                "requires_openai_auth": "true",
                "wire_api": '"responses"',
                "request_max_retries": "0",
                "stream_max_retries": "0",
                "supports_websockets": "false",
            }.items():
                command.extend(["-c", f"model_providers.{provider}.{key}={value}"])
            command.extend(["--disable", "unbounded_connection_retries", "--disable", "view_image"])
            for feature in (
                "shell_tool",
                "unified_exec",
                "sleep_tool",
                "collaboration_modes",
                "multi_agent",
                "apps",
                "plugins",
                "browser_use",
                "computer_use",
                "image_generation",
                "hooks",
                "memories",
                "skill_search",
            ):
                command.extend(["--disable", feature])
            for path in images:
                command.extend(["--image", str(Path(path).resolve())])
            command.extend(["--", "-"])
            start = time.monotonic()
            with (
                (directory / "events.jsonl").open("w") as events,
                (directory / "stderr.log").open("w") as err,
            ):
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=events,
                    stderr=err,
                    text=True,
                    env=self.environment,
                    start_new_session=True,
                )
                try:
                    process.communicate(prompt, timeout=self.timeout)
                except BaseException as exc:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    events.flush()
                    from fds.common.retry import transient_codex_failure
                    parsed = []
                    for line in (directory / "events.jsonl").read_text().splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict):
                            parsed.append(event)
                    if not transient_codex_failure("request timed out", parsed):
                        exc.request_replay_forbidden = True
                    raise
            if process.returncode or not output.exists():
                detail = ""
                for line in reversed((directory / "events.jsonl").read_text().splitlines()):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "turn.failed":
                        error = event.get("error")
                        message = error.get("message") if isinstance(error, dict) else error
                    elif event.get("type") == "error":
                        message = event.get("message")
                    else:
                        continue
                    if isinstance(message, str) and message.strip():
                        detail = ": " + message.strip()
                        break
                from fds.common.retry import TransportError, transient_codex_failure

                parsed_events = []
                for line in (directory / "events.jsonl").read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        parsed_events.append(event)
                # The shared recovery wrapper needs classification even when it owns all retries.
                error_class = (
                    TransportError
                    if transient_codex_failure(detail, parsed_events)
                    else RuntimeError
                )
                failure = error_class(
                    f"Sys2 Codex call failed (exit {process.returncode}){detail}; "
                    f"inspect {directory / 'events.jsonl'} and {directory / 'stderr.log'}"
                )
                if error_class is RuntimeError:
                    failure.request_replay_forbidden = True
                raise failure
            usage = None
            completed = False
            for line in (directory / "events.jsonl").read_text().splitlines():
                event = json.loads(line)
                # The CLI also emits error events for recoverable reconnects.
                # Accept only an eventual completed turn; terminal failures remain fatal.
                if event.get("type") == "turn.failed":
                    raise RuntimeError("Sys2 Codex turn failed")
                item_type = event.get("item", {}).get("type", "")
                if event.get("type") == "error" or item_type == "error":
                    raise RuntimeError(
                        "One-pass Sys2 transport error; possible observation replay is invalid"
                    )
                # Transport fallback notices may be emitted as diagnostic items.
                # They do not execute a tool and still require a completed turn.
                if item_type and item_type not in ("reasoning", "agent_message", "error"):
                    if item_type == "collab_tool_call" and event.get("item", {}).get("tool") == "wait":
                        continue
                    raise RuntimeError("External tool use detected; this trial is invalid")
                if event.get("type") == "turn.completed":
                    completed, usage = True, event.get("usage")
            if not completed:
                raise RuntimeError("Sys2 turn did not complete")
            (directory / "stats.json").write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "effort": self.effort,
                        "sandbox_mode": "workspace-write",
                        "approval_policy": "never",
                        "prompt_protocol": self.prompt_protocol,
                        "skills_include_instructions": False,
                        "seconds": time.monotonic() - start,
                        "usage": usage,
                    }
                )
            )
            return Decision.parse_json(output.read_text(), schema)


class ResponsesPlanner:
    def __init__(self, model="gpt-6-astra", effort="medium", timeout=180):
        from openai import OpenAI

        self.client = OpenAI(timeout=timeout, max_retries=0)
        self.model, self.effort = model, effort

    def decide(self, context, images, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        prompt = make_prompt(context)
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        content = [{"type": "input_text", "text": prompt}]
        for path in images:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
        client = self.client.with_options(max_retries=0)
        started = time.monotonic()
        _requested_stats(directory, self)
        response = client.responses.create(
            model=self.model,
            reasoning={"effort": self.effort},
            store=False,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "robot_decision",
                    "schema": decision_schema(context),
                    "strict": True,
                }
            },
        )
        (directory / "response.json").write_text(response.model_dump_json(), encoding="utf-8")
        (directory / "stats.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "effort": self.effort,
                    "seconds": time.monotonic() - started,
                    "usage": response.usage.model_dump() if response.usage else None,
                }
            )
        )
        if response.status != "completed":
            raise RuntimeError(f"Sys2 response incomplete: {response.status}")
        return Decision.parse_json(response.output_text, decision_schema(context))

    def close(self):
        self.client.close()


class ChatPlanner:
    """Multimodal OpenAI-compatible endpoint for local Qwen and other Sys2 models."""

    def __init__(
        self,
        model,
        base_url,
        timeout=180,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=0.0,
        top_k=None,
        enable_thinking=None,
        effort=None,
        api_key_env="HARNESS_SYS2_API_KEY",
    ):
        from openai import OpenAI

        address = urlparse(base_url)
        if address.scheme not in ("http", "https") or not address.hostname or address.username:
            raise ValueError(
                "Use a valid base_url; pass authentication through an environment variable"
            )
        key = os.environ.get(api_key_env)
        if not key and address.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(f"Configure {api_key_env} for this non-local endpoint")
        self.client = OpenAI(
            base_url=base_url, api_key=key or "local", timeout=timeout, max_retries=0
        )
        self.model = model
        self.sampling = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
        }
        self.extra = {}
        if top_k is not None:
            self.extra["top_k"] = top_k
        if enable_thinking is not None:
            self.extra["chat_template_kwargs"] = {
                "enable_thinking": enable_thinking,
                "preserve_thinking": False,
            }
        if effort is not None:
            self.sampling["reasoning_effort"] = effort

    def decide(self, context, images, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        prompt = make_prompt(context)
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        content = [{"type": "text", "text": prompt}]
        for path in images:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        start = time.monotonic()
        _requested_stats(directory, self)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_decision",
                    "schema": decision_schema(context),
                    "strict": True,
                },
            },
            extra_body=self.extra,
            **self.sampling,
        )
        (directory / "response.json").write_text(response.model_dump_json(), encoding="utf-8")
        (directory / "stats.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "returned_model": response.model,
                    "seconds": time.monotonic() - start,
                    "usage": response.usage.model_dump() if response.usage else None,
                },
                indent=2,
            )
        )
        if not response.choices or response.choices[0].finish_reason != "stop":
            reason = response.choices[0].finish_reason if response.choices else "missing_choice"
            raise RuntimeError(
                f"Sys2 chat response did not finish normally: finish_reason={reason}; "
                f"raw response saved at {directory / 'response.json'}"
            )
        message = response.choices[0].message
        if message.tool_calls or not message.content:
            raise RuntimeError("Sys2 must return a JSON decision for the host to execute")
        return Decision.parse_json(message.content, decision_schema(context))

    def close(self):
        self.client.close()


def _requested_stats(directory, planner):
    """Record requested settings before a provider can fail; usage stays unknown."""
    (Path(directory) / "stats.json").write_text(json.dumps({
        "requested_model": planner.model, "requested_effort": getattr(planner, "effort", None),
        "requested_sandbox_mode": getattr(planner, "sandbox_mode", None),
        "requested_approval_policy": getattr(planner, "approval_policy", None),
        "requested_prompt_protocol": getattr(planner, "prompt_protocol", None),
        "requested_skills_include_instructions": getattr(planner, "skills_include_instructions", None),
        "configuration_provenance": "requested", "configuration_verified": False, "usage": None,
    }))
