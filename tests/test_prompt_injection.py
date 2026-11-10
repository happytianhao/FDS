"""Validate actual command construction and message separation without API calls."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

from fds.planners import CodexPlanner
from fds.task_harness.contracts import definitions
from fds.task_harness.protocol import (
    SYSTEM_PROMPT,
    make_system_prompt,
    make_user_prompt,
    mock_decision,
)


def context():
    return {
        "decision_protocol": "task_harness", "contract_mode": "full",
        "tool_definitions": definitions(8, False, 4),
        "max_tool_steps": 8, "max_tool_calls": 8,
        "execution_contract": {"test_rule": "fixed-rule-sentinel"},
        "task_goal": "dynamic-goal-sentinel", "memory": {"events": []}, "skills": [],
    }


def fake_process(monkeypatch, answer, captured):
    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["settings"] = dict(
                command[i + 1].split("=", 1)
                for i, argument in enumerate(command[:-1]) if argument == "-c"
            )
            path = json.loads(captured["settings"]["model_instructions_file"])
            captured["instructions"] = Path(path).read_text()
            self.command, self.output = command, kwargs["stdout"]

        def communicate(self, prompt, timeout):
            captured["user_prompt"] = prompt
            Path(self.command[self.command.index("-o") + 1]).write_text(json.dumps(answer))
            self.output.write(json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 9, "output_tokens": 3,
            }}) + "\n")

    monkeypatch.setattr("subprocess.Popen", Process)


def test_codex_overrides_base_instructions_and_disables_skills(tmp_path, monkeypatch):
    ctx, captured = context(), {}
    fake_process(monkeypatch, asdict(mock_decision(ctx)), captured)
    planner = object.__new__(CodexPlanner)
    planner.model, planner.effort, planner.timeout = "gpt-6-astra", "medium", 20
    planner.environment = {}
    decision = planner._decide_once(ctx, [], tmp_path / "call")
    assert decision.tool_batch["tool_calls"][0]["name"] == "wait"
    assert captured["instructions"] == make_system_prompt(ctx)
    assert captured["user_prompt"] == make_user_prompt(ctx)
    assert "fixed-rule-sentinel" in captured["instructions"]
    assert "dynamic-goal-sentinel" not in captured["instructions"]
    assert "fixed-rule-sentinel" not in captured["user_prompt"]
    assert "dynamic-goal-sentinel" in captured["user_prompt"]
    assert SYSTEM_PROMPT not in captured["user_prompt"]
    assert captured["settings"]["skills.include_instructions"] == "false"
    assert captured["settings"]["approval_policy"] == '"never"'
    assert captured["settings"]["model_reasoning_effort"] == '"medium"'
    cmd = captured["command"]
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
    for feature in ("skill_search", "plugins", "shell_tool", "memories"):
        assert any(cmd[i:i + 2] == ["--disable", feature] for i in range(len(cmd) - 1))
    manifest = json.loads((tmp_path / "call/prompt_manifest.json").read_text())
    assert manifest["prompt_protocol"] == "codex_base_instructions_v1"
    assert manifest["skills_include_instructions"] is False
    stats = json.loads((tmp_path / "call/stats.json").read_text())
    assert stats["skills_include_instructions"] is False


def test_rule_split_leaves_episode_state_in_user_input():
    ctx = context()
    ctx.update(outcome_protocol={"version": "postcheck-rule"},
               pending_outcome_review={"action_call": 7},
               last_environment_result={"executed_steps": 5})
    instructions = make_system_prompt(ctx)
    user = json.loads(make_user_prompt(ctx))
    assert "postcheck-rule" in instructions
    assert "pending_outcome_review" not in instructions
    assert user["pending_outcome_review"]["action_call"] == 7
    assert user["last_environment_result"]["executed_steps"] == 5
    assert "outcome_protocol" not in user and "contract_mode" not in user
    assert user["skills"] == []


def test_interface_ablation_uses_same_real_override(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/experiments/interface_ablation/run_one.py"
    spec = importlib.util.spec_from_file_location("interface_ablation_under_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    captured = {}
    answer = {"description": "Offline fixture", "waypoint": None, "gripper": 0}
    fake_process(monkeypatch, answer, captured)
    prompt, images = runner.request_content("dynamic-task", [])
    actual, _ = runner.infer(prompt, images, tmp_path / "call", 20, system="fixed-upstream-rules")
    assert actual == answer
    assert captured["instructions"] == "fixed-upstream-rules"
    assert "fixed-upstream-rules" not in captured["user_prompt"]
    assert "dynamic-task" in captured["user_prompt"]
    assert captured["settings"]["skills.include_instructions"] == "false"
    settings = json.loads((tmp_path / "call/requested_settings.json").read_text())
    assert settings["prompt_protocol"] == "codex_base_instructions_v1"
