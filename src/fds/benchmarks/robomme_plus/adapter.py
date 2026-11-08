from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np

from fds.benchmarks.robomme.adapter import TASKS, RoboMME
from fds.common.paths import workspace_root
from fds.types import ActionSpec


class RoboMMEPlus(RoboMME):
    splits = ("train", "test")
    accounting_fields = (
        "max_steps",
        "remaining_steps",
        "execution_control_steps",
        "execution_penalty_steps",
        "execution_total_control_steps",
    )
    feedback_fields = (
        "action_error",
        "error_message",
        "exception_type",
        "error_penalty_steps",
        "trial_reset",
        "trial_index",
        "trials_remaining",
        "message",
    )

    def __init__(self, task="BinFill", split="test", max_steps=10000, calibration=True):
        self.output = None
        self.identity = None
        self.accounting = {}
        self.public_feedback = {}
        self.official_status = "unknown"
        self.failure_reason = None
        self.run_error = None
        self.stop_reason = None
        self.validation_only = False
        self.executor = "task_harness"
        self.task, self.split = task, split
        super().__init__(task, split, max_steps, calibration)

    def _make_builder(self, task, split, max_steps):
        import robomme_plus

        self.repo = workspace_root() / "third_party/robomme_plus"
        spec = tomllib.loads((workspace_root() / "configs/dependencies.toml").read_text())[
            "robomme_plus"
        ]
        self.revision = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if self.revision != spec["revision"]:
            raise RuntimeError("RoboMME-Plus source differs from configs/dependencies.toml")
        if not Path(robomme_plus.__file__).resolve().is_relative_to(self.repo):
            raise RuntimeError("Run RoboMME-Plus using the project's .venv-robomme-plus")
        if set(robomme_plus.TASKS) != set(TASKS):
            raise RuntimeError("RoboMME-Plus task registry differs from the adapter")
        return robomme_plus.BenchmarkEnvBuilder(
            task, dataset=split, action_space="joint_angle", gui_render=False
        )

    def _make_action_spec(self):
        spec = super()._make_action_spec()
        if spec.dim == 15:
            # The public joint_angle API accepts seven positions plus gripper.
            # Some released controllers internally add seven velocity channels.
            indices = [*range(7), 14]
            return ActionSpec(spec.low[indices], spec.high[indices])
        return spec

    def set_output(self, output):
        self.output = Path(output)

    def set_executor(self, method):
        if method != "task_harness":
            raise ValueError("Unsupported RoboMME-Plus executor")
        self.executor = method

    def set_validation_only(self, value):
        self.validation_only = bool(value)

    def reset(self, episode):
        self.identity = None
        self.accounting = {}
        self.run_error = self.stop_reason = self.failure_reason = None
        self.public_feedback = {}
        self.official_status = "unknown"
        record = self.builder.get_episode_metadata(episode)
        observation = super().reset(episode)
        # Evaluator provenance only. No test metadata is passed to the policy.
        self.identity = {
            "benchmark": "robomme_plus",
            "task": self.task,
            "split": self.split,
            "episode": episode,
            "seed": record["seed"],
            "difficulty": record["difficulty"],
            "variant": record.get("variant"),
            "control_freq": record["control_freq"],
            "official_max_steps": self.builder.episode_max_steps(episode),
            "episode_metadata_sha256": hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest(),
            "source_revision": self.revision,
            "controller": f"joint_angle via {self.executor} TCP/IK",
            "validation_only": self.validation_only,
        }
        self._write_metadata()
        return observation

    def _record_info(self, info):
        self.accounting = {key: int(info[key]) for key in self.accounting_fields if key in info}
        self.public_feedback = {key: info[key] for key in self.feedback_fields if key in info}
        self.official_status = str(info.get("status", "unknown"))
        self.failure_reason = info.get("failure_reason")

    def remaining_execution_steps(self):
        value = self.accounting.get("remaining_steps")
        if value is None or value < 0:
            raise ValueError("RoboMME-Plus did not supply a valid public remaining_steps")
        return value

    def step(self, action):
        result = super().step(action)
        result.feedback = dict(self.public_feedback)
        return result

    def set_run_outcome(self, stop_reason, error):
        self.stop_reason, self.run_error = stop_reason, error

    def benchmark_metrics(self):
        return {
            "success": self.official_status == "success" and self.run_error is None,
            "official_status": self.official_status,
            "termination": self.stop_reason,
            "completed_without_error": self.run_error is None,
            "failure_reason": self.failure_reason,
            **self.accounting,
        }

    def _write_metadata(self):
        if self.output is not None and self.identity is not None:
            document = {
                **self.identity,
                "result": self.benchmark_metrics(),
                "run_error": self.run_error,
            }
            (self.output / "benchmark_metadata.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2, default=_json_value) + "\n"
            )

    def close(self):
        try:
            super().close()
        finally:
            self._write_metadata()


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")
