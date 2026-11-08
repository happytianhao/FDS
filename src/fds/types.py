from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Frame:
    index: int
    images: dict[str, np.ndarray]
    state: np.ndarray  # seven joints in rad + first finger position in meters
    cue: bool = False
    calibration: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    goal: str
    frames: list[Frame]

    @property
    def current(self) -> Frame:
        return self.frames[-1]


@dataclass
class StepResult:
    observation: Observation | None
    terminated: bool
    truncated: bool
    status: str  # scorer-only; never included in a planner observation
    feedback: dict[str, Any] = field(default_factory=dict)  # public execution errors/notices only

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated or self.status == "error"


@dataclass
class ActionSpec:
    low: np.ndarray
    high: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.low)

    def validate(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.low.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected a finite action with shape {self.low.shape}")
        invalid = np.flatnonzero((action < self.low) | (action > self.high))
        if len(invalid):
            index = int(invalid[0])
            raise ValueError(
                f"Action exceeds robot joint or gripper limits at dimension {index}: "
                f"{action[index]:.8g} outside [{self.low[index]:.8g}, {self.high[index]:.8g}]"
            )
        return action


class Benchmark(Protocol):
    action_spec: ActionSpec

    def reset(self, episode: int) -> Observation: ...
    def step(self, action: np.ndarray) -> StepResult: ...
    def close(self) -> None: ...


class Policy(Protocol):
    def reset(self) -> None: ...
    def infer(self, observation: Observation, subgoal: str) -> np.ndarray: ...
    def close(self) -> None: ...
