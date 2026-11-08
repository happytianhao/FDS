"""The two supported RoboMME benchmark adapters."""

from dataclasses import dataclass
from importlib import import_module

from fds.task_harness.executor import EXECUTOR_NAME


@dataclass(frozen=True)
class BenchmarkSpec:
    module: str
    adapter: str
    splits: tuple[str, ...]
    control: str


BENCHMARKS = {
    "robomme_plus": BenchmarkSpec(
        "fds.benchmarks.robomme_plus",
        "RoboMMEPlus",
        ("train", "test"),
        "panda_joint_control",
    ),
    "robomme": BenchmarkSpec(
        "fds.benchmarks.robomme",
        "RoboMME",
        ("train", "val", "test"),
        "panda_joint_control",
    ),
}


def benchmark_spec(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise ValueError(
            f"Unsupported benchmark: {name}; choose robomme_plus or robomme"
        ) from None


def make_benchmark(name: str, **kwargs):
    spec = benchmark_spec(name)
    return getattr(import_module(spec.module), spec.adapter)(**kwargs)


def benchmark_tasks(name: str):
    return import_module(benchmark_spec(name).module).TASKS


def validate_executor(name: str, method: str, *, policy_enabled: bool = False):
    benchmark_spec(name)
    if method != EXECUTOR_NAME:
        raise ValueError(f"FDS supports only {EXECUTOR_NAME}; got {method}")
    if policy_enabled:
        raise ValueError("FDS task_harness is a direct Astra/Sys2 executor and does not use a VLA")
