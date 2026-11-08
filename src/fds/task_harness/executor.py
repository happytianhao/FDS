"""Public task-harness registry used by the runner and meta harness."""

EXECUTOR_NAME = "task_harness"
EXECUTORS = (EXECUTOR_NAME,)


def executor_slug(method: str = EXECUTOR_NAME) -> str:
    if method != EXECUTOR_NAME:
        raise ValueError(f"FDS supports only {EXECUTOR_NAME}; got {method}")
    return EXECUTOR_NAME


def recorded_executor(method: str = EXECUTOR_NAME, memory_enabled: bool = True) -> str:
    if method != EXECUTOR_NAME:
        raise ValueError(f"Unknown FDS executor: {method}")
    return EXECUTOR_NAME
