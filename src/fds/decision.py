"""FDS decision dispatch: exactly one structured task-harness protocol."""

import json


def decision_schema(context):
    if context.get("decision_protocol", "task_harness") != "task_harness":
        raise ValueError("FDS supports only the task_harness decision protocol")
    from fds.task_harness.protocol import schema_for

    return schema_for(context)


class Decision:
    @classmethod
    def parse_json(cls, response, schema):
        from fds.task_harness.protocol import DecisionValidationError

        try:
            value = json.loads(response)
        except json.JSONDecodeError as exc:
            raise DecisionValidationError(
                str(exc), code="invalid_json", path=[], response=response
            ) from exc
        return cls.parse(value, schema)

    @classmethod
    def parse(cls, value, schema=None):
        if schema is None:
            raise ValueError("FDS decisions require their explicit schema")
        from fds.task_harness.protocol import TaskDecision

        return TaskDecision.parse(value, schema)


def hold_decision(steps=4):
    """Offline transport fixture; never substituted for a failed model response."""
    from fds.task_harness.protocol import mock_decision

    return mock_decision({"max_tool_steps": steps, "max_tool_calls": 1})
