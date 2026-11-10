import json
import subprocess
from types import SimpleNamespace

import pytest

from fds.common.recovery import RecoveryConfig
from fds.common.retry import (
    TransportError,
    classify_request_failure,
    retry_decision,
    transient_codex_failure,
)


@pytest.mark.parametrize('error,category,retry,blocker', [
    (TimeoutError(), 'timeout', True, False),
    (ConnectionResetError(), 'transport', True, False),
    (RuntimeError('unknown'), 'unknown', False, False),
    ('rate_limit_exceeded', 'rate_limit', True, False),
    ('insufficient_quota', 'quota', False, True),
    ('authentication failed', 'authentication', False, True),
    ('invalid_json_schema', 'schema', False, False),
    (KeyboardInterrupt(), 'interrupted', False, False),
    (SystemExit(), 'interrupted', False, False),
])
def test_classifier(error, category, retry, blocker):
    assert classify_request_failure(error) == dict(failure_category=category, retryable=retry, global_blocker=blocker)


def test_codex_cannot_replay_completed_or_tool_events():
    for event in ({'type': 'turn.completed'}, {'type': 'item.completed', 'item': {'type': 'command_execution'}}):
        assert not transient_codex_failure('request timed out', [event])
    assert transient_codex_failure('error sending request for url', [])
    error = subprocess.TimeoutExpired('codex', 600)
    error.request_replay_forbidden = True
    assert not classify_request_failure(error)['retryable']


def invoke(tmp_path, monkeypatch, fail, **options):
    monkeypatch.setattr('fds.planners.make_prompt', lambda c: json.dumps(c))
    monkeypatch.setattr('fds.decision.decision_schema', lambda c: {})
    waits = []
    monkeypatch.setattr('fds.common.retry.time.sleep', waits.append)
    monkeypatch.setattr('fds.common.retry.random.SystemRandom.uniform', lambda *a: 1)
    calls = []
    image = tmp_path / 'image.png'
    image.write_bytes(b'unchanged')
    context = {'memory': ['original']}
    def decide(ctx, images, directory):
        calls.append((json.loads(json.dumps(ctx)), images[0].read_bytes()))
        ctx['memory'].append('local mutation')
        fail(len(calls))
        return 'accepted'
    params = dict(retries=3, delay=2, max_delay=20, model='fixture', effort='medium', sandbox_mode='workspace-write', approval_policy='never')
    params.update(options)
    try:
        value = retry_decision(decide, context, [image], tmp_path / 'request', **params)
    except BaseException as error:
        value = error
    assert context == {'memory': ['original']}
    assert all(c == ({'memory': ['original']}, b'unchanged') for c in calls)
    return value, calls, waits, json.loads((tmp_path / 'request/retry.json').read_text())


def test_bounded_cooldown_and_fresh_backoff(tmp_path, monkeypatch):
    def fail(n):
        raise TransportError('network')
    result, calls, waits, journal = invoke(tmp_path, monkeypatch, fail, auto_resume_after_seconds=60)
    assert isinstance(result, TransportError)
    assert len(calls) == 8
    assert waits == [2, 4, 8, 60, 2, 4, 8]
    assert journal['automatic_resume'] and not journal['manual_resume']
    assert journal['environment_advanced_during_retry'] is False
    for n in range(1, 9):
        stats = json.loads((tmp_path / f'request/attempt_{n:04d}/stats.json').read_text())
        assert stats['usage'] is None
        assert stats['requested_sandbox_mode'] == 'workspace-write'
        assert stats['requested_approval_policy'] == 'never'
        assert not stats['configuration_verified']


@pytest.mark.parametrize('error', [RuntimeError('insufficient_quota'), KeyboardInterrupt(), SystemExit(), RuntimeError('unknown')])
def test_no_retry_on_permanent_or_shutdown(tmp_path, monkeypatch, error):
    def fail(n):
        raise error
    result, calls, waits, _ = invoke(tmp_path, monkeypatch, fail, auto_resume_after_seconds=60)
    assert result is error
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize('server_wait,attempts,waits', [('7', 2, [7]), ('100', 1, [])])
def test_retry_after_respects_policy(tmp_path, monkeypatch, server_wait, attempts, waits):
    error = TransportError('network')
    error.response = SimpleNamespace(headers={'retry-after': server_wait})
    def fail(n):
        if n == 1:
            raise error
    _, calls, actual, _ = invoke(tmp_path, monkeypatch, fail)
    assert len(calls) == attempts and actual == waits


@pytest.mark.parametrize('value', [-1, 3601, float('inf'), True])
def test_invalid_cooldown(value):
    with pytest.raises(ValueError):
        RecoveryConfig(auto_resume_after_seconds=value)


@pytest.mark.parametrize('status,code,retry,category', [
    (429, 'insufficient_quota', False, 'quota'),
    (429, 'rate_limit_exceeded', True, 'rate_limit'),
    (401, None, False, 'authentication'),
    (403, None, False, 'authorization'),
    (400, None, False, 'invalid_request'),
    (503, None, True, 'service_unavailable'),
])
def test_http_status_and_quota(status, code, retry, category):
    error = RuntimeError('provider failed')
    error.status_code = status
    error.body = {'error': {'code': code}}
    result = classify_request_failure(error)
    assert result['retryable'] is retry and result['failure_category'] == category


def test_recovery_returns_single_accepted_decision(tmp_path, monkeypatch):
    def fail(n):
        if n <= 4:
            raise TransportError('network')
    result, calls, waits, journal = invoke(tmp_path, monkeypatch, fail, auto_resume_after_seconds=60)
    assert result == 'accepted' and len(calls) == 5
    assert journal['accepted_attempt'] == 5
    assert waits == [2, 4, 8, 60]


def test_default_has_no_automatic_second_window(tmp_path, monkeypatch):
    def fail(n):
        raise TransportError('network')
    _, calls, waits, journal = invoke(tmp_path, monkeypatch, fail)
    assert len(calls) == 4 and waits == [2, 4, 8]
    assert not journal.get('automatic_resume')


def test_interrupted_cooldown_never_resends(tmp_path, monkeypatch):
    from fds.common.recovery import automatic_cooldown
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr('fds.common.recovery.time.sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        automatic_cooldown(tmp_path, 60, TransportError('network'), 4)
    assert json.loads((tmp_path / 'recovery.json').read_text())['status'] == 'interrupted'
