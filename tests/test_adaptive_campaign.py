"""Offline queue safety tests: no scene/model invocation."""
import selectors
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/experiments/adaptive_train'
sys.path.insert(0, str(SCRIPTS))
import campaign as queue  # noqa: E402
from common import FileEvents, atomic, hashes, identity_alive, lock, read, sha  # noqa: E402
from run_one import worker_environment  # noqa: E402


@pytest.fixture
def out(tmp_path):
    cases = [dict(id=f'{t}_ep{e}', task=t, episode=e, difficulty=['easy', 'medium', 'hard'][e], seed=7)
             for t in ('A', 'B') for e in range(3)]
    atomic(tmp_path / 'campaign.json', dict(root=str(tmp_path), output=str(tmp_path), tasks=['A', 'B'],
           cases=cases, max_attempts=3, max_concurrency=3, config_path='unused'))
    atomic(tmp_path / 'control.json', dict(active_version='v000', halt=False))
    for version in ('v000', 'v001'):
        folder = tmp_path / 'versions' / version
        (folder / 'src/fds').mkdir(parents=True)
        (folder / 'src/fds/__init__.py').write_text('VALUE = "frozen"\n')
        (folder / 'config.toml').write_text('method = "task_harness"\n')
        atomic(folder / 'version.json', dict(id=version, source_sha256=hashes(folder / 'src'), config_sha256=sha(folder / 'config.toml')))
        atomic(folder / 'validation.json', {'passed': True})
    return tmp_path


def finish(out, case='A_ep0', index=1, success=False, version='v000'):
    task, ep = case.split('_ep')
    folder = out / 'attempts' / case / f'attempt_{index:02d}'
    req = dict(id=f'{case}_attempt_{index:02d}', case_id=case, task=task, episode=int(ep), attempt=index,
               root=str(out), out=str(out), output=str(folder / 'output'), version=version)
    atomic(folder / 'request.json', req)
    atomic(folder / 'status.json', {'status': 'completed'})
    atomic(folder / 'output' / task / f'episode_{int(ep):04d}' / 'summary.json',
           {'success': success, 'official_status': 'success' if success else 'fail'})
    atomic(folder / 'output/metrics.json', {'status': 'completed'})
    return req


def states(out):
    return {r['id']: r['state'] for r in queue.build_state(out)['cases']}


def test_entries_and_official_prerequisites(out):
    finish(out, success=True)
    assert states(out)['A_ep1'] == 'locked'  # B has not been started.
    finish(out, case='B_ep0')
    assert states(out)['A_ep1'] == 'pending'
    assert states(out)['B_ep1'] == 'locked'
    assert states(out)['B_ep0'] == 'waiting_review'
    queue.review(out, 'B_ep0_attempt_01', 'stop', 'Verified task failure')
    assert states(out)['B_ep2'] == 'blocked'


def test_movecube_variants_do_not_depend_on_each_other(out):
    manifest = read(out / 'campaign.json')
    manifest['tasks'].append('MoveCube')
    manifest['cases'].extend(dict(id=f'MoveCube_ep{ep}', task='MoveCube', episode=ep,
                                 difficulty=None, seed=ep) for ep in range(5))
    atomic(out / 'campaign.json', manifest)
    finish(out, case='MoveCube_ep0')
    queue.review(out, 'MoveCube_ep0_attempt_01', 'stop', 'Independent variant failed')
    found = states(out)
    assert found['MoveCube_ep0'] == 'stopped'
    assert all(found[f'MoveCube_ep{ep}'] == 'pending' for ep in range(1, 5))


def test_three_total_attempts_across_versions(out):
    for index in range(1, 4):
        row = finish(out, index=index, version='v001' if index == 2 else 'v000')
        if index < 3:
            queue.review(out, row['id'], 'retry', 'Trace diagnosis', 'v001')
            assert states(out)['A_ep0'] == 'retry'
    assert states(out)['A_ep0'] == 'stopped'
    with pytest.raises(ValueError, match='cap'):
        queue.review(out, row['id'], 'retry', 'Trace diagnosis', 'v001')
    assert states(out)['A_ep1'] == 'blocked'


def test_regression_review_and_duplicate_prevention(out):
    row = finish(out, success=True)
    queue.review(out, row['id'], 'accept', 'Official success verified')
    with pytest.raises(ValueError, match='regression'):
        queue.review(out, row['id'], 'retry', 'regression', 'v000')
    queue.review(out, row['id'], 'retry', 'regression after generic fix', 'v001')
    with pytest.raises(ValueError, match='already reviewed'):
        queue.review(out, row['id'], 'retry', 'regression after generic fix', 'v001')
    assert read(out / 'reviews' / (row['id'] + '.accepted.json'))['action'] == 'accept'
    finish(out, index=2, version='v001')
    assert states(out)['A_ep0'] == 'waiting_review'


def test_stale_pid_and_command_mismatch():
    proc = psutil.Process()
    assert identity_alive(proc.pid, proc.create_time(), proc.cmdline())
    assert not identity_alive(proc.pid, proc.create_time() - 1)
    assert not identity_alive(proc.pid, proc.create_time(), ['wrong'])
    assert not identity_alive(None, None)


def test_snapshot_import_isolation_and_integrity(out):
    snapshot = out / 'versions/v000/src'
    req = {'root': str(out), 'source_root': str(snapshot)}
    # cwd is the version directory, and absolute PYTHONPATH wins over live checkout.
    result = subprocess.run([sys.executable, '-c', 'import fds; print(fds.VALUE); print(fds.__file__)'],
                            cwd=snapshot.parent, env=worker_environment(req), capture_output=True, text=True, check=True)
    assert 'frozen' in result.stdout and str(snapshot) in result.stdout
    queue.verify_version(out, 'v000')
    (snapshot / 'fds/__init__.py').write_text('VALUE = "mutated"')
    with pytest.raises(ValueError, match='integrity'):
        queue.verify_version(out, 'v000')


def test_missing_result_is_durable_failed_unknown(out):
    row = finish(out)
    summary = Path(row['output']) / 'A/episode_0000/summary.json'
    summary.unlink()
    result = queue.build_state(out)['completed'][0]
    assert result['outcome'] == 'failed'
    assert result['official_status'] == 'unknown'
    assert result['failure_category'] == 'infrastructure'
    assert states(out)['A_ep0'] == 'waiting_review'


def test_token_digits_cannot_turn_task_failure_into_auth_failure(out):
    row = finish(out)
    (Path(row['output']).parent / 'worker.log').write_text('prompt_tokens=401429; remaining_steps=401')
    assert queue.build_state(out)['completed'][0]['failure_category'] == 'task'


def test_auth_and_network_error_classification(out):
    row = finish(out)
    folder = Path(row['output']).parent
    atomic(folder / 'status.json', {'status': 'failed'})
    summary = Path(row['output']) / 'A/episode_0000/summary.json'
    atomic(summary, {'success': False, 'official_status': 'unknown', 'stop_reason': 'error',
                     'error': 'Authentication failed: unauthorized'})
    assert queue.build_state(out)['completed'][0]['failure_category'] == 'auth'
    atomic(summary, {'success': False, 'official_status': 'unknown', 'stop_reason': 'error',
                     'error': 'Connection failed: error sending request'})
    assert queue.build_state(out)['completed'][0]['failure_category'] == 'network'


def test_invalid_candidate_and_scope(out):
    (out / 'versions/v001/validation.json').unlink()
    with pytest.raises(ValueError, match='validation'):
        queue.verify_version(out, 'v001')
    assert not queue.allowed_change('fds/task_harness/skills.py')
    assert not queue.allowed_change('fds/benchmarks/robomme_plus/adapter.py')
    assert queue.allowed_change('fds/task_harness/feedback.py')


def test_reserved_attempt_cannot_be_overwritten(out):
    row = finish(out)
    case = read(out / 'campaign.json')['cases'][0]
    case['past_attempts'] = []  # stale dispatcher state still cannot clobber disk.
    before = (Path(row['output']).parent / 'request.json').read_bytes()
    with pytest.raises(FileExistsError):
        queue.dispatch(out, case, 'v000')
    assert (Path(row['output']).parent / 'request.json').read_bytes() == before


def test_verified_orphan_halts_only_case(out):
    row = finish(out)
    child = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])
    try:
        proc = psutil.Process(child.pid)
        atomic(Path(row['output']).parent / 'status.json', dict(status='running',
               worker_pid=child.pid, worker_create_time=proc.create_time(), command=proc.cmdline()))
        assert states(out)['A_ep0'] == 'needs_diagnosis'
        assert states(out)['B_ep0'] == 'pending'
        # A reused/stale PID is never adopted.
        atomic(Path(row['output']).parent / 'status.json', dict(status='running',
               worker_pid=child.pid, worker_create_time=proc.create_time() - 5, command=proc.cmdline()))
        assert states(out)['A_ep0'] == 'waiting_review'
    finally:
        child.terminate()
        child.wait()


def test_dispatch_batch_limit_and_singleton(out, monkeypatch):
    (out / 'reviews').mkdir()
    initial = dict(status='running', active_runs=[], completed=[], cases=[
        dict(id=f'T{i}_ep0', task=f'T{i}', episode=0, state='pending', past_attempts=[]) for i in range(16)])
    finished = dict(status='completed', active_runs=[], completed=[], cases=[])
    snapshots = iter([initial, finished])
    monkeypatch.setattr(queue, 'build_state', lambda _: next(snapshots))
    launched = []
    class Child:
        returncode = 0
        def poll(self):
            return 0
    def dispatch(*args):
        launched.append(args[1]['id'])
        return Child()
    monkeypatch.setattr(queue, 'dispatch', dispatch)
    with lock(out / 'campaign.lock', nonblocking=True):
        with pytest.raises(BlockingIOError):
            queue.run(out)
    queue.run(out)
    assert len(launched) == 3


def test_atomic_file_events_wake_without_polling(tmp_path):
    events = FileEvents()
    try:
        events.watch(tmp_path)
        atomic(tmp_path / 'control.json', {'halt': True})
        with selectors.DefaultSelector() as selector:
            selector.register(events.fd, selectors.EVENT_READ)
            assert selector.select(timeout=1)
            events.drain()
            assert not selector.select(timeout=0)
    finally:
        events.close()


def test_local_initialization_failure_does_not_hold_other_cases(out):
    req = finish(out)
    folder = Path(req['output']).parent
    atomic(folder / 'status.json', {'status': 'failed'})
    atomic(Path(req['output']) / 'A/episode_0000/summary.json',
           {'success': False, 'official_status': 'unknown',
            'error': 'RuntimeError: recorded demonstration failed: missing_requirements'})
    result = queue.build_state(out)
    failed = result['completed'][0]
    assert failed['failure_category'] == 'infrastructure'
    assert not queue.requires_global_hold(failed)
    assert states(out)['A_ep0'] == 'waiting_review'
    assert states(out)['B_ep0'] == 'pending'
    control = {'hold_reason': queue.AUTO_HOLD_PREFIX + req['id'], 'halt': False}
    reconciled = queue.reconcile_global_hold(control, result['completed'])
    assert reconciled['hold_reason'] is None
    assert reconciled['global_failure_attempts'] == []
    assert failed['official_status'] == 'unknown'


@pytest.mark.parametrize('category', ['auth', 'authentication', 'authorization', 'quota', 'network', 'transport'])
def test_shared_outage_still_holds_all_admissions(category):
    row = {'id': 'A_ep0_attempt_01', 'failure_category': category, 'success': False}
    assert queue.requires_global_hold(row)
    control = queue.reconcile_global_hold({}, [row])
    assert control['global_failure_attempts'] == [row['id']]
    assert control['hold_reason'] == queue.AUTO_HOLD_PREFIX + row['id']
    control['cleared_infrastructure_attempts'] = [row['id']]
    assert queue.reconcile_global_hold(control, [row])['hold_reason'] is None


def test_manual_hold_preserved_and_explicit_shared_evidence_required():
    row = {'id': 'A_ep0_attempt_01', 'failure_category': 'infrastructure', 'failure_scope': 'global'}
    assert not queue.requires_global_hold(row)
    row['shared_failure_evidence'] = {'reason': 'Shared dependency missing', 'path': '/evidence/import.log'}
    assert queue.requires_global_hold(row)
    control = {'hold_reason': 'Operator maintenance', 'hold_until': 'manual-clear', 'halt': True}
    result = queue.reconcile_global_hold(control, [row])
    assert result['hold_reason'] == control['hold_reason']
    assert result['halt'] and result['hold_until'] == 'manual-clear'
    assert queue.reconcile_global_hold(result, [])['hold_reason'] == control['hold_reason']
    row['success'] = True
    assert not queue.requires_global_hold(row)


@pytest.mark.parametrize('case_id', sorted(queue.QUOTA_EXTENSION_CASES))
def test_named_quota_extension_fourth_only_with_user_evidence(out, monkeypatch, case_id):
    manifest = read(out / 'campaign.json')
    task, episode = case_id.split('_ep')
    manifest['cases'] = [dict(id=f'{task}_ep{e}', task=task, episode=e, difficulty=['easy', 'medium'][e], seed=7)
                         for e in range(int(episode) + 1)]
    if int(episode):
        finish(out, task + '_ep0', success=True)
    for index in range(1, 4):
        req = finish(out, case_id, index)
    summary = Path(req['output']) / task / f'episode_{int(episode):04d}/summary.json'
    atomic(summary, {'success': False, 'official_status': 'unknown', 'failure_category': 'quota'})
    atomic(out / 'campaign.json', manifest)
    assert next(c for c in queue.build_state(out)['cases'] if c['id'] == case_id)['attempt_limit'] == 3
    queue.review(out, req['id'], 'stop', 'Original quota interruption exhausted default allowance')
    evidence = out / 'user_authorization.json'
    atomic(evidence, {'source': 'user', 'case_id': case_id, 'additional_attempts': 1})
    manifest['attempt_limit_overrides'] = {case_id: {'attempt_limit': 4,
        'interrupted_attempt': req['id'], 'interruption_category': 'quota',
        'authorization': {'source': 'user', 'reason': 'Quota interruption replacement explicitly authorized', 'evidence': [str(evidence)]}}}
    atomic(out / 'campaign.json', manifest)
    queue.review(out, req['id'], 'retry', 'Authorized replacement after quota restored', 'v001')
    case = next(c for c in queue.build_state(out)['cases'] if c['id'] == case_id)
    assert case['state'] == 'retry' and case['attempt_limit'] == 4
    assert read(out / 'reviews' / (req['id'] + '.stop_before_authorized_extension.json'))['action'] == 'stop'
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *args, **kw: (_ for _ in ()).throw(OSError('synthetic launch error')))
    assert queue.dispatch(out, case, 'v001') is None
    fourth = read(out / 'attempts' / case_id / 'attempt_04/request.json')
    assert fourth['attempt_limit'] == 4 and fourth['attempt_limit_authorization']['authorization']['source'] == 'user'
    case = next(c for c in queue.build_state(out)['cases'] if c['id'] == case_id)
    with pytest.raises(ValueError, match='cap'):
        queue.dispatch(out, case, 'v001')
    with pytest.raises(ValueError, match='cap'):
        queue.review(out, fourth['id'], 'retry', 'Fifth attempt forbidden', 'v001')


def test_override_cannot_expand_other_cases_or_skip_evidence(out):
    manifest = {'max_attempts': 4, 'attempt_limit_overrides': {}}
    assert queue.attempt_limit(manifest, 'Other_ep0') == 3
    assert queue.attempt_limit(manifest, 'PatternLock_ep1') == 3
    override = {'attempt_limit': 4, 'interrupted_attempt': 'PatternLock_ep1_attempt_03',
                'interruption_category': 'quota', 'authorization': {'source': 'user', 'reason': 'authorized', 'evidence': []}}
    manifest['attempt_limit_overrides']['PatternLock_ep1'] = override
    assert queue.attempt_limit(manifest, 'PatternLock_ep1') == 3
    evidence = out / 'authorization.json'
    atomic(evidence, {'source': 'user'})
    override['authorization']['evidence'] = [str(evidence)]
    assert queue.attempt_limit(manifest, 'PatternLock_ep1') == 4
    override['attempt_limit'] = 5
    assert queue.attempt_limit(manifest, 'PatternLock_ep1') == 3
    override['attempt_limit'] = 4
    past = [{}, {}, {'id': 'PatternLock_ep1_attempt_03', 'state': 'finished', 'failure_category': 'task', 'success': False}]
    assert queue.attempt_limit(manifest, 'PatternLock_ep1', past) == 3
    override['interrupted_attempt'] = 'Other_ep0_attempt_03'
    manifest['attempt_limit_overrides']['Other_ep0'] = override
    assert queue.attempt_limit(manifest, 'Other_ep0') == 3
