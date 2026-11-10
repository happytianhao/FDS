"""Offline selection/scope checks for a long-running autonomous code reviewer."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments/adaptive_train'))
optimizer = importlib.import_module('optimizer')


def row(version, case, success, **kw):
    return dict(version=version, case_id=case, success=success, terminal=True,
                failure_category=None if success else 'task', **kw)


def test_best_requires_prior_success_coverage_and_no_regressions():
    rows = [row('v000', 'A', True), row('v000', 'B', False), row('v001', 'B', True)]
    assert optimizer.version_evidence({'rows': rows})['best_version'] == 'v000'
    rows.append(row('v001', 'A', False))
    assert optimizer.version_evidence({'rows': rows})['best_version'] == 'v000'
    rows.append(row('v001', 'A', True))
    assert optimizer.version_evidence({'rows': rows})['best_version'] == 'v001'


def test_untested_versions_and_mixed_coverage_never_prove_plateau():
    rows = [row('v000', 'A', True), row('v001', 'B', True), row('v002', 'C', True)]
    result = optimizer.version_evidence({'rows': rows})
    assert result['best_version'] == 'v000'
    assert result['unverified_success_cases'] == ['B', 'C']
    assert result['comparable_plateau_evidence'] is False


def test_equal_success_but_better_recovery_is_not_plateau():
    rows = [row('v000', 'A', False), row('v001', 'A', False), row('v002', 'A', False, recoveries=1)]
    assert not optimizer.version_evidence({'rows': rows})['comparable_plateau_evidence']


def test_new_task_specific_code_requires_confirmation(tmp_path):
    base, work = tmp_path / 'base', tmp_path / 'work'
    for root in (base, work):
        path = root / 'src/fds/task_harness/contracts.py'
        path.parent.mkdir(parents=True)
        path.write_text('VALUE = 1\n')
    (work / 'src/fds/task_harness/contracts.py').write_text('VALUE = "InsertPeg"\n')
    with pytest.raises(ValueError, match='Task-name-specific'):
        optimizer.validate_candidate_scope(base, work, ['InsertPeg'])
    (work / 'src/fds/task_harness/contracts.py').write_text('VALUE = 2\n')
    assert optimizer.validate_candidate_scope(base, work, ['InsertPeg']) == ['fds/task_harness/contracts.py']


def test_config_edit_cannot_hide_behind_passing_tests(tmp_path):
    for name in ('scripts', 'configs', 'docs'):
        (tmp_path / name).mkdir()
    for name in ('pyproject.toml', 'uv.lock'):
        (tmp_path / name).write_text('original')
    before = optimizer.ancillary_hashes(tmp_path)
    (tmp_path / 'configs/task.toml').write_text('skills = ["new-skill"]')
    assert optimizer.ancillary_hashes(tmp_path) != before


def failed_case(attempt=1, category='task'):
    return {'id': 'A_ep0', 'state': 'waiting_review', 'past_attempts': [
        {'id': f'A_ep0_attempt_{n:02d}', 'state': 'finished', 'success': False,
         'version': 'v000', 'failure_category': category} for n in range(1, attempt + 1)]}


def test_task_retry_requires_improvement_and_cannot_stop_early(tmp_path):
    from common import atomic
    atomic(tmp_path / 'versions/v000/version.json', {'source_sha256': {'a': 'old'}})
    atomic(tmp_path / 'versions/v001/version.json', {'source_sha256': {'a': 'new'}})
    case = failed_case()
    focus = case['past_attempts'][-1]['id']
    proposal = {'decisions': [{'attempt_id': focus, 'action': 'retry'}], 'clear_infrastructure_hold': False}
    with pytest.raises(ValueError, match='improved source'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], focus)
    optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v001', [], focus)
    optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', ['a'], focus, {'a': 'new'})
    proposal['decisions'][0]['action'] = 'stop'
    with pytest.raises(ValueError, match='remaining attempts'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v001', ['a'], focus)


def test_third_failed_attempt_analyzed_but_never_retried(tmp_path):
    case = failed_case(3)
    focus = case['past_attempts'][-1]['id']
    proposal = {'decisions': [{'attempt_id': focus, 'action': 'retry'}]}
    with pytest.raises(ValueError, match='no fourth'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v001', ['a'], focus)
    proposal['decisions'][0]['action'] = 'stop'
    optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], focus)


def test_only_verified_transport_recovery_may_replay_without_source_change(tmp_path):
    case = failed_case(category='network')
    focus = case['past_attempts'][-1]['id']
    proposal = {'decisions': [{'attempt_id': focus, 'action': 'retry'}], 'clear_infrastructure_hold': False}
    with pytest.raises(ValueError, match='improved source'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], focus)
    proposal['clear_infrastructure_hold'] = True
    optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], focus)
    case['past_attempts'][-1]['failure_category'] = 'quota'
    with pytest.raises(ValueError, match='Account recovery'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v001', ['a'], focus)


def test_failure_notifications_are_deduplicated_and_bad_review_is_bounded(tmp_path, monkeypatch):
    case = failed_case(3)
    focus = case['past_attempts'][-1]['id']
    state = {'cases': [case]}
    monkeypatch.setattr(optimizer.time, 'time', lambda: 100)
    assert [a['id'] for a in optimizer.pending_failures(tmp_path, state, {}, 100)] == [focus]
    outcome = {'status': 'candidate_rejected'}
    for _ in range(3):
        optimizer.record_failure_review(tmp_path, focus, outcome)
    ledger = optimizer.read(tmp_path / 'optimizer/failure-reviews.json')
    assert ledger[focus]['analysis_failures'] == 3
    assert not optimizer.pending_failures(tmp_path, state, ledger, 100000)
    assert (tmp_path / 'NEEDS_USER_DECISION.json').exists()


def test_each_pending_failure_triggers_immediately_without_periodic_wait(tmp_path, monkeypatch):
    from common import atomic
    first, second = failed_case(), failed_case(2)
    second['id'] = 'B_ep0'
    for a in second['past_attempts']:
        a['id'] = a['id'].replace('A_', 'B_')
    atomic(tmp_path / 'control.json', {'halt': False})
    atomic(tmp_path / 'status.json', {'cases': [first, second]})
    (tmp_path / 'reviews').mkdir()
    seen = []

    def checkpoint(out, focus):
        seen.append(focus)
        return {'status': 'completed' if len(seen) == 2 else 'reviewed',
                'decisions': [{'attempt_id': focus, 'applied': True}]}

    def no_wait(*args):
        pytest.fail('A pending failure must not wait for the 30-minute timer')

    monkeypatch.setattr(optimizer, 'checkpoint', checkpoint)
    monkeypatch.setattr(optimizer.select, 'select', no_wait)
    optimizer.serve(tmp_path)
    assert seen == ['A_ep0_attempt_01', 'B_ep0_attempt_02']


def test_health_review_cannot_override_failed_case_hold(tmp_path):
    case = failed_case()
    proposal = {'decisions': [{'attempt_id': case['past_attempts'][-1]['id'], 'action': 'retry'}]}
    with pytest.raises(ValueError, match='focused diagnosis'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v001', ['a'], None, {'a': 'new'})


def test_version_cycle_is_not_a_new_improvement(tmp_path):
    from common import atomic
    atomic(tmp_path / 'versions/v000/version.json', {'source_sha256': {'a': 'first'}})
    atomic(tmp_path / 'versions/v001/version.json', {'source_sha256': {'a': 'second'}})
    case = failed_case(2)
    case['past_attempts'][1]['version'] = 'v001'
    focus = case['past_attempts'][-1]['id']
    proposal = {'decisions': [{'attempt_id': focus, 'action': 'retry'}]}
    with pytest.raises(ValueError, match='not previously tried'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], focus)
    with pytest.raises(ValueError, match='not previously tried'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v002', ['a'], focus, {'a': 'first'})
    optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v002', ['a'], focus, {'a': 'third'})


@pytest.fixture
def routing_fixture(tmp_path, monkeypatch):
    from common import atomic, hashes, read
    out, work, directory = tmp_path / 'out', tmp_path / 'work', tmp_path / 'review'
    (work / 'src/fds/task_harness').mkdir(parents=True)
    (work / 'src/fds/task_harness/contracts.py').write_text('VALUE = 2\n')
    directory.mkdir()
    atomic(directory / 'ancillary-hashes.json', {})
    atomic(out / 'campaign.json', {'tasks': ['A', 'B']})
    atomic(out / 'control.json', {'active_version': 'v002', 'halt': False})
    for version in ('v000', 'v001', 'v002'):
        atomic(out / 'versions' / version / 'version.json',
               {'source_sha256': {'a': version}, 'lineage': ['v000', version] if version != 'v000' else ['v000']})
    case = failed_case()
    success = {'id': 'B_ep0', 'state': 'success', 'past_attempts': [
        {'id': 'B_ep0_attempt_01', 'state': 'finished', 'success': True, 'version': 'v000'}]}
    state = {'cases': [case, success], 'completed': [], 'active_runs': []}
    proposal = dict(change_kind='none', rationale='Reviewed evidence', recommended_version='v001',
                    decisions=[dict(attempt_id='A_ep0_attempt_01', action='retry', reason='focused repair')],
                    regression_cases=['B_ep0'], evidence=[], clear_infrastructure_hold=False,
                    finish_campaign=False)
    monkeypatch.setattr(optimizer, 'ancillary_hashes', lambda _: {})
    monkeypatch.setattr(optimizer, 'validate_candidate_scope', lambda *args: [])
    monkeypatch.setattr(optimizer.campaign, 'build_state', lambda _: state)
    monkeypatch.setattr(optimizer.campaign, 'verify_version', lambda out, v: read(out / 'versions' / v / 'version.json'))
    reviews = []
    monkeypatch.setattr(optimizer.campaign, 'review', lambda *args: reviews.append(args))
    return out, work, directory, state, proposal, reviews, hashes


def test_case_fallback_routes_retry_without_global_rollback_or_regression(routing_fixture):
    out, work, directory, state, proposal, reviews, _ = routing_fixture
    result = optimizer.apply_review(out, out, directory, work, proposal, 'v002', 'A_ep0_attempt_01')
    assert optimizer.read(out / 'control.json')['active_version'] == 'v002'
    assert len(reviews) == 1 and reviews[0][1] == 'A_ep0_attempt_01' and reviews[0][4] == 'v001'
    assert result['recommendation']['scope'] == 'focused_case_only'
    assert result['skipped_regression_cases'] == ['B_ep0']


def test_health_recommendation_and_explicit_success_retry_cannot_roll_back(routing_fixture):
    out, work, directory, state, proposal, reviews, _ = routing_fixture
    proposal['decisions'] = [dict(attempt_id='B_ep0_attempt_01', action='retry', reason='regression')]
    result = optimizer.apply_review(out, out, directory, work, proposal, 'v002')
    assert optimizer.read(out / 'control.json')['active_version'] == 'v002'
    assert not reviews
    assert result['decisions'][0]['applied'] is False
    assert result['recommendation']['scope'] == 'advisory_only'


@pytest.mark.parametrize('active_changed', [False, True])
def test_candidate_publication_records_parent_and_rejects_stale_base(routing_fixture, monkeypatch, active_changed):
    from types import SimpleNamespace

    from common import atomic
    out, work, directory, state, proposal, reviews, hashes = routing_fixture
    proposal.update(change_kind='routine', recommended_version='')
    monkeypatch.setattr(optimizer, 'validate_candidate_scope', lambda *args: ['fds/task_harness/contracts.py'])
    def validated(*args, **kwargs):
        if active_changed:
            atomic(out / 'control.json', {'active_version': 'v001'})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(optimizer.subprocess, 'run', validated)
    def snapshot(out, reason, source):
        atomic(out / 'versions/v003/version.json', {'source_sha256': hashes(work / 'src')})
        return 'v003'
    monkeypatch.setattr(optimizer.campaign, 'snapshot', snapshot)
    if active_changed:
        with pytest.raises(ValueError, match='rebased'):
            optimizer.apply_review(out, out, directory, work, proposal, 'v002', 'A_ep0_attempt_01')
        assert not (out / 'versions/v003').exists()
        assert optimizer.read(out / 'control.json')['active_version'] == 'v001'
    else:
        optimizer.apply_review(out, out, directory, work, proposal, 'v002', 'A_ep0_attempt_01')
        assert optimizer.read(out / 'control.json')['active_version'] == 'v003'
        meta = optimizer.read(out / 'versions/v003/version.json')
        validation = optimizer.read(out / 'versions/v003/validation.json')
        assert meta['parent_version'] == validation['parent_version'] == 'v002'
        assert meta['lineage'] == ['v000', 'v002', 'v003']
        assert len(reviews) == 2  # Focused retry plus candidate-specific regression.


@pytest.mark.parametrize('case_id', sorted(optimizer.campaign.QUOTA_EXTENSION_CASES))
def test_optimizer_authorized_quota_fourth_requires_recovery_and_never_fifth(tmp_path, case_id):
    from common import atomic
    evidence = tmp_path / 'authorization.json'
    atomic(evidence, {'source': 'user', 'case_id': case_id})
    interrupted = case_id + '_attempt_03'
    atomic(tmp_path / 'campaign.json', {'attempt_limit_overrides': {case_id: {
        'attempt_limit': 4, 'interrupted_attempt': interrupted, 'interruption_category': 'quota',
        'authorization': {'source': 'user', 'reason': 'One replacement for quota interruption', 'evidence': [str(evidence)]}}}})
    atomic(tmp_path / 'versions/v000/version.json', {'source_sha256': {'a': 'same'}})
    case = {'id': case_id, 'past_attempts': [dict(id=f'{case_id}_attempt_{i:02d}', state='finished',
            version='v000', success=False, failure_category='quota') for i in range(1, 4)]}
    proposal = {'decisions': [{'attempt_id': interrupted, 'action': 'retry'}]}
    with pytest.raises(ValueError, match='Account recovery'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], interrupted)
    atomic(tmp_path / 'control.json', {'cleared_infrastructure_attempts': [interrupted]})
    optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], interrupted)
    case['past_attempts'].append(dict(id=case_id + '_attempt_04', state='finished', version='v000', success=False, failure_category='task'))
    proposal['decisions'][0]['attempt_id'] = case_id + '_attempt_04'
    with pytest.raises(ValueError, match='cap 4'):
        optimizer.validate_failure_decisions(tmp_path, {'cases': [case]}, proposal, 'v000', [], case_id + '_attempt_04')
