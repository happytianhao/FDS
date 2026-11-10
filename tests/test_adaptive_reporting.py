"""Synthetic reporting fixtures never invoke scenes, providers or the optimizer."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments/adaptive_train'))
import reporting
from common import atomic


def fixture(tmp_path):
    cases, completed, active = [], [], []
    specs = [('Success', 'success', True, 'success', None),
             ('TaskFail', 'stopped', False, 'fail', 'task'),
             ('Infra', 'stopped', False, 'unknown', 'auth'),
             ('Unknown', 'stopped', False, 'unknown', None),
             ('Orphan', 'needs_diagnosis', False, 'unknown', None),
             ('Reserved', 'pending', False, 'unknown', None),
             ('Blocked', 'blocked', False, 'unknown', None)]
    for task, state, success, official, category in specs:
        case = dict(id=f'{task}_ep0', task=task, episode=0, difficulty='easy', state=state, past_attempts=[])
        if task != 'Blocked':
            folder = tmp_path / 'attempts' / case['id'] / 'attempt_01'
            row = dict(id=case['id'] + '_attempt_01', case_id=case['id'], task=task, episode=0,
                       difficulty='easy', attempt=1, version='v000', output=str(folder / 'output'),
                       success=success, official_status=official, failure_category=category,
                       state=state if task in ('Orphan', 'Reserved') else 'finished')
            atomic(folder / 'request.json', row)
            case['past_attempts'] = [row]
            if task not in ('Orphan', 'Reserved'):
                completed.append(row)
                atomic(folder / 'output' / task / 'episode_0000/summary.json',
                       dict(success=success, official_status=official, steps=10, planner_calls=1))
                atomic(folder / 'output/metrics.json', {'status': 'completed'})
                atomic(folder / 'output' / task / 'episode_0000/call_0001/stats.json',
                       {'usage': {'input_tokens': 10, 'output_tokens': 2}})
        cases.append(case)
    atomic(tmp_path / 'campaign.json', {'root': str(tmp_path), 'cases': cases})
    atomic(tmp_path / 'status.json', {'status': 'needs_diagnosis', 'cases': cases,
                                    'completed': completed, 'active_runs': active})
    return tmp_path


def test_task_infrastructure_unknown_and_nonterminal_are_distinct(tmp_path):
    out = fixture(tmp_path)
    data = reporting.widget_data(out)
    exp = data['experiment']
    assert exp['task_failures'] == 1
    assert exp['errors'] == 1
    assert exp['unknown_failures'] == 1
    assert exp['blocked'] == 1 and exp['unattempted'] == 1
    assert exp['unresolved'] == 2
    items = {r['task'].split(' · ')[0]: r for r in exp['tasks']}
    assert items['Orphan']['state'] == 'needs_diagnosis'
    assert items['Reserved']['state'] == 'reserved'
    assert items['Blocked']['state'] == 'blocked'
    assert not items['Orphan']['terminal']
    totals = reporting.collect(out)['totals']
    assert totals['failures'] == 3 and totals['task_failures'] == 1


def test_historical_and_optimizer_tokens_do_not_mix_with_scenes(tmp_path):
    out = fixture(tmp_path)
    assert reporting.widget_data(out)['usage']['previous_run_tokens'] is None
    atomic(out / 'runs/retrospective_20261109/results.json',
           {'generated_utc': 'sample-time', 'totals': {'total_tokens': 12345}})
    atomic(out / 'optimizer/review1/usage.json',
           dict(input_tokens=100, output_tokens=20, completed_turns=1, usage_missing=False))
    atomic(out / 'optimizer/review2/request.json', {'model': 'synthetic'})
    data = reporting.widget_data(out)
    assert data['usage']['previous_run_tokens'] == 12345
    assert data['usage']['total_tokens'] == 48  # Four scenes only.
    assert data['optimizer_usage']['total_tokens'] == 120
    assert data['optimizer_usage']['missing_usage_runs'] == 1
    assert data['optimizer_usage']['usage_missing']
    assert data['usage']['historical_usage']['generated_utc'] == 'sample-time'


def test_final_candidate_unknown_coverage_and_csv_taxonomy(tmp_path):
    out = fixture(tmp_path)
    atomic(out / 'FINAL.json', {'status': 'budget_limited', 'recommended_version': 'v001',
                              'stop_basis': 'budget_limited'})
    result = reporting.write_report(out)
    assert result['status'] == 'budget_limited'
    assert result['queue_status'] == 'needs_diagnosis'
    assert len(result['final_validation']['unknown_cases']) == 7
    assert not result['final_validation']['fully_verified']
    assert result['final_validation']['successful_cases'] == []  # v000 success is not v001 evidence.
    with (out / 'results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    infra = next(r for r in rows if r['task'] == 'Infra')
    assert infra['failure_category'] == 'auth' and infra['failure_kind'] == 'infrastructure'
    atomic(out / 'FINAL.json', {'status': 'completed', 'recommended_version': 'v000'})
    verification = reporting.collect(out)['final_validation']
    assert verification['successful_cases'] == ['Success_ep0']
    assert verification['failed_cases'] == ['TaskFail_ep0']
    assert 'Infra_ep0' in verification['unknown_cases']


def test_partial_success_summary_is_not_completed_success(tmp_path):
    out = fixture(tmp_path)
    atomic(out / 'attempts/Reserved_ep0/attempt_01/output/Reserved/episode_0000/summary.json',
           {'success': True, 'official_status': 'success'})
    row = next(r for r in reporting.collect(out)['rows'] if r['task'] == 'Reserved')
    assert row['outcome'] == 'reserved'
    assert not row['success'] and not row['terminal']


def test_outcome_detail_preserves_official_unknown_and_verified_failure_definition(tmp_path):
    out = fixture(tmp_path)
    root = out / 'attempts/Unknown_ep0/attempt_01/output/Unknown/episode_0000'
    for stop, detail in [('planner_finish', 'planner_finish_unverified'),
                         ('budget_exhausted', 'budget_exhausted'),
                         ('decision_recovery_exhausted', 'decision_recovery_exhausted')]:
        atomic(root / 'summary.json', {'success': False, 'official_status': 'ongoing',
                                      'stop_reason': stop, 'planner_calls': 5, 'steps': 0})
        result = reporting.write_report(out)
        row = next(r for r in result['rows'] if r['task'] == 'Unknown')
        assert row['outcome_detail'] == detail
        assert row['official_status'] == 'ongoing' and row['failure_kind'] == 'unknown'
        assert result['totals']['task_failures'] == 1
        assert result['totals']['outcome_details'][detail] == 1
        assert row['outcome_detail_label'] in (out / 'REPORT.md').read_text()
        with (out / 'results.csv').open() as stream:
            exported = next(r for r in csv.DictReader(stream) if r['task'] == 'Unknown')
        assert exported['outcome_detail'] == detail and exported['official_status'] == 'ongoing'
        widget = reporting.widget_data(out)
        assert widget['experiment']['outcome_details'][detail] == 1
        shown = next(r for r in widget['experiment']['tasks'] if r.get('case_id') == 'Unknown_ep0')
        assert shown['outcome_detail'] == detail


def test_failure_detail_separates_quota_initialization_and_verified_failure():
    base = dict(terminal=True, success=False, official_status='unknown', stop_reason='error', planner_calls=0)
    assert reporting.outcome_detail({**base, 'failure_category': 'quota'}) == 'quota'
    assert reporting.outcome_detail({**base, 'failure_category': 'auth'}) == 'authentication'
    assert reporting.outcome_detail({**base, 'failure_category': 'infrastructure'}) == 'initialization_error'
    assert reporting.outcome_detail({**base, 'planner_calls': 8, 'failure_category': 'transport'}) == 'request_error'
    assert reporting.outcome_detail({**base, 'planner_calls': 8}) == 'execution_error'
    for official, detail in [('fail', 'task_failure'), ('timeout', 'task_timeout')]:
        row = {**base, 'official_status': official, 'failure_category': 'task', 'stop_reason': 'environment_done'}
        assert reporting.outcome_detail(row) == detail
        assert reporting.failure_kind(row) == 'task'
    unfinished = {**base, 'terminal': False, 'outcome': 'running', 'success': True, 'official_status': 'success'}
    assert reporting.outcome_detail(unfinished) == 'running'
