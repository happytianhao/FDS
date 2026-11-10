"""Offline, version-specific metrics and a widget data view; no model calls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import atomic, identity_alive, lock, now, read


def safe_read(path):
    try:
        return read(path)
    except (OSError, ValueError):
        return {}


def measure(request, terminal=False):
    folder = Path(request['output'])
    ep = folder / request['task'] / f"episode_{request['episode']:04d}"
    summary = safe_read(ep / 'summary.json')
    meta = safe_read(ep / 'benchmark_metadata.json')
    calls = sorted(ep.glob('call_*'))
    attempts = [attempt for call in calls for attempt in (sorted(call.glob('attempt_*')) or [call])]
    counters = Counter()
    settings = Counter()
    for attempt in attempts:
        stats = safe_read(attempt / 'stats.json')
        usage = stats.get('usage') or {}
        inp, output = usage.get('input_tokens', usage.get('prompt_tokens')), usage.get('output_tokens', usage.get('completion_tokens'))
        if inp is None or output is None:
            counters['missing_usage' if terminal else 'pending_usage'] += 1
        counters['prompt_tokens'] += inp or 0
        counters['completion_tokens'] += output or 0
        # Requested settings are separate from independently recorded successful responses.
        settings[tuple(stats.get(k) for k in ('model', 'effort', 'sandbox_mode', 'approval_policy'))] += 1
    accepted = list((ep / 'audit').glob('call_*/fds_contract.json'))
    rejected = list((ep / 'audit').glob('call_*/decision_rejection.json'))
    recovered = list((ep / 'audit').glob('call_*/decision_recovery.json'))
    verdicts = Counter(safe_read(p).get('status', 'unknown') for p in (ep / 'audit').glob('call_*/fds_verdict.json'))
    steps = summary.get('steps')
    if steps is None:
        latest = ep / 'audit' / calls[-1].name if calls else ep / 'audit/nonexistent'
        steps = max(safe_read(latest / n).get('step', 0) for n in ('before.json', 'after.json'))
    return {
        **request, 'terminal': terminal, 'success': bool(summary.get('success')),
        'official_status': summary.get('official_status', 'unknown'),
        'stop_reason': summary.get('stop_reason'), 'planner_calls': summary.get('planner_calls', len(calls)),
        'request_attempts': len(attempts), 'transport_retries': max(0, len(attempts) - len(calls)),
        'environment_steps': steps, 'prompt_tokens': counters['prompt_tokens'],
        'completion_tokens': counters['completion_tokens'],
        'total_tokens': counters['prompt_tokens'] + counters['completion_tokens'],
        'missing_usage': counters['missing_usage'], 'pending_usage': counters['pending_usage'],
        'contracts_accepted': len(accepted), 'decision_rejections': len(rejected),
        'recoveries': len(recovered), 'contradicted': verdicts['contradicted'], 'uncertain': verdicts['uncertain'],
        'error': summary.get('error') or meta.get('result', {}).get('failure_reason'),
        'failure_category': request.get('failure_category'),
        'artifact_complete': bool(summary) and safe_read(folder / 'metrics.json').get('status') == 'completed',
        'requested_settings_note': 'Requested configuration does not substitute for a provider response/usage receipt.',
        'request_settings': [{'values': list(k), 'count': v} for k, v in settings.items()],
        'summary_path': str(ep / 'summary.json'), 'audit_path': str(ep / 'audit'),
    }


SUM_FIELDS = ('planner_calls', 'request_attempts', 'transport_retries', 'prompt_tokens',
              'completion_tokens', 'total_tokens', 'environment_steps', 'missing_usage',
              'pending_usage', 'contracts_accepted', 'decision_rejections', 'recoveries',
              'contradicted', 'uncertain')


def failure_kind(row):
    """Unknown official results are never counted as benchmark task failures."""
    if not row.get('terminal') or row.get('success'):
        return None
    category = row.get('failure_category')
    if category and category != 'task':
        return 'infrastructure'
    if row.get('official_status') in ('fail', 'timeout'):
        return 'task'
    return 'unknown'



OUTCOME_DETAIL_LABELS = {
    'success': '官方成功', 'task_failure': '官方任务失败', 'task_timeout': '官方超时',
    'planner_finish_unverified': '模型结束，未确认完成',
    'budget_exhausted': '预算耗尽，未确认完成',
    'decision_recovery_exhausted': '协议拒绝纠错耗尽',
    'quota': '额度阻塞', 'authentication': '鉴权/权限阻塞',
    'initialization_error': '初始化失败', 'request_error': '请求失败',
    'execution_error': '执行异常', 'infrastructure_error': '基础设施异常',
    'unknown': '结果未知', 'running': '运行中', 'reserved': '已预留',
    'needs_diagnosis': '待诊断', 'pending': '未开始', 'blocked': '前置条件阻塞',
}


def outcome_detail(row):
    """Explain termination without changing official results or verified-failure counts."""
    if not row.get('terminal'):
        return row.get('outcome') or 'pending'
    if row.get('success'):
        return 'success'
    category, stop = row.get('failure_category'), row.get('stop_reason')
    if category == 'quota':
        return 'quota'
    if category in ('auth', 'authentication', 'authorization', 'permission'):
        return 'authentication'
    if category in ('network', 'transport', 'timeout', 'rate_limit', 'service_unavailable'):
        return 'request_error'
    if stop == 'error' and row.get('planner_calls') == 0:
        return 'initialization_error'
    if stop == 'decision_recovery_exhausted':
        return 'decision_recovery_exhausted'
    if failure_kind(row) == 'task':
        return 'task_timeout' if row.get('official_status') == 'timeout' else 'task_failure'
    if stop == 'planner_finish':
        return 'planner_finish_unverified'
    if stop == 'budget_exhausted':
        return 'budget_exhausted'
    if stop == 'error':
        return 'execution_error'
    if failure_kind(row) == 'infrastructure':
        return 'infrastructure_error'
    return 'unknown'

def separate_usage(out, manifest):
    history_path = Path(manifest.get('history_results') or
                        Path(manifest['root']) / 'runs/retrospective_20261109/results.json')
    history = safe_read(history_path)
    historical = {'total_tokens': history.get('totals', {}).get('total_tokens'),
                  'generated_utc': history.get('generated_utc'), 'source': str(history_path),
                  'available': bool(history)}
    optimizer = dict(input_tokens=0, output_tokens=0, total_tokens=0, completed_turns=0,
                     usage_missing=False, missing_usage_runs=0, recorded_runs=0)
    directories = [p for p in (out / 'optimizer').glob('*') if p.is_dir() and
                   ((p / 'request.json').exists() or (p / 'usage.json').exists())]
    for directory in directories:
        usage = safe_read(directory / 'usage.json')
        missing = not usage or usage.get('usage_missing', False)
        optimizer['missing_usage_runs'] += bool(missing)
        optimizer['usage_missing'] |= bool(missing)
        optimizer['recorded_runs'] += bool(usage)
        for key in ('input_tokens', 'output_tokens', 'completed_turns'):
            optimizer[key] += usage.get(key, 0) or 0
    optimizer['total_tokens'] = optimizer['input_tokens'] + optimizer['output_tokens']
    return historical, optimizer


def collect(out):
    manifest, state = read(out / 'campaign.json'), safe_read(out / 'status.json')
    finished = {r['id']: r for r in state.get('completed', [])}
    active = {r['id']: r for r in state.get('active_runs', [])}
    known = {a['id']: a for c in state.get('cases', []) for a in c.get('past_attempts', [])}
    rows = []
    for path in sorted((out / 'attempts').glob('*/attempt_*/request.json')):
        request = safe_read(path)
        if not request:
            continue
        row = measure({**request, **known.get(request['id'], {}), **finished.get(request['id'], {}), **active.get(request['id'], {})}, request['id'] in finished)
        if request['id'] in finished:
            row['success'] = bool(finished[request['id']].get('success'))
            row['outcome'] = 'success' if row['success'] else 'failed'
            if not row['error'] and finished[request['id']].get('errors'):
                row['error'] = str(finished[request['id']]['errors'])
        else:
            row['success'] = False  # A partial summary is not a completed official result.
            row['outcome'] = ('running' if request['id'] in active else
                              'needs_diagnosis' if known.get(request['id'], {}).get('state') == 'needs_diagnosis' else 'reserved')
        row['failure_kind'] = failure_kind(row)
        row['outcome_detail'] = outcome_detail(row)
        row['outcome_detail_label'] = OUTCOME_DETAIL_LABELS.get(row['outcome_detail'], row['outcome_detail'])
        rows.append(row)
    totals = {k: sum(r[k] for r in rows) for k in SUM_FIELDS}
    totals.update(finished=sum(r['terminal'] for r in rows), successes=sum(r['terminal'] and r['success'] for r in rows),
                  failures=sum(r['terminal'] and not r['success'] for r in rows),
                  task_failures=sum(r['failure_kind'] == 'task' for r in rows),
                  infrastructure_failures=sum(r['failure_kind'] == 'infrastructure' for r in rows),
                  unknown_failures=sum(r['failure_kind'] == 'unknown' for r in rows),
                  outcome_details=dict(Counter(r['outcome_detail'] for r in rows if r['terminal'])))
    versions = {}
    for version in sorted({r['version'] for r in rows}):
        selected = [r for r in rows if r['version'] == version and r['terminal']]
        versions[version] = {k: sum(r[k] for r in selected) for k in SUM_FIELDS}
        versions[version].update(attempts=len(selected), successes=sum(r['success'] for r in selected),
                                 evaluated_cases=sorted({r['case_id'] for r in selected}),
                                 successful_cases=sorted({r['case_id'] for r in selected if r['success']}))
    final = safe_read(out / 'FINAL.json')
    final_version = final.get('recommended_version') or final.get('version')
    final_rows = {r['case_id']: r for r in rows if r['version'] == final_version and r['terminal']}
    all_cases = {c['id'] for c in manifest['cases']}
    verified = {key for key, row in final_rows.items() if row['success'] or row['failure_kind'] == 'task'}
    final_validation = {'version': final_version, 'evaluated_cases': sorted(final_rows),
                        'successful_cases': sorted(k for k, r in final_rows.items() if r['success']),
                        'failed_cases': sorted(k for k, r in final_rows.items() if r['failure_kind'] == 'task'),
                        'unknown_cases': sorted(all_cases - verified),
                        'fully_verified': bool(final_version) and verified == all_cases}
    historical, optimizer_usage = separate_usage(out, manifest)
    return {'generated_utc': now(), 'campaign': str(out), 'status': final.get('status') or state.get('status', 'initializing'),
            'queue_status': state.get('status', 'initializing'), 'final': final or None,
            'final_validation': final_validation, 'historical_usage': historical, 'optimizer_usage': optimizer_usage,
            'cases': state.get('cases', []), 'rows': rows, 'totals': totals, 'versions': versions,
            'case_count': len(manifest['cases']),
            'notes': ['Training only; cross-version coverage is not a single-version success rate.',
                      'Every physical run counts: default cap 3; only the three user-authorized quota cases permit one fourth run.',
                      'Missing usage is unknown; known token sums are lower bounds.',
                      'Optimization-agent usage is recorded separately under optimizer/.']}


def write_report(out):
    with lock(out / 'report.lock'):
        return _write_report(out)


def _write_report(out):
    result = collect(out)
    atomic(out / 'results.json', result)
    columns = ['id', 'task', 'episode', 'difficulty', 'attempt', 'version', 'outcome', 'official_status',
               'stop_reason', 'failure_category', 'failure_kind', 'outcome_detail', 'outcome_detail_label', *SUM_FIELDS, 'error', 'output']
    with (out / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['rows'])
    total = result['totals']
    lines = ['# E4 training optimization', '', f"Updated: {result['generated_utc']}", '',
             f"State: {result['status']}. Finished attempts: {total['finished']}; successes: {total['successes']}; failures: {total['failures']}.",
             '', 'All runs are RoboMME-Plus TRAIN. MoveCube ep0–4 are independent, unlabelled variants.',
             'Version rows have different case sets: do not compare raw percentages as a paired experiment.', '',
             '| Case | Attempt | Version | Outcome | Detail | Official | Calls | Tokens | Steps |',
             '|---|---:|---|---|---|---|---:|---:|---:|']
    for r in result['rows']:
        lines.append(f"| [{r['case_id']}]({r['output']}) | {r['attempt']} | {r['version']} | {r['outcome']} | {r['outcome_detail_label']} | {r['official_status']} | {r['planner_calls']} | {r['total_tokens']:,} | {r['environment_steps']} |")
    lines.extend(['', f"Known input tokens: {total['prompt_tokens']:,}; output: {total['completion_tokens']:,}; missing usage: {total['missing_usage']}.",
                  '', f"Cost = {total['prompt_tokens'] / 1e6:.6f} × input_price + {total['completion_tokens'] / 1e6:.6f} × output_price (prices per million tokens).", '',
                  '## Failure evidence', ''])
    for r in result['rows']:
        if r['terminal'] and not r['success']:
            lines.append(f"- {r['id']} ({r['version']}): {r['outcome_detail_label']} ({r['failure_category'] or 'unclassified'}); {r['error'] or r['stop_reason']}. [Audit]({r['audit_path']})")
    lines.extend(['', '## Limits', '', *['- ' + n for n in result['notes']], ''])
    if result['final']:
        verification = result['final_validation']
        lines.extend(['## Final candidate', '', f"Version: {verification['version']}; stop basis: {result['final'].get('stop_basis', 'unspecified')}.",
                      f"Verified successes: {len(verification['successful_cases'])}; verified task failures: {len(verification['failed_cases'])}; unknown/unverified cases: {len(verification['unknown_cases'])}.",
                      'Unknown/unverified: ' + (', '.join(verification['unknown_cases']) or 'none'), ''])
    opt = result['optimizer_usage']
    lines.extend([f"Optimizer known tokens (separate from scenes): {opt['total_tokens']:,}; runs with missing usage: {opt['missing_usage_runs']}.", ''])
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return result


def widget_data(out):
    result = collect(out)
    manifest, state = read(out / 'campaign.json'), safe_read(out / 'status.json')
    case_states = {r['id']: r for r in state.get('cases', [])}
    tasks = []
    for case in manifest['cases']:
        history = [r for r in result['rows'] if r['case_id'] == case['id']]
        item = history[-1] if history else {}
        case_state = case_states.get(case['id'], {}).get('state', 'pending')
        attempt_limit = case_states.get(case['id'], {}).get('attempt_limit', manifest.get('max_attempts', 3))
        label = f"{case['task']} · {case['difficulty'] or '独立'} ep{case['episode']}"
        if history:
            label += f" · {item['attempt']}/{attempt_limit} · {item['version']}"
        row = {**item, 'task': label, 'case_state': case_state,
               'attempt_limit': attempt_limit,
               'state': 'completed' if item.get('terminal') else item.get('outcome', case_state),
               'rejections': item.get('decision_rejections', 0), 'success': item.get('success', False),
               'stop_reason': item.get('stop_reason')}
        row.setdefault('outcome_detail', case_state)
        row.setdefault('outcome_detail_label', OUTCOME_DETAIL_LABELS.get(row['outcome_detail'], row['outcome_detail']))
        tasks.append(row)
    finished_cases = [r for r in tasks if r['case_state'] in ('success', 'stopped', 'blocked')]
    active = [r for r in tasks if r['state'] == 'running']
    quota = safe_read(out / 'quota-snapshot.json')
    optimizer = safe_read(out / 'optimizer/status.json')
    total = result['totals']
    return {'updated_at': now(), 'experiment': {
        'status': result['status'], 'queue_status': result['queue_status'],
        'final': result['final'], 'final_validation': result['final_validation'],
        'coordinator_alive': identity_alive(state.get('pid'), state.get('create_time')),
        'planned': len(tasks), 'completed': len(finished_cases), 'remaining': len(tasks) - len(finished_cases),
        'successes': sum(r['case_state'] == 'success' for r in tasks),
        'task_failures': sum(r.get('failure_kind') == 'task' for r in tasks),
        'errors': sum(r.get('failure_kind') == 'infrastructure' for r in tasks),
        'unknown_failures': sum(r.get('failure_kind') == 'unknown' for r in tasks),
        'outcome_details': dict(Counter(r['outcome_detail'] for r in tasks if r.get('terminal'))),
        'unattempted': sum(not r.get('id') for r in tasks),
        'unresolved': sum(r.get('state') in ('reserved', 'needs_diagnosis') for r in tasks),
        'blocked': sum(r['case_state'] == 'blocked' for r in tasks),
        'active': active[0] if active else None, 'active_runs': active, 'tasks': tasks,
        'max_concurrency': state.get('max_concurrency', manifest.get('max_concurrency')),
        'concurrency_ceiling': manifest.get('max_concurrency'), 'resources': state.get('resources'),
        'benchmark_label': 'RoboMME-Plus · E4 难度递进', 'split': 'train', 'episode': '0→1→2 / 独立0–4',
        'model': 'gpt-6-astra', 'effort': 'medium', 'call_budget': 100, 'policy_label': 'workspace-write / 无审批 / skills=[]',
        'hold_reason': read(out / 'control.json').get('hold_reason'),
    }, 'usage': {**total, 'rejections': total['decision_rejections'], 'recovered_decisions': total['recoveries'],
        'pending_requests': total['pending_usage'], 'previous_run_tokens': result['historical_usage']['total_tokens'],
        'historical_usage': result['historical_usage'],
        'note': '本轮训练用量；默认每项最多3次，三项额度中断场景经用户特批各补第4次；实际限额见场景。不同版本成绩分开，优化代理用量另记。',
        'cost_formula': f"成本 = {total['prompt_tokens']/1e6:.6f} × 输入单价 + {total['completion_tokens']/1e6:.6f} × 输出单价（每百万 token）",
    }, 'quota': quota, 'optimizer': optimizer, 'optimizer_usage': result['optimizer_usage'],
        'monitor': {'interval_label': '失败即分析 · 每 30 分钟巡检', 'next_check_at': optimizer.get('next_check_at'),
                    'progress_source': '后台文件事件推送；对话不轮询', 'quota_interval_seconds': 300}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = write_report(args.out.resolve())
    print(json.dumps({'status': report['status'], 'totals': report['totals']}))
