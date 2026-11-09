"""Render rerun evidence without changing the original experiment results."""
import collections
import datetime
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def link(label, path):
    return f'[{label}]({path})'


campaign, state, result = (read(OUT / name) for name in ('campaign.json', 'status.json', 'results.json'))
rows = result.get('rows', [])
by_task = {r['task']: r for r in rows}
old = read(Path(campaign['old_results'])) if campaign.get('old_results') else {}
previous = {r['task']: r for r in read(Path(campaign['previous_campaign']) / 'results.json').get('rows', [])} if campaign.get('previous_campaign') else {}
old_rows = {(r['condition'], r['task']): r for r in old.get('rows', [])}
totals = result.get('totals', {}).get('all', {})
checks, rejection_reasons, failure_reasons = [], collections.Counter(), collections.Counter()
for row in rows:
    ep = Path(row['summary_path']).parent
    prior_row = previous.get(row['task']) or old_rows.get(('f1', row['task']), {})
    prior = Path(prior_row['summary_path']).parent if prior_row.get('summary_path') else None
    initial = read(ep / 'audit/call_0000/input.json')
    original_initial = read(prior / 'audit/call_0000/input.json') if prior else {}
    checked = []
    for path in sorted((ep / 'audit').glob('call_*/decision_rejection.json')):
        rejection = read(path)
        rejection_reasons[rejection['message']] += 1
        before, after = read(path.parent / 'before.json'), read(path.parent / 'after.json')
        checked.append({'call': path.parent.name, 'unchanged_state': before == after,
                        'no_memory_delta_applied': not (ep / 'memory_versions' / f'{path.parent.name}.json').exists(),
                        'no_execution_verdict': not (path.parent / 'fds_verdict.json').exists()})
    checks.append({'task': row['task'], 'seed_matches': row['official_seed'] == prior_row.get('official_seed') if prior else None,
                   'initial_images_match': bool(initial) and [x['sha256'] for x in initial['images']] == [x['sha256'] for x in original_initial.get('images', [])],
                   'within_original_call_budget': row['planner_calls'] <= 100,
                   'rejection_checks': checked})
    if not row['success']:
        failure_reasons[row.get('failure_explanation') or 'unknown'] += 1
audit = {'source_unchanged_during_rerun': result.get('source_unchanged'), 'runs': checks,
         'rejection_reasons': dict(rejection_reasons), 'failure_reasons': dict(failure_reasons)}
(OUT / 'audit_checks.json.tmp').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
(OUT / 'audit_checks.json.tmp').replace(OUT / 'audit_checks.json')

done = state.get('status') == 'completed'
lines = [
    '# F1 train 失效模式优化复跑 E3', '',
    f'状态：{"已结束" if done else "运行中"}；已记录 {len(rows)}/{len(campaign["tasks"])}。更新 UTC：{datetime.datetime.now(datetime.UTC).isoformat()}。', '',
    '范围：RoboMME-Plus，train，easy episode 0；只重跑 E2 未成功的 5 个任务及 2 个异常场景，以最多 3 个场景并行执行。',
    'Astra gpt-6-astra / medium；workspace-write；approval never；代理 127.0.0.1:21084；项目 .venv；memory 开启、skills=[]；seed=7；每集 max_calls=100、max_steps=10000、max_tool_steps=400，官方任务步数上限仍有效。', '',
    '改动：返回每个原始 waypoint 的实测 TCP 跟踪；保留最近六次动作反馈；加强前置事件、视频顺序、目标身份和停滞核验；外层等待已有有限传输重试完成。无 oracle、无新增 skill。',
    '每场景 max_calls=100（决策调用，含纠正）；传输重试沿用每决策最多 3 次，实际请求另行统计。关闭人工等待/额外恢复窗口。',
    'E1、E2 结果保持不动；E3 是经过 train 失败分析后的工程复跑，不构成同版本完整消融，也不是未见数据泛化结果。', '',
    '| 本次完成 | 正常结束 | 成功 | 任务未成功 | 进程错误 |',
    '|---:|---:|---:|---:|---:|',
    f"| {len(rows)} | {totals.get('completed_without_error', 0)} | {totals.get('successes', 0)} | {totals.get('task_failures', 0)} | {totals.get('process_errors', 0)} |", '',
    '| Task（train ep0） | E2 status | E2 stop_reason | 本次 F1 success/status | stop_reason | calls | prompt | completion | steps | 拒绝/纠正/恢复 |',
    '|---|---|---|---|---|---:|---:|---:|---:|---|',
]
for task in campaign['tasks']:
    row = by_task.get(task)
    prior_row = previous.get(task, {'official_status': 'N/A', 'stop_reason': 'N/A'})
    if row:
        lines.append(f"| {task} | {prior_row['official_status']} | {prior_row['stop_reason']} | {row['success']} / {row['official_status']} | {row['stop_reason']} | {row['planner_calls']} | {row['prompt_tokens']} | {row['completion_tokens']} | {row['environment_steps']} | {row['decision_rejections']}/{row['correction_calls']}/{row['recovered_decisions']} |")
    else:
        lines.append(f"| {task} | {prior_row['official_status']} | {prior_row['stop_reason']} | 待完成 | — | — | — | — | — | — |")
lines.extend(['', '拒绝/纠正/恢复分别表示无效响应数量、用于纠正的模型调用数量、最终通过校验的纠正序列数量；通过校验不代表任务成功。', ''])
proposed = totals.get('contracts_proposed', 0)
accepted = totals.get('contracts_accepted', 0)
lines.extend([
    f"合约：proposed={proposed}，accepted={accepted}，rejected={totals.get('contracts_rejected', 0)}，validity={accepted / proposed:.2%}。" if proposed else '暂无可统计合约。',
    f"verdict：contradicted={totals.get('contradicted', 0)}，uncertain={totals.get('uncertain', 0)}。合约拒绝和执行 contradicted 分开计数。",
    f"总 calls={totals.get('planner_calls', 0)}；prompt={totals.get('prompt_tokens', 0)}；completion={totals.get('completion_tokens', 0)}；总 token={totals.get('prompt_tokens', 0) + totals.get('completion_tokens', 0)}；环境 steps={totals.get('environment_steps', 0)}。",
    f"transport retries={totals.get('transport_retries', 0)}；usage unknown（报告值）={totals.get('reported_failed_attempt_usage_unknown', 0)}；实际缺失 usage 的请求={totals.get('attempts_without_recorded_token_usage', 0)}。", '',
    '成本只按 token 数给出，不假设 Astra 单价：', '',
    '```text',
    f"estimated_cost = {totals.get('prompt_tokens', 0) / 1e6:.6f} * input_price + {totals.get('completion_tokens', 0) / 1e6:.6f} * output_price",
    '```', '', 'input_price/output_price 为每百万 token 价格。E1 与 E2 的 token 均不包含在 E3 合计中；缺失 usage 时已知合计只是下界。', '',
    '## 错误与审计', '',
    f"运行期间源码快照一致：{result.get('source_unchanged')}；初始图像一致：{sum(c['initial_images_match'] for c in checks)}/{len(checks)}；官方 seed 一致：{sum(c['seed_matches'] is True for c in checks)}/{len(checks)}。",
    f"全部已记录请求符合指定模型/effort/sandbox/approval：{all(not r['settings_issues'] for r in rows)}。",
    '每次拒绝前后状态、memory 和 verdict 检查见 ' + link('audit_checks.json', OUT / 'audit_checks.json') + '。', '',
])
for row in rows:
    if not row['success']:
        lines.append(f"- {row['task']}：{row.get('failure_explanation')}")
if rejection_reasons:
    lines.extend(['', '收到的拒绝原因：', ''])
    for reason, count in rejection_reasons.items():
        lines.append(f'- {count} 次：{reason}')
lines.extend(['', '## 后续候选与输出', ''])
successes = [r['task'] for r in rows if r['success']]
lines.append('本次成功任务：' + ('、'.join(successes) if successes else '暂无') + '。')
lines.append('下一轮优先在 train 复核上述任务，再做同版本 F0/F1 配对；通过后再考虑测试集。原先 RouteStick 的 F1 单独成功和 PatternLock 双方成功可作为候选/对照，本轮未重跑。')
lines.append('尚未通过的任务按各自失败原因修复；若纠正次数耗尽，需要先改善合约生成；若任务正常结束但未成功，需要分析感知、身份跟踪、定位和动作执行，不能混同为恢复失败。')
lines.extend(['', link('结果 JSON', OUT / 'results.json') + ' · ' + link('结果 CSV', OUT / 'results.csv') + ' · ' + link('源码改动', OUT / 'source.patch') + ' · ' + link('配置快照', OUT / 'config-snapshot.toml') + ' · ' + link('原实验报告', ROOT / 'runs/overnight/REPORT.md'), ''])
for task in campaign['tasks']:
    path = OUT / 'train_f1' / f'{task}_ep0'
    lines.append(f'- {task}：{link(str(path), path)}')
(OUT / 'REPORT.md.tmp').write_text('\n'.join(lines) + '\n')
(OUT / 'REPORT.md.tmp').replace(OUT / 'REPORT.md')
print(json.dumps({'reported': len(rows), 'status': state.get('status'), 'successes': successes}))
