import json
from pathlib import Path

out = Path(__file__).resolve().parent
state = json.loads((out / 'status.json').read_text())
active = state.get('active_runs', []) or ([state['active']] if state.get('active') else [])
progress = {'status': state['status'], 'completed': len(state['completed']), 'planned': state['planned'],
            'max_concurrency': state.get('max_concurrency', 1),
            'results': [{k: r.get(k) for k in ('task', 'success', 'official_status', 'stop_reason', 'error')} for r in state['completed']],
            'active': []}
for run in active:
    episode = Path(run['output']) / run['task'] / 'episode_0000'
    calls = sorted(episode.glob('call_*'))
    item = {'task': run['task'], 'calls_started': len(calls),
            'rejections': len(list((episode / 'audit').glob('call_*/decision_rejection.json')))}
    before = episode / 'audit' / calls[-1].name / 'before.json' if calls else None
    if before and before.exists():
        item['steps_at_last_call'] = json.loads(before.read_text()).get('step')
    progress['active'].append(item)
print(json.dumps(progress, ensure_ascii=False))
