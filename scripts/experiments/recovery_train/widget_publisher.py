"""Event-driven widget publisher. No inference, experiment control, or AI polling."""
import argparse
import ctypes
import datetime
import fcntl
import json
import os
import select
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
from read_quota import read_quota

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
# The widget target belongs to the deployment, not to the source template.
IDENTITY = json.loads((OUT / 'widget-target.json').read_text())
API = 'https://memory.whalent.com/pt/public/user-kb-api/widget/'
CREDS = Path.home() / '.config/fds-monitor/widget-credentials.json'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:21084'}))
cache = {}


def utc():
    return datetime.datetime.now(datetime.UTC).isoformat()


def read(path, default=None):
    try:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        old = cache.get(path)
        if old and old[0] == key:
            return old[1]
        value = json.loads(path.read_text())
        cache[path] = (key, value)
        return value
    except (OSError, ValueError):
        return {} if default is None else default


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def api(endpoint, fields):
    secret = json.loads(CREDS.read_text())['token']
    request = urllib.request.Request(
        API + endpoint, data=json.dumps({**IDENTITY, **fields}, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + secret}, method='POST',
    )
    with OPENER.open(request, timeout=20) as response:
        value = json.load(response)
    if value.get('ok') is False or value.get('error'):
        raise RuntimeError('Widget API rejected request')
    return value


def alive(state):
    try:
        process = psutil.Process(state['pid'])
        return process.create_time() == state['create_time'] and process.status() != psutil.STATUS_ZOMBIE
    except (KeyError, psutil.NoSuchProcess):
        return False


def snapshot():
    campaign, state = read(OUT / 'campaign.json'), read(OUT / 'status.json')
    config = __import__('tomllib').loads((OUT / 'config-snapshot.toml').read_text())
    records = {r['task']: r for r in state.get('completed', [])}
    active_tasks = {r['task'] for r in state.get('active_runs', [])}
    if not active_tasks and state.get('active'):
        active_tasks.add(state['active']['task'])
    tasks = []
    sums = {k: 0 for k in ('prompt_tokens', 'completion_tokens', 'planner_calls', 'environment_steps', 'rejections', 'recovered_decisions', 'missing_usage', 'pending_requests')}
    for task in campaign['tasks']:
        directory = OUT / 'train_f1' / f'{task}_ep0'
        ep = directory / task / 'episode_0000'
        summary, metadata = read(ep / 'summary.json'), read(ep / 'benchmark_metadata.json')
        calls = sorted(ep.glob('call_*'))
        totals = {'prompt_tokens': 0, 'completion_tokens': 0, 'missing_usage': 0, 'pending_requests': 0}
        for call in calls:
            attempts = sorted(call.glob('attempt_*')) or [call]
            for attempt in attempts:
                stats = read(attempt / 'stats.json')
                usage = stats.get('usage') or {}
                if not usage:
                    if task in records:
                        totals['missing_usage'] += 1
                    else:
                        totals['pending_requests'] += 1
                for key, alternate in (('prompt_tokens', 'input_tokens'), ('completion_tokens', 'output_tokens')):
                    totals[key] += usage.get(alternate, usage.get(key, 0)) or 0
        steps = summary.get('steps')
        if steps is None:
            last = calls[-1].name if calls else 'call_0000'
            steps = max(read(ep / 'audit' / last / name).get('step', 0) for name in ('before.json', 'after.json'))
        record = records.get(task, {})
        is_error = bool(record.get('errors')) or summary.get('stop_reason') == 'error'
        row = {
            'task': task, 'state': ('error' if is_error else 'completed') if task in records else 'running' if task in active_tasks else 'pending',
            'success': bool(summary.get('success')) if task in records else False,
            'official_status': summary.get('official_status'), 'stop_reason': summary.get('stop_reason'),
            'planner_calls': summary.get('planner_calls', len(calls)), 'environment_steps': steps,
            'rejections': len(list((ep / 'audit').glob('call_*/decision_rejection.json'))),
            'recoveries': len(list((ep / 'audit').glob('call_*/decision_recovery.json'))),
            'error': summary.get('error') or metadata.get('result', {}).get('failure_reason') or record.get('error'),
            **totals, 'total_tokens': totals['prompt_tokens'] + totals['completion_tokens'],
        }
        if isinstance(row['error'], dict):
            row['error'] = row['error'].get('error', '运行异常，详见报告')
        if not row['error'] and row['official_status'] == 'timeout':
            row['error'] = f'官方环境步数耗尽：{steps} 步'
        tasks.append(row)
        for key in sums:
            sums[key] += row.get('recoveries' if key == 'recovered_decisions' else key, 0)
    finished = [r for r in tasks if r['task'] in records]
    sums['total_tokens'] = sums['prompt_tokens'] + sums['completion_tokens']
    previous = read(ROOT / 'runs/overnight/results.json').get('totals', {}).get('all', {})
    e2_usage = read(Path(campaign['previous_campaign']) / 'results.json').get('totals', {}).get('all', {})
    sums.update(
        previous_run_tokens=previous.get('prompt_tokens', 0) + previous.get('completion_tokens', 0) + e2_usage.get('prompt_tokens', 0) + e2_usage.get('completion_tokens', 0),
        note='E3：仅 7 项未成功/异常复跑；此前用量合并 E1+E2，已知值不含 E2 两个缺失请求。每份回执计一次，调用数含当前请求。',
        cost_formula=f"成本 = {sums['prompt_tokens']/1e6:.6f} × 输入单价 + {sums['completion_tokens']/1e6:.6f} × 输出单价（每百万 token；单价未设置）",
    )
    schedule = read(OUT / 'schedule-arrangement.json')
    return {
        'updated_at': utc(),
        'experiment': {
            'status': state.get('status', 'unknown'), 'coordinator_alive': alive(state),
            'planned': len(tasks), 'completed': len(finished), 'remaining': len(tasks)-len(finished),
            'successes': sum(r['success'] for r in finished),
            'task_failures': sum(not r['success'] and r['state'] != 'error' for r in finished),
            'errors': sum(r['state'] == 'error' for r in finished),
            'active': next((r for r in tasks if r['task'] in active_tasks), None),
            'active_runs': [r for r in tasks if r['task'] in active_tasks],
            'max_concurrency': state.get('max_concurrency', 1), 'tasks': tasks,
            'benchmark_label': 'RoboMME-Plus · E3', 'split': config['benchmark']['split'], 'episode': config['run']['episodes'][0],
            'model': config['planner']['model'], 'effort': config['planner']['effort'],
            'call_budget': config['run']['max_calls'], 'policy_label': 'workspace-write / 无审批',
        },
        'usage': sums, 'quota': read(OUT / 'quota-snapshot.json'),
        'monitor': {'interval_label': '待重新启用' if schedule.get('time_rejected') else '每 30 分钟', 'next_check_at': datetime.datetime.fromtimestamp(schedule['next_run_at']/1000, datetime.UTC).isoformat() if schedule.get('next_run_at') else None,
                    'scheduling_error': schedule.get('time_rejection_reason'),
                    'progress_source': '文件变更事件推送，最长约 15 秒合并；60 秒心跳', 'quota_interval_seconds': 300},
    }


def refresh_quota():
    old = read(OUT / 'quota-snapshot.json')
    value = read_quota()
    if value['status'] == 'unknown' and old.get('status') == 'available':
        value = {**old, 'refresh_error': value.get('error'), 'last_attempt_at': value['sampled_at']}
    atomic(OUT / 'quota-snapshot.json', value)
    with (OUT / 'quota-history.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def publish():
    value = snapshot()
    atomic(OUT / 'widget-data.json', value)
    response = api('data', {'data': value, 'mode': 'merge'})
    return value, response


def serve():
    lock = (OUT / 'widget-publisher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(), 'started_at': utc()}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError('Could not subscribe to file events')
    watched = set()

    def watch():
        paths = [OUT, OUT / 'logs', OUT / 'train_f1']
        for pattern in ('train_f1/*', 'train_f1/*/*', 'train_f1/*/*/episode_*',
                        'train_f1/*/*/episode_*/call_*', 'train_f1/*/*/episode_*/call_*/attempt_*',
                        'train_f1/*/*/episode_*/audit', 'train_f1/*/*/episode_*/audit/call_*'):
            paths.extend(OUT.glob(pattern))
        for path in paths:
            if path.is_dir() and path not in watched:
                if libc.inotify_add_watch(fd, os.fsencode(path), 0x8 | 0x80 | 0x100) >= 0:
                    watched.add(path)

    relevant = {'status.json', 'results.json', 'metrics.json', 'summary.json', 'stats.json', 'before.json', 'after.json', 'decision_rejection.json', 'decision_recovery.json', 'fds_verdict.json', 'fds_contract.json'}
    last_push, next_quota, dirty = 0, time.monotonic(), True
    if read(OUT / 'quota-snapshot.json').get('sampled_at'):
        sampled = datetime.datetime.fromisoformat(read(OUT / 'quota-snapshot.json')['sampled_at']).timestamp()
        next_quota += max(0, 300 - (time.time() - sampled))
    try:
        watch()
        while True:
            current = time.monotonic()
            if current >= next_quota:
                refresh_quota()
                next_quota = time.monotonic() + 300
                dirty = True
            if (dirty and current - last_push >= 15) or current - last_push >= 60:
                try:
                    value, response = publish()
                    finished = value['experiment']['status'] == 'completed'
                    atomic(OUT / 'publisher-status.json', {**identity, 'status': 'completed' if finished else 'running',
                           'last_success_at': utc(), 'data_revision': response.get('data_revision'), 'last_error': None})
                    dirty = False
                    if finished:
                        return
                except Exception as error:
                    atomic(OUT / 'publisher-status.json', {**identity, 'status': 'publish_error', 'updated_at': utc(), 'last_error': type(error).__name__})
                    if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                        return
                last_push = time.monotonic()
            timeout = max(.1, min(next_quota - time.monotonic(), (last_push + (15 if dirty else 60)) - time.monotonic()))
            if select.select([fd], [], [], timeout)[0]:
                raw = os.read(fd, 262144)
                offset, new_directory = 0, False
                while offset + 16 <= len(raw):
                    _, mask, _, length = struct.unpack_from('iIII', raw, offset)
                    name = raw[offset+16:offset+16+length].split(b'\0')[0].decode(errors='replace')
                    offset += 16 + length
                    if mask & 0x40000000:
                        new_directory = True
                    if name in relevant or name.endswith('.status.json') or mask & 0x4000:
                        dirty = True
                if new_directory:
                    watch()
                    dirty = True
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--write-source', action='store_true')
    args = parser.parse_args()
    if args.write_source:
        source = (OUT / 'Widget.vue').read_text()
        assert len(source.encode()) < 512 * 1024
        assert json.loads(CREDS.read_text())['token'] not in source
        response = api('write', {'source': source})
        print(json.dumps({'revision': response.get('revision')}))
    elif args.once:
        value, response = publish()
        print(json.dumps({'completed': value['experiment']['completed'], 'total_tokens': value['usage']['total_tokens'], 'data_revision': response.get('data_revision')}, ensure_ascii=False))
    else:
        serve()
