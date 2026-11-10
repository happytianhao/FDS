"""Data-only widget updates from file events; account quota is sampled separately."""
import argparse
import ctypes
import importlib.util
import json
import os
import select
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
from common import atomic, lock, now, read
from reporting import safe_read, widget_data, write_report


def api(out, endpoint, fields):
    identity = read(out / 'widget-target.json')
    credential_file = Path.home() / '.config/fds-monitor/widget-credentials.json'
    if credential_file.stat().st_mode & 0o077:
        raise ValueError('Widget credential file must be private (0600)')
    secret = read(credential_file)['token']
    request = urllib.request.Request('https://memory.whalent.com/pt/public/user-kb-api/widget/' + endpoint,
        data=json.dumps({**identity, **fields}, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + secret}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:21084'}))
    with opener.open(request, timeout=20) as response:
        value = json.load(response)
    if value.get('ok') is False or value.get('error'):
        raise RuntimeError('Widget API rejected request')
    return value


def refresh_quota(out):
    path = Path(__file__).resolve().parents[1] / 'recovery_train/read_quota.py'
    spec = importlib.util.spec_from_file_location('fds_quota_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.read_quota()
    old = safe_read(out / 'quota-snapshot.json')
    if value['status'] == 'unknown' and old.get('status') == 'available':
        value = {**old, 'refresh_error': value.get('error'), 'last_attempt_at': value['sampled_at']}
    atomic(out / 'quota-snapshot.json', value)
    with (out / 'quota-history.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def publish(out):
    write_report(out)
    value = widget_data(out)
    atomic(out / 'widget-data.json', value)
    response = api(out, 'data', {'data': value, 'mode': 'merge'})
    return response


def serve(out):
    global_lock = Path.home() / '.config/fds-monitor/adaptive-widget.lock'
    with lock(global_lock, nonblocking=True):
        identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(), 'started_at': now()}
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError('inotify_init1 failed')
        watched = set()
        last_push, next_quota, dirty = 0, time.monotonic(), True
        names = {'status.json', 'control.json', 'summary.json', 'stats.json', 'before.json', 'after.json',
                 'decision_rejection.json', 'decision_recovery.json', 'fds_verdict.json', 'FINAL.json',
                 'NEEDS_USER_DECISION.json', 'review.json'}

        def watch():
            paths = [out, out / 'attempts', out / 'optimizer']
            for pattern in ('attempts/*', 'attempts/*/attempt_*', 'attempts/*/attempt_*/output/*/episode_*',
                            'attempts/*/attempt_*/output/*/episode_*/call_*',
                            'attempts/*/attempt_*/output/*/episode_*/call_*/attempt_*',
                            'attempts/*/attempt_*/output/*/episode_*/audit',
                            'attempts/*/attempt_*/output/*/episode_*/audit/call_*'):
                paths.extend(out.glob(pattern))
            for path in paths:
                if path.is_dir() and path not in watched:
                    if libc.inotify_add_watch(fd, os.fsencode(path), 0x8 | 0x80 | 0x100) >= 0:
                        watched.add(path)
        try:
            while True:
                watch()
                current = time.monotonic()
                if current >= next_quota:
                    refresh_quota(out)
                    next_quota = time.monotonic() + 300
                    dirty = True
                if dirty and current - last_push >= 15 or current - last_push >= 60:
                    try:
                        response = publish(out)
                        done = bool(safe_read(out / 'FINAL.json'))
                        atomic(out / 'publisher-status.json', {**identity, 'status': 'completed' if done else 'running',
                               'last_success_at': now(), 'data_revision': response.get('data_revision')})
                        if done:
                            return
                        dirty = False
                    except Exception as error:
                        atomic(out / 'publisher-status.json', {**identity, 'status': 'publish_error',
                               'updated_at': now(), 'error_type': type(error).__name__})
                        if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                            return
                    last_push = time.monotonic()
                timeout = max(0.1, min(next_quota - time.monotonic(), last_push + (15 if dirty else 60) - time.monotonic()))
                if select.select([fd], [], [], timeout)[0]:
                    raw = os.read(fd, 262144)
                    offset = 0
                    while offset + 16 <= len(raw):
                        _, mask, _, length = struct.unpack_from('iIII', raw, offset)
                        name = raw[offset + 16:offset + 16 + length].split(b'\0')[0].decode(errors='replace')
                        offset += 16 + length
                        if name in names or mask & (0x40000000 | 0x4000):
                            dirty = True
        finally:
            os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--write-source', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.write_source:
        existing = api(out, 'read', {})
        atomic(out / 'widget-before.json', existing)
        source = Path(__file__).with_name('Widget.vue').read_text()
        if read(Path.home() / '.config/fds-monitor/widget-credentials.json')['token'] in source:
            raise ValueError('Private credential found in source')
        response = api(out, 'write', {'source': source})
        print(json.dumps({'revision': response.get('revision'), 'source_updated': True}))
    elif args.once:
        response = publish(out)
        print(json.dumps({'data_revision': response.get('data_revision')}))
    else:
        serve(out)
