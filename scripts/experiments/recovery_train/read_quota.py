"""Read Codex account usage only. Never start a thread/turn or expose auth/account IDs.

Protocol: https://learn.chatgpt.com/docs/app-server
Schema checked against the locally installed Codex 0.163.0.
"""
import datetime
import json
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path


def read_quota():
    process = subprocess.Popen(
        [os.environ.get('FDS_CODEX_BIN', 'codex'), 'app-server', '--listen', 'stdio://'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffer = b''
    deadline = time.monotonic() + 40

    def send(value):
        process.stdin.write((json.dumps(value) + '\n').encode())
        process.stdin.flush()

    def response(identifier):
        nonlocal buffer
        while time.monotonic() < deadline:
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('id') == identifier:
                    if 'error' in event:
                        # Do not copy arbitrary backend error text into public data.
                        raise RuntimeError('Codex usage RPC failed')
                    return event['result']
            if process.poll() is not None:
                raise RuntimeError('Codex usage reader exited')
            for key, _ in selector.select(max(0, min(1, deadline - time.monotonic()))):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    raise RuntimeError('Codex usage stream closed')
                buffer += chunk
        raise TimeoutError('Codex usage read timed out')

    sampled = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        send({'id': 1, 'method': 'initialize', 'params': {
            'clientInfo': {'name': 'fds_usage_widget', 'title': 'FDS usage widget', 'version': '1.0'},
            'capabilities': {'experimentalApi': True},
        }})
        response(1)
        send({'method': 'initialized', 'params': {}})
        send({'id': 2, 'method': 'account/rateLimits/read', 'params': {
            'excludeResetCreditDetails': True, 'supportsLunaReserve': False,
        }})
        value = response(2)
        buckets = value.get('rateLimitsByLimitId') or {'codex': value.get('rateLimits', {})}
        safe_buckets = []
        for identifier, bucket in buckets.items():
            windows = []
            for name in ('primary', 'secondary'):
                window = bucket.get(name)
                if window is not None:
                    windows.append({
                        'name': name, 'used_percent': window.get('usedPercent'),
                        'remaining_percent': max(0, 100 - window['usedPercent']) if window.get('usedPercent') is not None else None,
                        'duration_minutes': window.get('windowDurationMins'), 'resets_at': window.get('resetsAt'),
                    })
            safe_buckets.append({
                'id': identifier, 'name': bucket.get('limitName') or identifier,
                'model': bucket.get('normalModelSlug'), 'plan': bucket.get('planType'),
                'windows': windows, 'credits': bucket.get('credits'),
                'spend_control_reached': bucket.get('spendControlReached'),
            })
        return {'status': 'available' if any(b['windows'] for b in safe_buckets) else 'unknown',
                'sampled_at': sampled, 'source': 'Codex account/rateLimits/read',
                'ordinary_usage_allowed': value.get('ordinaryUsageAllowed'), 'buckets': safe_buckets,
                'note': '账户共享额度，可能含其他会话；百分比不能换算为精确剩余 token。'}
    except Exception as error:
        return {'status': 'unknown', 'sampled_at': sampled, 'source': 'Codex account/rateLimits/read',
                'error': type(error).__name__, 'buckets': [], 'note': '额度暂不可读；不会从 token 用量推算剩余额度。'}
    finally:
        selector.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        process.stdin.close()
        process.stdout.close()


if __name__ == '__main__':
    value = read_quota()
    path = Path(__file__).resolve().parent / 'quota-snapshot.json'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(value, ensure_ascii=False))
