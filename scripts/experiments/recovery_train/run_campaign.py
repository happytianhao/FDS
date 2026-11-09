"""Bounded process concurrency with adoption of existing scene runners; no overwrite."""
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
import platform
import selectors
import subprocess
import traceback
from pathlib import Path

import psutil


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def identity_alive(pid, created):
    try:
        p = psutil.Process(pid)
        return p.create_time() == created and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, TypeError):
        return False


def process_exit_fd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # This project's Python/glibc omit pidfd_open; the Linux x86_64 kernel supports it.
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Process-exit descriptor unavailable on this platform')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    os.set_inheritable(fd, False)
    return fd


def run_queue(pending, active, limit, start, finish, changed):
    """Wait on Linux process-exit events, counting adopted runners toward the cap."""
    assert 1 <= limit <= 3 and len(active) <= limit
    pending = list(pending)
    with selectors.DefaultSelector() as selector:
        def watch(task, item):
            try:
                fd = process_exit_fd(item['pid'])
            except ProcessLookupError:
                finish(task, item)
                active.pop(task)
                return
            selector.register(fd, selectors.EVENT_READ, task)

        for task, item in list(active.items()):
            watch(task, item)
        while pending or active:
            while pending and len(active) < limit:
                task = pending.pop(0)
                item = start(task)
                if item is not None:
                    active[task] = item
                    watch(task, item)
            changed(active, pending)
            if not active:
                continue
            for event, _ in selector.select():
                task = event.data
                selector.unregister(event.fd)
                os.close(event.fd)
                item = active.pop(task)
                if item.get('process') is not None:
                    item['process'].wait()  # pidfd already signalled; reap our child.
                finish(task, item)
        changed(active, pending)


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[3]
    python = root / '.venv/bin/python'
    campaign = read(out / 'campaign.json')
    limit = campaign['execution_policy']['max_concurrency']
    lock = (out / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_state = read(out / 'status.json')
    completed = {r['task']: r for r in previous_state.get('completed', [])}
    state = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
             'planned': len(campaign['tasks']), 'max_concurrency': limit,
             'completed': list(completed.values()), 'active': None, 'active_runs': []}

    def record(task):
        return {'condition': 'f1', 'task': task, 'episode': 0,
                'output': str(out / 'train_f1' / f'{task}_ep0')}

    def save(**updates):
        state.update(updates, updated_utc=datetime.datetime.now(datetime.UTC).isoformat())
        atomic(out / 'status.json', state)

    def collect():
        for name in ('collect_results.py', 'render_report.py'):
            result = subprocess.run([str(python), str(out / name)], cwd=root,
                                    capture_output=True, text=True)
            with (out / 'logs/collector.log').open('a') as log:
                log.write(result.stdout + result.stderr)
            if result.returncode:
                state.setdefault('collection_errors', []).append({
                    'script': name, 'error': result.stderr[-2000:]})

    def finish(task, item):
        runner = read(out / 'logs' / f'f1_{task}_ep0.status.json')
        # A dead wrapper is not proof its worker has exited: never free that slot silently.
        if identity_alive(runner.get('worker_pid'), runner.get('worker_create_time')):
            raise RuntimeError(f'{task}: runner exited but identified worker is still alive; stopping dispatch')
        directory = out / 'train_f1' / f'{task}_ep0'
        summary = read(directory / task / 'episode_0000/summary.json')
        metrics = read(directory / 'metrics.json')
        row = record(task)
        row.update(success=summary.get('success', False),
                   official_status=summary.get('official_status', 'unknown'),
                   stop_reason=summary.get('stop_reason', 'missing_result'),
                   recovery=summary.get('decision_recovery', {}))
        if not summary or metrics.get('status') != 'completed':
            row.update(errors=1, error_type='IncompleteScene',
                       error='Missing complete summary/metrics; partial artifacts preserved')
        elif runner.get('status') == 'failed' or metrics.get('errors'):
            row.update(errors=1, error_type=summary.get('error_type', 'SceneError'),
                       error=summary.get('error') or runner.get('infrastructure_error'))
        if item.get('dispatch_error'):
            row.update(errors=1, error_type='DispatchError', error=item['dispatch_error'])
        if row.get('errors'):
            path = out / 'logs' / f'f1_{task}_ep0.failure.json'
            if not path.exists():
                atomic(path, row)
        completed[task] = row
        state['completed'] = [completed[t] for t in campaign['tasks'] if t in completed]

    def verify_frozen():
        for name, digest in campaign['fds_source_sha256'].items():
            if hashlib.sha256((root / 'src/fds' / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Source changed during campaign: {name}')
        if hashlib.sha256(Path(campaign['configs']['f1']).read_bytes()).hexdigest() != campaign['config_sha256']:
            raise RuntimeError('Temporary configuration changed during campaign')

    def start(task):
        verify_frozen()
        assert not (out / 'logs' / f'f1_{task}_ep0.status.json').exists()
        assert not (out / 'train_f1' / f'{task}_ep0').exists()
        with (out / 'logs' / f'f1_{task}_ep0.runner.log').open('xb') as log:
            process = subprocess.Popen([str(python), str(out / 'run_one.py'), 'f1', task, '0'],
                                       cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        return {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
                'process': process, 'adopted': False}

    def changed(active, pending):
        running = [{**record(task), **{k: item[k] for k in ('pid', 'create_time', 'adopted')}}
                   for task, item in active.items()]
        save(status='running', active=running[0] if running else None,
             active_runs=running, pending=pending,
             completed=[completed[t] for t in campaign['tasks'] if t in completed])
        collect()

    try:
        verify_frozen()
        active, pending = {}, []
        for task in campaign['tasks']:
            runner = read(out / 'logs' / f'f1_{task}_ep0.status.json')
            if identity_alive(runner.get('runner_pid'), runner.get('runner_create_time')):
                proc = psutil.Process(runner['runner_pid'])
                args = proc.cmdline()
                if args[1:] != [str(out / 'run_one.py'), 'f1', task, '0']:
                    raise RuntimeError(f'{task}: runner identity mismatch')
                active[task] = {'pid': proc.pid, 'create_time': proc.create_time(), 'adopted': True}
                completed.pop(task, None)
            elif task in completed:
                continue
            elif runner or (out / 'train_f1' / f'{task}_ep0').exists():
                finish(task, {})
            else:
                pending.append(task)
        # The single coordinator lock owns all starts. Existing runners are adopted, not relaunched.
        run_queue(pending, active, limit, start, finish, changed)
        save(status='completed', active=None, active_runs=[], pending=[],
             finished_utc=datetime.datetime.now(datetime.UTC).isoformat())
        collect()
    except BaseException as error:
        save(status='needs_diagnosis', dispatch_error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
