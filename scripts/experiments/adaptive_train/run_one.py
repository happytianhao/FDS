"""One immutable train attempt. The coordinator alone allocates attempts."""
import argparse
import os
import signal
import subprocess
import tomllib
from pathlib import Path

import psutil
from campaign import verify_version
from common import atomic, lock, now, read


def worker_command(req):
    return [str(Path(req['root']) / '.venv/bin/python'), '-m', 'fds.cli', 'run',
            '--config', req['config'], '--task', req['task'], '--episodes', str(req['episode']),
            '--output', req['output']]


def worker_environment(req):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(req['source_root']).resolve())
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['FDS_WORKSPACE'] = req['root']
    policy = req.get('resource_policy', {})
    if policy.get('enabled'):
        threads = str(max(1, int(policy.get('threads_per_scene', 1))))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[key] = threads
    env['PATH'] = str(Path.home() / '.local/bin') + os.pathsep + env.get('PATH', '')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = 'http://127.0.0.1:21084'
    env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost,::1'
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    return env


def stop_child(child):
    try:
        descendants = psutil.Process(child.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
    for proc in descendants:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(descendants, timeout=3)
    for proc in remaining:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(remaining, timeout=3)


def execute(request_path):
    req = read(request_path)
    folder = request_path.parent
    out = Path(req['out'])
    with lock(folder / 'runner.lock', nonblocking=True):
        child = None
        state = {}
        def save(**updates):
            state.update(updates, updated_utc=now())
            atomic(folder / 'status.json', state)
        try:
            with lock(out / 'state.lock'):
                state.update(read(folder / 'status.json'))
                if state.get('status') != 'starting' or state.get('runner_pid') != os.getpid():
                    raise ValueError('Runner reservation mismatch; refusing duplicate')
                verify_version(out, req['version'])
                if Path(req['output']).exists() or (folder / 'worker.log').exists():
                    raise ValueError('Output already exists; refusing overwrite')
                cmd = worker_command(req)
                env = worker_environment(req)
                save(command=cmd, runner_pid=os.getpid(), runner_create_time=psutil.Process().create_time(),
                     thread_limits={k: env.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
                with (folder / 'worker.log').open('xb') as stream:
                    child = subprocess.Popen(cmd, cwd=Path(req['source_root']).parent,
                                             env=env, stdin=subprocess.DEVNULL,
                                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                save(status='running', worker_pid=child.pid, worker_create_time=psutil.Process(child.pid).create_time())
            limit = tomllib.loads(Path(req['config']).read_text())['run']['episode_timeout'] + 60
            child.wait(timeout=limit)
            save(status='completed' if child.returncode == 0 else 'failed', returncode=child.returncode,
                 finished_utc=now())
        except BaseException as error:
            if child is not None:
                stop_child(child)
            # Do not replace a completed attempt when a manually duplicated wrapper is invoked.
            if state.get('runner_pid') == os.getpid():
                save(status='failed', returncode=child.returncode if child else None,
                     infrastructure_error={'error_type': type(error).__name__, 'error': str(error)}, finished_utc=now())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    def terminated(signum, frame):
        raise InterruptedError(f'Runner received signal {signum}')
    signal.signal(signal.SIGTERM, terminated)
    execute(args.request.resolve())


if __name__ == '__main__':
    main()
