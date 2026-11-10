"""Synthetic resource admission tests, without scenes or provider requests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments/adaptive_train'))
import campaign
import resources
from common import FileEvents, atomic, lock, read
from run_one import worker_environment


def ample():
    return dict(cpu_cores=12, memory_available_bytes=90 * resources.GIB,
                gpu_available_bytes=78 * resources.GIB, gpu_utilization=1)


def test_quota_not_host_affinity_and_cgroup_memory(tmp_path, monkeypatch):
    cpu, memory = tmp_path / 'cpu', tmp_path / 'memory'
    cpu.mkdir()
    memory.mkdir()
    (cpu / 'cpu.cfs_quota_us').write_text('1200000')
    (cpu / 'cpu.cfs_period_us').write_text('100000')
    (cpu / 'cpuacct.usage').write_text('2000000000')
    (memory / 'memory.limit_in_bytes').write_text(str(120 * resources.GIB))
    (memory / 'memory.usage_in_bytes').write_text(str(30 * resources.GIB))
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda _: set(range(128)))
    metrics = resources.container_metrics(tmp_path)
    assert metrics['cpu_cores'] == 12
    assert metrics['memory_available_bytes'] <= 90 * resources.GIB
    assert metrics['cpu_usage_seconds'] == 2


def test_cgroup_v2(tmp_path, monkeypatch):
    (tmp_path / 'cpu.max').write_text('250000 100000')
    (tmp_path / 'cpu.stat').write_text('usage_usec 3000000\n')
    (tmp_path / 'memory.max').write_text(str(16 * resources.GIB))
    (tmp_path / 'memory.current').write_text(str(4 * resources.GIB))
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda _: set(range(128)))
    metrics = resources.container_metrics(tmp_path)
    assert metrics['cpu_cores'] == 2.5
    assert metrics['memory_available_bytes'] <= 12 * resources.GIB


def test_ramp_pressure_and_no_gpu_telemetry():
    metrics = ample()
    assert resources.admission({}, metrics, 2) == 6
    assert resources.admission({}, metrics, 6, previous=6) == 8
    assert resources.admission({}, metrics, 8, previous=8) == 10
    assert resources.admission({}, metrics, 10, previous=10) == 10
    assert resources.admission({}, {**metrics, 'cpu_utilization': .95}, 6, previous=8) == 6
    assert resources.admission({}, {**metrics, 'gpu_utilization': 95}, 6, previous=8) == 6
    assert resources.admission({}, {**metrics, 'gpu_available_bytes': None}, 6, previous=8) == 6
    # Existing scenes remain alive even when target shrinks; only admission changes.
    assert resources.admission({}, {**metrics, 'memory_available_bytes': 10 * resources.GIB}, 6, previous=8) == 6


def test_filtered_fileevents_ignore_own_status_and_lock(tmp_path):
    events = FileEvents()
    try:
        events.watch(tmp_path, names={'control.json'})
        atomic(tmp_path / 'status.json', {'status': 'running'})
        with lock(tmp_path / 'state.lock'):
            pass
        assert not events.drain()
        atomic(tmp_path / 'control.json', {'halt': False})
        assert events.drain()
    finally:
        events.close()


def test_more_than_three_dispatches_reviewed_retry_first(tmp_path, monkeypatch):
    (tmp_path / 'reviews').mkdir()
    atomic(tmp_path / 'campaign.json', {'max_concurrency': 10, 'resource_policy': resources.DEFAULT_POLICY})
    atomic(tmp_path / 'control.json', {'active_version': 'v000'})
    cases = [dict(id=f'T{i}_ep0', task=f'T{i}', episode=0, state='pending', past_attempts=[]) for i in range(16)]
    cases[12].update(state='retry', past_attempts=[{}])
    snapshots = iter([dict(status='running', active_runs=[], completed=[], cases=cases),
                      dict(status='completed', active_runs=[], completed=[], cases=[])])
    monkeypatch.setattr(campaign, 'build_state', lambda _: next(snapshots))
    monkeypatch.setattr(resources.ResourceController, 'sample', lambda *args: {'target': 6})
    launched = []
    class Child:
        returncode = 0
        def poll(self):
            return 0
    def dispatch(_, case, version):
        launched.append(case['id'])
        return Child()
    monkeypatch.setattr(campaign, 'dispatch', dispatch)
    campaign.run(tmp_path)
    assert len(launched) == 6
    assert launched[0] == 'T12_ep0'
    assert read(tmp_path / 'status.json')['max_concurrency'] == 6


def test_thread_caps_only_for_adaptive_policy(tmp_path):
    req = {'root': str(tmp_path), 'out': str(tmp_path), 'source_root': str(tmp_path / 'src')}
    req['resource_policy'] = {'enabled': True, 'threads_per_scene': 1}
    env = worker_environment(req)
    assert env['OMP_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == env['MKL_NUM_THREADS'] == '1'
