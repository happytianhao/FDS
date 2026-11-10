"""Container-aware resource admission. Existing scenes are never terminated."""
import ctypes
import math
import os
import subprocess
import time
from pathlib import Path

import psutil

GIB = 1024 ** 3
DEFAULT_POLICY = dict(enabled=True, max_concurrency=10, initial_concurrency=6,
                      ramp_step=2, sample_seconds=30, cpu_reserve=2, cpu_per_scene=1,
                      memory_reserve_gib=12, memory_per_scene_gib=4,
                      gpu_reserve_gib=8, gpu_per_scene_gib=2, threads_per_scene=1)


def number(path, default=None):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return default


def container_metrics(base=Path('/sys/fs/cgroup')):
    affinity = len(os.sched_getaffinity(0))
    quota = None
    v2 = base / 'cpu.max'
    if v2.exists():
        raw = v2.read_text().split()
        if raw[0] != 'max':
            quota = int(raw[0]) / int(raw[1])
        stat = dict(line.split() for line in (base / 'cpu.stat').read_text().splitlines())
        usage = int(stat.get('usage_usec', 0)) / 1e6
        limit, used = number(base / 'memory.max'), number(base / 'memory.current', 0)
    else:
        cpu = base / 'cpu'
        if not cpu.exists():
            cpu = base / 'cpu,cpuacct'
        q, period = number(cpu / 'cpu.cfs_quota_us'), number(cpu / 'cpu.cfs_period_us')
        if q and q > 0 and period:
            quota = q / period
        usage = number(cpu / 'cpuacct.usage', 0) / 1e9
        limit, used = number(base / 'memory/memory.limit_in_bytes'), number(base / 'memory/memory.usage_in_bytes', 0)
    host = psutil.virtual_memory()
    limit = min(limit or host.total, host.total)
    return dict(cpu_cores=min(affinity, quota or affinity), cpu_usage_seconds=usage,
                memory_limit_bytes=limit, memory_available_bytes=max(0, min(host.available, limit - used)))


def gpu_metrics():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        rows = [[float(v.strip()) for v in line.split(',')] for line in result.stdout.splitlines()]
        # Use the first visible GPU conservatively, never aggregate unrelated devices.
        total, used, utilization = rows[0]
        return dict(gpu_available_bytes=(total - used) * 1024 ** 2, gpu_utilization=utilization)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return dict(gpu_available_bytes=None, gpu_utilization=None)


def admission(policy, metrics, occupied, previous=None):
    p = {**DEFAULT_POLICY, **policy}
    ceiling = max(1, int(p['max_concurrency']))
    cpu_limit = max(1, math.floor((metrics['cpu_cores'] - p['cpu_reserve']) / p['cpu_per_scene']))
    memory_slots = max(0, math.floor((metrics['memory_available_bytes'] - p['memory_reserve_gib'] * GIB) /
                                    (max(p['memory_per_scene_gib'] * GIB, metrics.get('scene_rss_bytes', 0) * 1.5))))
    target = min(ceiling, cpu_limit, occupied + memory_slots)
    available = metrics.get('gpu_available_bytes')
    if available is None:
        target = min(target, occupied)  # Missing GPU telemetry: preserve runs, admit none.
    else:
        gpu_slots = max(0, math.floor((available - p['gpu_reserve_gib'] * GIB) / (p['gpu_per_scene_gib'] * GIB)))
        target = min(target, occupied + gpu_slots)
    pressure = metrics.get('cpu_utilization', 0) > .9 or (metrics.get('gpu_utilization') or 0) > 90
    if pressure:
        target = min(target, occupied)
    ramp = p['initial_concurrency'] if previous is None else previous + p['ramp_step']
    return max(0, min(target, ramp))


class ResourceController:
    def __init__(self):
        self.previous_sample = None
        self.previous_target = None
        self.cached = None

    def sample(self, manifest, active, occupied):
        policy = manifest.get('resource_policy', {})
        if not policy.get('enabled', False):
            return dict(target=manifest['max_concurrency'], policy_enabled=False)
        policy = {**DEFAULT_POLICY, **policy, 'max_concurrency': manifest['max_concurrency']}
        stamp = time.monotonic()
        if self.cached and stamp - self.cached['monotonic'] < policy['sample_seconds']:
            return self.cached
        metrics = {**container_metrics(), **gpu_metrics()}
        rss = []
        for row in active:
            try:
                proc = psutil.Process(row['pid'])
                rss.append(sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)]))
            except psutil.Error:
                pass
        metrics['scene_rss_bytes'] = max(rss, default=0)
        if self.previous_sample:
            elapsed = stamp - self.previous_sample[0]
            metrics['cpu_utilization'] = max(0, metrics['cpu_usage_seconds'] - self.previous_sample[1]) / elapsed / metrics['cpu_cores']
        self.previous_sample = (stamp, metrics['cpu_usage_seconds'])
        target = admission(policy, metrics, occupied, self.previous_target)
        self.previous_target = target
        self.cached = dict(target=target, policy_enabled=True, monotonic=stamp, metrics=metrics, policy=policy)
        return self.cached


class ResourceTimer:
    """Linux timerfd wakes resource sampling independently of scene/file activity."""
    def __init__(self, seconds):
        class Timespec(ctypes.Structure):
            _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]
        class Itimerspec(ctypes.Structure):
            _fields_ = [('interval', Timespec), ('value', Timespec)]
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.timerfd_create(1, os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'timerfd_create')
        spec = Itimerspec(Timespec(seconds, 0), Timespec(seconds, 0))
        if libc.timerfd_settime(self.fd, 0, ctypes.byref(spec), None) < 0:
            os.close(self.fd)
            raise OSError(ctypes.get_errno(), 'timerfd_settime')

    def drain(self):
        os.read(self.fd, 8)

    def close(self):
        os.close(self.fd)
