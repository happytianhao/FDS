"""Offline process tests: no FDS worker or model requests."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil", reason="optional experiment monitor dependency")

spec = importlib.util.spec_from_file_location('campaign_dispatch', Path(__file__).resolve().parents[1] / 'scripts/experiments/recovery_train/run_campaign.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_adoption_cap_and_continue_after_failed_child():
    children, starts, ends, counts = [], [], [], []

    def launch(task):
        # Distinct child completion times exercise replacement while an adopted run is alive.
        process = subprocess.Popen([sys.executable, '-c',
            'import time,sys; time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))',
            '0.25' if task == 'adopted' else '0.08', '1' if task == 'failure' else '0'])
        children.append(process)
        return {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
                'process': process, 'adopted': task == 'adopted'}

    active = {'adopted': launch('adopted')}

    def start(task):
        assert task not in starts and task != 'adopted'
        starts.append(task)
        item = launch(task)
        assert sum(p.poll() is None for p in children) <= 3
        return item

    def finish(task, item):
        ends.append((task, item['process'].returncode))

    def changed(running, pending):
        counts.append(len(running))
        assert len(running) <= 3

    try:
        module.run_queue(['b', 'failure', 'd', 'e', 'f'], active, 3, start, finish, changed)
        assert starts == ['b', 'failure', 'd', 'e', 'f']
        assert len(ends) == 6 and len({t for t, _ in ends}) == 6
        assert dict(ends)['failure'] == 1
        assert max(counts) == 3 and counts[-1] == 0
        assert not active
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
