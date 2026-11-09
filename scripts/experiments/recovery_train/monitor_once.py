"""Read campaign progress once; no sleeps, polling loops, launches or scheduling."""
import json
import subprocess
from pathlib import Path

import psutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
state = json.loads((OUT / 'status.json').read_text())
try:
    process = psutil.Process(state['pid'])
    alive = process.create_time() == state['create_time'] and process.status() != psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:
    alive = False
subprocess.run([str(ROOT / '.venv/bin/python'), str(OUT / 'progress.py')], check=True)
results_path = OUT / 'results.json'
results = json.loads(results_path.read_text()) if results_path.exists() else {}
print(json.dumps({
    'coordinator_alive': alive,
    'needs_diagnosis': state['status'] != 'completed' and not alive,
    'collection_errors': state.get('collection_errors', []),
    'completed_totals': results.get('totals', {}).get('all'),
    'report': str(OUT / 'REPORT.md'),
}, ensure_ascii=False))
