"""Finite, paired train panel. Stops new admission on inference infrastructure errors."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASKS = ("PatternLock", "VideoPlaceOrder", "SwingXtimes", "ButtonUnmask")
VARIANTS = ("full", "no_dynamics", "no_history", "sparse16")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--tag", default="panel")
    parser.add_argument("--resume-errors", action="store_true", help="One explicit recovery attempt, with exact replay verification")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = [{"task": task, "episode": 0, "variant": variant} for variant in args.variants for task in args.tasks]
    queue = list(plan)
    (out / (args.tag + ".json")).write_text(json.dumps({"tasks": args.tasks, "variants": args.variants, "episodes": [0],
        "split": "train", "workers": args.workers, "planned_episodes": len(plan),
        "design": "Paired seed, one-factor-at-a-time pilot; no inferential population claims; max_calls=40."}, indent=2))
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HTTP_PROXY="http://127.0.0.1:21084", HTTPS_PROXY="http://127.0.0.1:21084",
                       ALL_PROXY="http://127.0.0.1:21084", NO_PROXY="127.0.0.1,localhost")

    def job(case):
        name = f"{case['variant']}_{case['task']}_ep{case['episode']}"
        target = out / name
        pilot = out / ("pilot_" + name)
        if (pilot / "result.json").exists():
            return json.loads((pilot / "result.json").read_text())
        resume_source = None
        if (target / "result.json").exists():
            result = json.loads((target / "result.json").read_text())
            recovered = out / (name + "_resume1") / "result.json"
            if result["outcome"] == "infrastructure_error" and recovered.exists():
                return json.loads(recovered.read_text())
            if not args.resume_errors or result["outcome"] != "infrastructure_error":
                return result
            resume_source = target
            name += "_resume1"
            target = out / name
            if (target / "result.json").exists():
                return json.loads((target / "result.json").read_text())
        if target.exists():
            raise RuntimeError("Unfinished output requires explicit diagnosis: " + str(target))
        cmd = [sys.executable, str(Path(__file__).with_name("run_one.py")), "--task", case["task"],
               "--variant", case["variant"], "--episode", str(case["episode"]), "--output", str(target)]
        if resume_source:
            cmd.extend(["--resume-from", str(resume_source)])
        with (out / (name + ".log")).open("x") as stream:
            process = subprocess.run(cmd, env=environment, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        result_path = target / "result.json"
        if not result_path.exists():
            return {**case, "success": None, "outcome": "missing_result", "returncode": process.returncode}
        return json.loads(result_path.read_text())

    results, pending, halted = [], {}, False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue or pending:
            while queue and len(pending) < args.workers and not halted:
                case = queue.pop(0)
                pending[pool.submit(job, case)] = case
                print(json.dumps({"event": "started", **case}), flush=True)
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                case = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**case, "success": None, "outcome": "runner_error", "error": repr(exc)}
                results.append(result)
                print(json.dumps({"event": "finished", **result}), flush=True)
                if result.get("success") is None:
                    halted = True
            (out / (args.tag + "_progress.json")).write_text(json.dumps({"updated": time.time(), "halted": halted,
                "finished": results, "running": list(pending.values()), "not_started": queue}, indent=2))
    print(json.dumps({"event": "panel_done", "halted": halted, "completed": len(results), "not_started": len(queue)}), flush=True)


if __name__ == "__main__":
    main()
