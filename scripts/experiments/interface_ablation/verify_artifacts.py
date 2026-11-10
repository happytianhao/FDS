"""Audit saved actions, observations and paired initial scenes without model calls."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    hashes, initial, seeds, checked = {}, {}, {}, []
    for path in sorted(args.runs.glob("*/result.json")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        result = json.loads(path.read_text())
        records = [json.loads(line) for line in (path.parent / "exchange.jsonl").read_text().splitlines()]
        requests = [r["request"] for r in records if "request" in r]
        responses = [r["response"] for r in records if "response" in r]
        executions = [r["execution"] for r in records if "execution" in r]
        assert result["calls"] == len(responses), path
        for answer, action in zip(responses, executions, strict=False):
            assert action["action"] == [*answer["waypoint"], answer["gripper"]], path
        assert len(executions) == len(responses) - int(bool(responses) and responses[-1]["waypoint"] is None), path
        for request in requests:
            for im in request["observation"]["images"]:
                p = Path(im["path"])
                if p not in hashes:
                    hashes[p] = hashlib.sha256(p.read_bytes()).hexdigest()
                assert hashes[p] == im["sha256"], p
        first = requests[0]["observation"]
        task = manifest["task"]
        identity = (manifest["seed"], manifest["difficulty"], result["goal"], result["official_max_steps"])
        assert seeds.setdefault(task, identity) == identity, (task, "seed/goal/budget mismatch")
        for im in first["images"]:
            key = (task, im["camera"], im["source_index"])
            assert initial.setdefault(key, im["sha256"]) == im["sha256"], (task, "initial pixel mismatch")
        checked.append({"run": path.parent.name, "responses": len(responses), "actions": len(executions)})
    result = {"checked_runs": checked, "unique_image_files_verified": len(hashes),
              "matched_initial_frames_across_conditions": len(initial),
              "paired_task_seeds": {k: list(v[:2]) for k, v in seeds.items()},
              "checks": "Image bytes match recorded hashes; action components equal model JSON; first-scene pixels, goals and budgets agree across conditions; response totals agree."}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checked_runs"}))


if __name__ == "__main__":
    main()
