"""Privileged post-hoc diagnosis, never a policy or leaderboard evaluation.

Replay one saved SwingXtimes rollout with a single change: lower the carrying
waypoints by 2 cm. This intervention was chosen after inspecting evaluator state.
It tests the diagnosed height failure; it cannot establish a prompt improvement.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    import numpy as np
    from robomme_plus import BenchmarkEnvBuilder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["task"] == "SwingXtimes" and manifest["split"] == "train"
    records = [json.loads(line) for line in (args.run / "exchange.jsonl").read_text().splitlines()]
    actions = [r["execution"]["action"] for r in records if "execution" in r]
    assert len(actions) == 19
    out = args.run / "height_probe"
    out.mkdir(exist_ok=False)
    env = BenchmarkEnvBuilder("SwingXtimes", dataset="train", action_space="waypoint").make_env_for_episode(manifest["episode"])
    rows = []
    try:
        env.reset()
        for turn, original in enumerate(actions):
            action = list(original)
            if 3 <= turn <= 14:
                action[2] -= 0.02
            obs, _, terminated, truncated, info = env.step(np.array(action))
            rows.append({"turn": turn, "action": action, "status": info["status"],
                         "execution_control_steps": int(info["execution_control_steps"]),
                         "evaluator_only_swing_count": env.unwrapped.swing_count,
                         "evaluator_only_cube_xyz": env.unwrapped.target_cube.pose.p.detach().cpu().numpy().reshape(-1).tolist(),
                         "failure_reason": info.get("failure_reason")})
            if terminated or truncated:
                break
        result = {"purpose": __doc__, "model_calls": 0, "source_run": str(args.run),
                  "intervention": "z -= 0.02 m for saved turns 3..14; all other action components unchanged",
                  "success": rows[-1]["status"] == "success", "steps": rows}
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
