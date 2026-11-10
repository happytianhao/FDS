"""No-model action intervention: hold after release, then perform saved retreat.

This is a post-hoc mechanism probe on a successful training rollout, not a new
policy episode or a leaderboard result. Only the final retreat is delayed.
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
    assert manifest["task"] == "VideoPlaceOrder" and manifest["split"] == "train"
    records = [json.loads(line) for line in (args.run / "exchange.jsonl").read_text().splitlines()]
    actions = [r["execution"]["action"] for r in records if "execution" in r]
    out = args.run / "retreat_probe"
    out.mkdir(exist_ok=False)
    env = BenchmarkEnvBuilder("VideoPlaceOrder", dataset="train", action_space="waypoint").make_env_for_episode(manifest["episode"])
    rows = []

    def step(action, phase):
        obs, _, terminated, truncated, info = env.step(np.array(action))
        rows.append({"phase": phase, "status": info["status"],
                     "execution_control_steps": int(info["execution_control_steps"]),
                     "remaining_steps": int(info["remaining_steps"]),
                     "terminated": bool(terminated), "truncated": bool(truncated)})
        return obs, terminated or truncated

    try:
        env.reset()
        for action in actions[:-1]:
            obs, done = step(action, "saved_prefix")
            assert not done
        release_step = rows[-1]["execution_control_steps"]
        holds = 0
        while rows[-1]["execution_control_steps"] - release_step < 40:
            # Command the measured pose, with unchanged gripper, using the public interface.
            pose = obs["eef_state_list"][-1].reshape(-1)
            if hasattr(pose, "cpu"):
                pose = pose.cpu().numpy()
            obs, done = step([*pose.tolist(), 0], "hold_after_release")
            holds += 1
            if done:
                break
            if holds > 80:
                raise RuntimeError("Hold did not advance sufficient simulated time")
        if not done:
            obs, done = step(actions[-1], "saved_retreat")
        result = {"purpose": "Post-hoc action intervention, no model calls", "source_run": str(args.run),
                  "hold_commands": holds, "steps": rows}
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
