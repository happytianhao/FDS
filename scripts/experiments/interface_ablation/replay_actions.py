"""Replay saved model actions for evaluator-only diagnosis; makes no model calls."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    import numpy as np
    from PIL import Image
    from robomme_plus import BenchmarkEnvBuilder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--evaluator-state", action="store_true", help="Log hidden state for diagnosis only; never policy input")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["split"] == "train"
    directory = args.run / ("diagnostic_state_replay" if args.evaluator_state else "diagnostic_replay")
    directory.mkdir(exist_ok=False)
    rows = [json.loads(x) for x in (args.run / "exchange.jsonl").read_text().splitlines()]
    actions = [x["execution"]["action"] for x in rows if "execution" in x]
    builder = BenchmarkEnvBuilder(manifest["task"], dataset="train", action_space="waypoint")
    env = builder.make_env_for_episode(manifest["episode"])
    diagnostics = []
    try:
        obs, info = env.reset()
        for turn, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            entry = {k: info[k] for k in ("status", "failure_reason", "action_error", "error_message", "execution_control_steps", "remaining_steps") if k in info}
            entry.update(turn=turn, action=action, terminated=bool(terminated), truncated=bool(truncated))
            if args.evaluator_state:
                base = env.unwrapped
                values = {}
                for key in ("swing_count", "swing_over_limit", "achieved_list", "match", "task_index", "current_task_idx"):
                    if hasattr(base, key):
                        value = getattr(base, key)
                        values[key] = value.tolist() if hasattr(value, "tolist") else value
                for key in ("target_cube", "target_right", "target_left"):
                    if hasattr(base, key):
                        value = getattr(base, key).pose.p
                        values[key + "_xyz"] = value.detach().cpu().numpy().reshape(-1).tolist()
                entry["evaluator_only_hidden_state"] = values
            if len(obs.get("front_rgb_list", [])):
                for cam in ("front", "wrist"):
                    value = obs[cam + "_rgb_list"][-1]
                    if hasattr(value, "cpu"):
                        value = value.cpu().numpy()
                    Image.fromarray(np.asarray(value).astype(np.uint8)).save(directory / f"t{turn:03d}_{cam}.png")
                value = obs["eef_state_list"][-1]
                if hasattr(value, "cpu"):
                    value = value.cpu().numpy()
                entry["tcp_after"] = np.asarray(value).reshape(-1).tolist()
            diagnostics.append(entry)
            if terminated or truncated:
                break
    finally:
        env.close()
    document = {"source_run": str(args.run), "model_calls": 0, "purpose": "Evaluator-only replay; not policy input", "steps": diagnostics}
    (directory / "diagnostics.json").write_text(json.dumps(document, indent=2, default=str))
    print(json.dumps(document, indent=2, default=str))


if __name__ == "__main__":
    main()
