"""Paired train-only waypoint experiments, independent of the FDS campaign.

Uses the upstream Astra interface prompt and the installed official simulator.
The model receives only RGB, measured robot state, calibration and public feedback.
Codex is used solely as a tool-disabled JSON inference transport. All retained
observations and visible answers are explicitly replayed in each fresh request;
this is NOT an exact reproduction of the native Responses conversation backend.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / ".cache/robomme_repo_review_20261110/embodied-astra-robomme/harness/astra_waypoint_eval_v3.py"
VARIANTS = ("full", "no_dynamics", "no_history", "sparse16", "no_calibration", "no_lr", "verified_progress")
PROMPT_PROTOCOL = "codex_base_instructions_v1"
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "description": {"type": "string"},
        "waypoint": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6}, {"type": "null"}]},
        "gripper": {"type": "integer", "enum": [-1, 0, 1]},
    },
    "required": ["description", "waypoint", "gripper"],
}
PUBLIC_FEEDBACK = ("action_error", "error_message", "exception_type", "error_penalty_steps", "trial_reset", "trial_index", "trials_remaining", "message")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interface_prompt(task, variant, max_steps):
    constants = {}
    for node in ast.parse(UPSTREAM.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("SYSTEM", "MODEL_NOTES", "LR_NOTE"):
                    constants[target.id] = ast.literal_eval(node.value)
    notes = "" if variant == "no_dynamics" else constants["MODEL_NOTES"]
    if variant != "no_lr":
        notes += constants["LR_NOTE"]
    stick = "In this task the robot holds a stick instead of a gripper: the gripper command is ignored.\n" if task in ("PatternLock", "RouteStick") else ""
    stride = 16 if variant == "sparse16" else 3
    prompt = constants["SYSTEM"].replace("STICK_NOTE\n", stick).replace("MODEL_NOTES\n", notes)
    prompt = prompt.replace("FRAME_STRIDE", str(stride)).replace("FRAME_DT", f"{stride / 20:g}")
    prompt = prompt.replace("after 1300 control steps.", f"after {max_steps} execution control steps.")
    if variant == "no_calibration":
        start = prompt.index("; and the camera calibration:")
        end = prompt.index("You get no depth", start)
        prompt = prompt[:start] + ". " + prompt[end:]
    if variant == "verified_progress":
        prompt += "\nCount and record verified physical events, rather than issued commands. Before the next dependent action, compare the newest observations with the intended outcome. If the outcome is ambiguous, obtain another observation through a safe action and revise your estimate. An object's identity and current location are distinct: maintain the identity through motion and occlusion. Movement takes time, so consider where a moving object will be when contact occurs.\n"
    return prompt


def numpy(value):
    import numpy as np
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def fmt(value, digits=4):
    import numpy as np
    return np.round(numpy(value).astype(float), digits).tolist()


def observation(obs, info, directory, turn, variant):
    import numpy as np
    from PIL import Image
    fr, wr = obs["front_rgb_list"], obs["wrist_rgb_list"]
    count = len(fr)
    if count < 1:
        raise ValueError("Observation has no frames")
    stride = 16 if variant == "sparse16" else 3
    idx = sorted(set(range(0, count, stride)) | {count - 1})
    if turn == 0 and len(idx) > 749:
        idx = [i * (count - 1) // 748 for i in range(749)]
    images = []
    for i in idx:
        path = directory / f"t{turn:03d}_front_{i:05d}.png"
        Image.fromarray(numpy(fr[i]).astype(np.uint8)).save(path)
        images.append({"path": str(path), "camera": "front", "source_index": i, "sha256": sha(path)})
    path = directory / f"t{turn:03d}_wrist.png"
    Image.fromarray(numpy(wr[-1]).astype(np.uint8)).save(path)
    images.append({"path": str(path), "camera": "wrist", "source_index": len(wr) - 1, "sha256": sha(path)})
    state = {
        "tcp_pose_xyz_rpy": fmt(numpy(obs["eef_state_list"][-1]).reshape(-1)),
        "finger_positions_m": fmt(numpy(obs["gripper_state_list"][-1]).reshape(-1)),
        "elapsed_control_steps": int(info.get("execution_control_steps", 0)),
        "remaining_steps": int(info["remaining_steps"]),
    }
    if variant != "no_calibration":
        for cam in ("front", "wrist"):
            state[cam + "_camera_K"] = fmt(numpy(info[cam + "_camera_intrinsic"]).reshape(3, 3), 2)
            state[cam + "_camera_extrinsic_world_to_camera"] = fmt(numpy(obs[cam + "_camera_extrinsic_list"][-1]).reshape(3, 4))
    return {"turn": turn, "source_frame_count": count, "front_indices": idx, "images": images,
            "state": state, "feedback": {k: info[k] for k in PUBLIC_FEEDBACK if k in info}}


def request_content(goal, history):
    lines = ["Task goal: " + goal,
             "The supplied images are in the exact order listed below. Observation groups and earlier answers are a chronological transcript. Only the final group is current. Return your next single action as JSON."]
    paths = []
    for item in history:
        begin = len(paths) + 1
        paths.extend(Path(im["path"]) for im in item["images"])
        lines.append(f"OBSERVATION TURN {item['turn']}: attached images {begin}..{len(paths)-1} are front frames {item['front_indices']} of {item['source_frame_count']}; image {len(paths)} is the current wrist view for this turn.")
        lines.append("Robot state and calibration: " + json.dumps(item["state"]))
        if item["feedback"]:
            lines.append("Public execution feedback: " + json.dumps(item["feedback"]))
        if "answer" in item:
            lines.append("Previous assistant action: " + json.dumps(item["answer"]))
    return "\n\n".join(lines), paths


class InferenceError(RuntimeError):
    pass


def infer(prompt, images, directory, timeout, *, system):
    import jsonschema
    directory.mkdir()
    write_json(directory / "schema.json", SCHEMA)
    (directory / "prompt.txt").write_text(prompt)
    instructions_path = directory / "system_prompt.txt"
    instructions_path.write_text(system)
    write_json(directory / "input_images.json", [{"path": str(p), "sha256": sha(p)} for p in images])
    environment = os.environ.copy()
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"):
        environment.pop(key, None)
    provider = "interface_ablation_inference"
    with tempfile.TemporaryDirectory(prefix="interface-ablation-inference-") as work:
        cmd = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "workspace-write", "--json", "--model", "gpt-6-astra", "--cd", work, "--output-schema", str(directory / "schema.json"), "-o", str(directory / "answer.json")]
        settings = {"forced_login_method": '"chatgpt"', "model_provider": json.dumps(provider),
                    "model_reasoning_effort": '"medium"', "approval_policy": '"never"',
                    "web_search": '"disabled"', "project_doc_max_bytes": "0",
                    "model_instructions_file": json.dumps(str(instructions_path.resolve())),
                    "skills.include_instructions": "false"}
        for key, value in {"name": '"OpenAI one-pass"', "requires_openai_auth": "true", "wire_api": '"responses"', "request_max_retries": "0", "stream_max_retries": "0", "supports_websockets": "false"}.items():
            settings[f"model_providers.{provider}.{key}"] = value
        for key, value in settings.items():
            cmd.extend(["-c", key + "=" + value])
        for feature in ("unbounded_connection_retries", "view_image", "shell_tool", "unified_exec", "sleep_tool", "collaboration_modes", "multi_agent", "apps", "plugins", "browser_use", "computer_use", "image_generation", "hooks", "memories", "skill_search"):
            cmd.extend(["--disable", feature])
        for path in images:
            cmd.extend(["--image", str(path)])
        cmd.extend(["--", "-"])
        write_json(directory / "requested_settings.json", {"model": "gpt-6-astra", "effort": "medium", "transport": "codex_explicit_replay", "image_count": len(images), "timeout": timeout,
                   "prompt_protocol": PROMPT_PROTOCOL, "skills_include_instructions": False,
                   "base_instructions_sha256": sha(instructions_path)})
        started = time.monotonic()
        with (directory / "events.jsonl").open("w") as output, (directory / "stderr.log").open("w") as err:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=output, stderr=err, text=True, env=environment, start_new_session=True)
            try:
                proc.communicate(prompt, timeout=timeout)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line.strip()]
        completed = [e for e in events if e.get("type") == "turn.completed"]
        errors = [e for e in events if e.get("type") in ("error", "turn.failed")]
        allowed = ("", "reasoning", "agent_message")
        tools = [e for e in events if e.get("item", {}).get("type", "") not in allowed]
        stats = {"seconds": time.monotonic() - started, "returncode": proc.returncode,
                 "usage": completed[-1].get("usage") if completed else None,
                 "completed": bool(completed), "errors": errors, "unexpected_items": tools}
        write_json(directory / "stats.json", stats)
        if proc.returncode or not completed or errors or tools or not (directory / "answer.json").exists():
            raise InferenceError(f"Inference failed: returncode={proc.returncode}, errors={errors}, unexpected_items={tools}")
        answer = json.loads((directory / "answer.json").read_text())
        jsonschema.validate(answer, SCHEMA)
        return answer, stats


def run(args):
    import numpy as np
    from robomme_plus import BenchmarkEnvBuilder
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    builder = BenchmarkEnvBuilder(args.task, dataset="train", action_space="waypoint", gui_render=False)
    metadata = builder.get_episode_metadata(args.episode)
    revision = subprocess.check_output(["git", "-C", str(ROOT / "third_party/robomme_plus"), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"benchmark": "robomme_plus", "benchmark_commit": revision, "split": "train", "task": args.task,
                "episode": args.episode, "variant": args.variant, "seed": metadata["seed"], "difficulty": metadata["difficulty"],
                "max_calls": args.max_calls, "timeout": args.timeout, "transport": "codex_explicit_replay",
                "model": "gpt-6-astra", "effort": "medium", "runner_sha256": sha(Path(__file__)), "upstream_prompt_sha256": sha(UPSTREAM),
                "prompt_protocol": PROMPT_PROTOCOL, "skills_include_instructions": False,
                "budget_note": "Pilot call cap differs from upstream 150; official environment execution limit is unchanged.",
                "evaluation_only_metadata": metadata}
    write_json(out / "manifest.json", manifest)
    started = time.monotonic()
    env, calls, history, status = None, 0, [], "unknown"
    result = {"success": None, "outcome": "running"}
    try:
        options = {f"include_{cam}_camera_{field}": True for cam in ("front", "wrist") for field in ("intrinsic", "extrinsic")}
        env = builder.make_env_for_episode(args.episode, **options)
        obs, info = env.reset()
        goal = info["task_goal"][0]
        system = interface_prompt(args.task, args.variant, int(info["max_steps"]))
        (out / "system_prompt.txt").write_text(system)
        result.update(goal=goal, official_max_steps=int(info["max_steps"]))
        if args.resume_from:
            source = args.resume_from.resolve()
            previous = json.loads((source / "manifest.json").read_text())
            for key in ("task", "episode", "variant", "benchmark_commit", "model", "effort", "upstream_prompt_sha256", "max_calls", "timeout", "prompt_protocol", "skills_include_instructions"):
                if previous.get(key) != manifest[key]:
                    raise ValueError("Resume mismatch: " + key)
            if (source / "system_prompt.txt").read_text() != system:
                raise ValueError("Resume requires an identical system prompt")
            prior_result = json.loads((source / "result.json").read_text())
            if prior_result["outcome"] != "infrastructure_error":
                raise ValueError("Only infrastructure interruptions can be resumed")
            records = [json.loads(line) for line in (source / "exchange.jsonl").read_text().splitlines()]
            executions = [record["execution"] for record in records if "execution" in record]
            answers = [record["response"] for record in records if "response" in record]
            requests = [record["request"] for record in records if "request" in record]
            if len(answers) != len(executions) or len(requests) != len(answers) + 1:
                raise ValueError("Resume requires a failed inference before an action")
            # The final request failed before producing an answer or action.
            for answer, executed, requested in zip(answers, executions, requests, strict=False):
                current = requested["observation"]
                current["answer"] = answer
                history.append(current)
                obs, _, terminated, truncated, info = env.step(np.array(executed["action"]))
                if terminated or truncated or int(info["execution_control_steps"]) != executed["execution_control_steps"]:
                    raise ValueError("Replay changed episode progression")
            calls = len(executions)
            status = str(info.get("status", "unknown"))
            recovered = observation(obs, info, out / "frames", calls, args.variant)
            expected = requests[-1]["observation"]
            if recovered["state"] != expected["state"] or [i["sha256"] for i in recovered["images"]] != [i["sha256"] for i in expected["images"]]:
                raise ValueError("Replay did not reproduce the interrupted observation exactly")
            with (out / "exchange.jsonl").open("w") as f:
                for record in records[:-1]:
                    f.write(json.dumps(record) + "\n")
            manifest.update(resume_source=str(source), replayed_actions=calls, replay_observation_match="exact state and PNG SHA256")
            write_json(out / "manifest.json", manifest)
            print(json.dumps({"event": "resumed", "actions_replayed": calls, "observation_verified": True}), flush=True)
        for turn in range(calls, args.max_calls):
            current = observation(obs, info, out / "frames", turn, args.variant)
            history.append(current)
            selected = history[-1:] if args.variant == "no_history" else history
            prompt, paths = request_content(goal, selected)
            if len(paths) > 1500:
                result.update(success=False, outcome="pilot_image_limit")
                break
            with (out / "exchange.jsonl").open("a") as f:
                f.write(json.dumps({"request": {"turn": turn, "observation": current, "retained_turns": [h["turn"] for h in selected], "images": len(paths)}}) + "\n")
            answer, stats = infer(prompt, paths, out / f"call_{turn:03d}", args.timeout, system=system)
            calls += 1
            current["answer"] = answer
            with (out / "exchange.jsonl").open("a") as f:
                f.write(json.dumps({"response": answer, "stats": stats}) + "\n")
            if answer["waypoint"] is None:
                result.update(success=False, outcome="model_declared_complete")
                break
            action = np.array([*answer["waypoint"], answer["gripper"]], dtype=np.float64)
            if not np.isfinite(action).all():
                raise ValueError("Nonfinite action")
            obs, reward, terminated, truncated, info = env.step(action)
            status = str(info.get("status", "unknown"))
            step = {"turn": turn, "action": action.tolist(), "status": status,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "execution_control_steps": int(info.get("execution_control_steps", 0)),
                    "remaining_steps": int(info.get("remaining_steps", 0)),
                    "feedback": {k: info[k] for k in PUBLIC_FEEDBACK if k in info}}
            if "failure_reason" in info:
                step["evaluator_failure_reason"] = info["failure_reason"]
            with (out / "exchange.jsonl").open("a") as f:
                f.write(json.dumps({"execution": step}) + "\n")
            print(json.dumps({"turn": turn, "status": status, "description": answer["description"], "images": len(paths)}), flush=True)
            if terminated or truncated:
                result.update(success=status == "success", outcome="environment_done")
                break
            if not len(obs.get("front_rgb_list", [])):
                raise RuntimeError("No-frame observation: stopping pilot instead of fabricating an observation")
        else:
            result.update(success=False, outcome="pilot_call_limit")
    except (InferenceError, subprocess.TimeoutExpired) as exc:
        result.update(success=None, outcome="infrastructure_error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    except Exception as exc:
        result.update(success=None, outcome="runner_or_environment_error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if env is not None:
            env.close()
        result.update(calls=calls, official_status=status, wall_seconds=time.monotonic() - started,
                      task=args.task, episode=args.episode, variant=args.variant)
        write_json(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 2 if result["success"] is None else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    raise SystemExit(run(args))
