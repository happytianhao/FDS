"""Command line interface for the training-only FDS meta harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema

from fds.meta_harness import controller, evaluation, evidence, improver
from fds.meta_harness.store import Store, read, safe_path
from fds.task_harness.executor import EXECUTOR_NAME, EXECUTORS


def _internal_target(target: str) -> str:
    return "executor" if target == EXECUTOR_NAME else target


def main(argv=None):
    parser = argparse.ArgumentParser(description="Versioned FDS task-harness meta loop")
    sub = parser.add_subparsers(dest="command", required=True)
    parsers = {}
    for name in (
        "init", "status", "collect", "propose", "stage", "check", "evaluate",
        "promote", "rollback", "cycle",
    ):
        p = sub.add_parser(name)
        p.add_argument("--workspace", type=Path, default=Path("runs/meta_harness/default"))
        parsers[name] = p
    parsers["init"].add_argument("--repo", type=Path, default=Path.cwd())
    parsers["init"].add_argument(
        "--config", type=Path, default=Path("configs/meta_harness/robomme_plus.toml")
    )
    parsers["init"].add_argument("--executor", choices=EXECUTORS, default=EXECUTOR_NAME)
    parsers["init"].add_argument(
        "--benchmark", choices=["robomme_plus", "robomme"], default=None,
        help="Override the benchmark declared by --config",
    )
    parsers["collect"].add_argument("inputs", nargs="+", type=Path)
    parsers["stage"].add_argument("proposal", type=Path)
    parsers["stage"].add_argument("--request", help="Prepared request ID")
    for name in ("check", "evaluate", "promote"):
        parsers[name].add_argument("candidate")
    for name in ("propose", "cycle"):
        parsers[name].add_argument("--target", choices=[EXECUTOR_NAME], default=EXECUTOR_NAME)
    for name in ("propose", "evaluate", "cycle"):
        parsers[name].add_argument(
            "--allow-model", action="store_true", help="Explicitly enable improver model calls"
        )
    for name in ("evaluate", "cycle"):
        parsers[name].add_argument(
            "--allow-experiments", action="store_true",
            help="Explicitly enable scored training-split experiments",
        )
    for name in ("check", "evaluate", "cycle"):
        parsers[name].add_argument("--python", help="Project virtualenv Python used for workers")
    parsers["cycle"].add_argument("--rounds", type=int)
    parsers["rollback"].add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            store = Store.initialize(
                args.workspace, args.repo, args.config, args.executor,
                args.benchmark,
            )
            result = controller.status(store)
        else:
            store = Store(args.workspace)
            with store.lock():
                if args.command == "status":
                    result = controller.status(store)
                elif args.command == "collect":
                    result = {"evidence_ids": evidence.collect(store, args.inputs)}
                elif args.command == "propose":
                    result = improver.propose(
                        store, _internal_target(args.target), allow_model=args.allow_model
                    )
                elif args.command == "stage":
                    request = (
                        read(safe_path(store.root / "requests", args.request) / "request.json")
                        if args.request else None
                    )
                    result = improver.stage_response(store, read(args.proposal), request)
                elif args.command == "check":
                    result = evaluation.check(store, args.candidate, python=args.python)
                elif args.command == "evaluate":
                    _, candidate = store.candidate(args.candidate)
                    if candidate["target"] == "improver":
                        result = controller.evaluate_improver(
                            store, args.candidate, allow_model=args.allow_model,
                            allow_experiments=args.allow_experiments, python=args.python,
                        )
                    else:
                        result = evaluation.evaluate(
                            store, args.candidate, allow_experiments=args.allow_experiments,
                            python=args.python,
                        )
                elif args.command == "promote":
                    result = evaluation.promote(store, args.candidate)
                elif args.command == "rollback":
                    result = store.rollback(args.reason)
                else:
                    result = controller.cycle(
                        store, target=_internal_target(args.target), rounds=args.rounds,
                        allow_model=args.allow_model, allow_experiments=args.allow_experiments,
                        python=args.python,
                    )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "check" and not result["passed"]:
            return 1
        if args.command == "evaluate" and not result["gate"]["passed"]:
            return 1
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, jsonschema.ValidationError) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
