"""Immutable research snapshots and atomic version pointers, not a security sandbox."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.metadata
import json
import shutil
import sys
import tomllib
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from fds.benchmarks import benchmark_spec, benchmark_tasks, validate_executor
from fds.task_harness.executor import EXECUTOR_NAME as EXECUTOR
from fds.task_harness.executor import EXECUTORS

STRATEGY_PATH = "configs/meta_harness/improver.json"
KINDS = ("executor", "improver")


def executor_config(executor, benchmark):
    if executor != EXECUTOR:
        raise ValueError(f"FDS supports only {EXECUTOR}")
    if benchmark not in ("robomme", "robomme_plus"):
        raise ValueError("FDS meta harness supports only robomme and robomme_plus")
    return f"configs/{benchmark}/task_harness.toml"


def smoke_config(executor, benchmark="robomme_plus"):
    return executor_config(executor, benchmark).replace(".toml", "_debug.toml")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controller_hashes():
    return {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def identifier(prefix):
    return prefix + "_" + uuid.uuid4().hex[:16]


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or ".." in p.parts or "\\" in relative or str(p) != relative:
        raise ValueError(f"Invalid relative path: {relative}")
    target = Path(root).joinpath(*p.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes snapshot: {relative}")
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root):
        raise ValueError("Symlinks are not allowed in snapshots")
    return target


def tree_hashes(tree):
    result = {}
    for p in sorted(Path(tree).rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p.is_symlink():
            raise ValueError(f"Snapshot contains a symlink: {p}")
        if p.is_file():
            result[p.relative_to(tree).as_posix()] = file_hash(p)
    return result


def strategy_valid(value):
    if set(value) != {"instructions", "failure_priority", "source_focus"}:
        raise ValueError(
            "Improver strategy may only change instructions, priorities and source focus"
        )
    if not isinstance(value["instructions"], str) or not 20 <= len(value["instructions"]) <= 16000:
        raise ValueError("Improver instructions must contain 20..16000 characters")
    priorities = value["failure_priority"]
    if not isinstance(priorities, list) or sorted(priorities) != sorted(
        ["error", "fail", "planner_finish", "budget_exhausted", "success"]
    ):
        raise ValueError("failure_priority must be a permutation of the five outcome types")
    focus = value["source_focus"]
    if (
        not isinstance(focus, list)
        or len(focus) > 12
        or not all(isinstance(p, str) and p.endswith(".py") for p in focus)
    ):
        raise ValueError("source_focus must contain at most 12 Python paths")


def suite_names(config):
    return ("development", "validation") if "validation" in config else ("development",)


def suite_cases(config, name):
    """Exact case pairs support uneven training curricula without a Cartesian expansion."""
    suite = config[name]
    if "cases" in suite:
        return [(case["task"], case["episode"]) for case in suite["cases"]]
    return [(task, episode) for task in suite["tasks"] for episode in suite["episodes"]]


def promotion_target(config):
    gate = config["gate"]
    return gate.get("suite", "validation"), gate.get(
        "min_success_gain", gate.get("min_validation_gain")
    )


def require_training_splits(config):
    if config.get("benchmark", "robomme") in ("robomme", "robomme_plus"):
        for name in suite_names(config):
            if config[name].get("split") != "train":
                raise ValueError(
                    f"RSI {name} must explicitly use split='train'; "
                    "val/test are reserved for frozen-version evaluation. Initialize a new session."
                )


def config_valid(config):
    if config["executor"] not in EXECUTORS:
        raise ValueError("Unknown executor")
    benchmark = config.get("benchmark", "robomme")
    validate_executor(benchmark, config["executor"])
    for key in (
        "max_rounds",
        "max_changed_files",
        "max_patch_bytes",
        "max_source_chars",
        "max_evidence_cases",
        "check_timeout_seconds",
        "episode_timeout_seconds",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_rounds"] > 20:
        raise ValueError("A cycle is bounded to at most 20 rounds")
    if config["improver"]["backend"] not in ("manual", "codex"):
        raise ValueError("Improver backend must be manual or codex")
    suites = {}
    for name in suite_names(config):
        suite = config[name]
        if "cases" in suite:
            if "tasks" in suite or "episodes" in suite:
                raise ValueError("Configure explicit cases or tasks/episodes, not both")
            cases = suite["cases"]
            if (
                not isinstance(cases, list)
                or not cases
                or any(
                    not isinstance(case, dict) or set(case) != {"task", "episode"} for case in cases
                )
            ):
                raise ValueError(f"Invalid {name} cases")
            tasks = list(
                dict.fromkeys(case["task"] for case in cases if isinstance(case["task"], str))
            )
            episodes = list(
                dict.fromkeys(case["episode"] for case in cases if type(case["episode"]) is int)
            )
            if any(
                not isinstance(case["task"], str) or type(case["episode"]) is not int
                for case in cases
            ):
                raise ValueError(f"Invalid {name} cases")
            if len(set(suite_cases(config, name))) != len(cases):
                raise ValueError(f"Duplicate {name} cases")
        else:
            tasks, episodes = suite["tasks"], suite["episodes"]
        if (
            not tasks
            or len(set(tasks)) != len(tasks)
            or not all(
                isinstance(t, str) and t and "/" not in t and "\\" not in t and t not in (".", "..")
                for t in tasks
            )
        ):
            raise ValueError(f"Invalid {name} tasks")
        if (
            not episodes
            or len(set(episodes)) != len(episodes)
            or any(type(ep) is not int or ep < 0 for ep in episodes)
        ):
            raise ValueError(f"Invalid {name} episodes")
        split = suite.get("split")
        if split is not None and split not in benchmark_spec(benchmark).splits:
            raise ValueError(f"Invalid {name} split for {benchmark}: {split}")
        suites[name] = {(split, t, ep) for t, ep in suite_cases(config, name)}
    if suites["development"] & suites.get("validation", set()):
        raise ValueError("Development and validation task/episode pairs must be disjoint")
    gate = config["gate"]
    target, gain = promotion_target(config)
    if target not in suites:
        raise ValueError("Promotion suite must be a configured development or validation suite")
    if "min_success_gain" in gate and "min_validation_gain" in gate:
        raise ValueError("Configure only one promotion gain threshold")
    if type(gain) is not int or gain < 1:
        raise ValueError("Promotion requires at least one additional success")
    if type(gate["max_regressions"]) is not int or gate["max_regressions"] < 0:
        raise ValueError("max_regressions must be a nonnegative integer")


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.session = read(self.root / "session.json")
        self.config = self.session["config"]
        config_valid(self.config)
        if digest(self.config) != self.session["config_sha256"]:
            raise ValueError("Frozen session config has changed; initialize a new session")

    @classmethod
    def initialize(
        cls, root, repo, config_path, executor=None, benchmark=None, *, asset_workspace=None
    ):
        repo, root = Path(repo).resolve(), Path(root).resolve()
        config = tomllib.loads(Path(config_path).read_text())
        if executor is not None:
            config["executor"] = executor
        if benchmark is not None:
            config["benchmark"] = benchmark
        config_valid(config)
        require_training_splits(config)
        config_path = executor_config(config["executor"], config.get("benchmark", "robomme"))
        method_config = tomllib.loads((repo / config_path).read_text())
        config.setdefault("benchmark", method_config["benchmark"]["name"])
        for suite in suite_names(config):
            config[suite].setdefault("split", method_config["benchmark"].get("split"))
            if not {task for task, _ in suite_cases(config, suite)}.issubset(
                benchmark_tasks(config["benchmark"])
            ):
                raise ValueError(f"Unknown {suite} task for {config['benchmark']}")
        config_valid(config)
        if method_config["method"] != config["executor"]:
            raise ValueError("Executor config identity mismatch")
        root.mkdir(parents=True, exist_ok=False)
        executor_id, improver_id = identifier("e"), identifier("i")
        tree = root / "versions" / executor_id / "tree"
        tree.mkdir(parents=True)
        # Includes dirty working-tree edits; never relies on HEAD or alters the checkout.
        for directory in ("src", "configs", "tests", "scripts", "requirements"):
            source = repo / directory
            if source.exists():
                for path in source.rglob("*"):
                    if path.is_symlink():
                        raise ValueError(f"Cannot snapshot symlink: {path}")
                shutil.copytree(
                    source,
                    tree / directory,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.local.toml"),
                )
        for name in ("pyproject.toml", "README.md", "CLAUDE.md", "LICENSE", "uv.lock"):
            if (repo / name).is_file():
                shutil.copy2(repo / name, tree / name)
        hashes = tree_hashes(tree)
        write(
            tree.parent / "version.json",
            {
                "id": executor_id,
                "target": "executor",
                "parent": None,
                "files": hashes,
                "sha256": digest(hashes),
            },
        )
        strategy = read(repo / STRATEGY_PATH)
        strategy_valid(strategy)
        write(root / "versions" / improver_id / "strategy.json", strategy)
        write(
            root / "versions" / improver_id / "version.json",
            {
                "id": improver_id,
                "target": "improver",
                "parent": None,
                "sha256": digest(strategy),
            },
        )
        from fds.cli import git_state

        write(
            root / "session.json",
            {
                "schema_version": 2,
                "created_utc": datetime.now(UTC).isoformat(),
                "repo": str(repo),
                "asset_workspace": str(Path(asset_workspace).resolve())
                if asset_workspace
                else str(repo),
                "config": config,
                "config_sha256": digest(config),
                "baseline_executor": executor_id,
                "baseline_improver": improver_id,
                "executor_config": config_path,
                "smoke_config": smoke_config(config["executor"], config.get("benchmark", "robomme_plus")),
                "benchmark": method_config["benchmark"]["name"],
                "controller_sha256": controller_hashes(),
                "git": git_state(repo),
                "python": sys.version,
                "packages": {
                    p.metadata["Name"]: p.version for p in importlib.metadata.distributions()
                },
                "scope": "cooperative local research; process isolation is not a security sandbox",
            },
        )
        write(
            root / "state.json",
            {"executor": executor_id, "improver": improver_id, "history": [], "generation": 0},
        )
        store = cls(root)
        store.event("initialized", executor=executor_id, improver=improver_id)
        return store

    @contextmanager
    def lock(self):
        with (self.root / "controller.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another RSI command owns this workspace") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def event(self, event, **values):
        with (self.root / "events.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {"utc": datetime.now(UTC).isoformat(), "event": event, **values},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    def assert_controller(self):
        expected = self.session.get("controller_sha256")
        if expected is not None and expected != controller_hashes():
            raise ValueError(
                "RSI controller changed; initialize a new session for comparable evaluations"
            )
        require_training_splits(self.config)

    @property
    def state(self):
        return read(self.root / "state.json")

    def version(self, version_id):
        root = safe_path(self.root / "versions", version_id)
        meta = read(root / "version.json")
        value = (
            tree_hashes(root / "tree")
            if meta["target"] == "executor"
            else read(root / "strategy.json")
        )
        if digest(value) != meta["sha256"]:
            raise ValueError(f"Version content changed: {version_id}")
        return root, meta

    def candidate(self, candidate_id):
        root = safe_path(self.root / "candidates", candidate_id)
        record = read(root / "candidate.json")
        _, meta = self.version(record["version"])
        if meta["sha256"] != record["version_sha256"]:
            raise ValueError("Candidate version fingerprint mismatch")
        return root, record

    def current_parents(self, record):
        state = self.state
        if (
            record["parent_executor"] != state["executor"]
            or record["parent_improver"] != state["improver"]
        ):
            raise ValueError("Stale candidate: active executor or improver changed")

    def editable(self, path, target):
        if target == "improver":
            return path == STRATEGY_PATH
        method = self.config["executor"]
        prefixes = ["src/fds/task_harness/"] if method == "task_harness" else []
        return (
            path.startswith(tuple(prefixes))
            and path.endswith(".py")
            or path == self.session["executor_config"]
            or method == "task_harness"
            and path == self.session["executor_config"].replace(".toml", "_debug.toml")
        )

    def stage(self, proposal):
        if proposal["target"] not in KINDS:
            raise ValueError("Unknown proposal target")
        self.current_parents(proposal)
        changes = proposal["changes"]
        if not changes or len({c["path"] for c in changes}) > self.config["max_changed_files"]:
            raise ValueError("Proposal has no changes or exceeds file budget")
        if len(json.dumps(changes).encode()) > self.config["max_patch_bytes"]:
            raise ValueError("Proposal exceeds patch byte budget")
        if not proposal.get("hypothesis", "").strip() or not proposal.get("evidence_ids"):
            raise ValueError("A proposal needs a hypothesis and development evidence")
        for eid in proposal["evidence_ids"]:
            evidence = read(safe_path(self.root / "evidence", eid + ".json"))
            if evidence["role"] != "development":
                raise ValueError("Validation traces cannot be proposal evidence")
        target = proposal["target"]
        parent_id = proposal["parent_" + target]
        parent, parent_meta = self.version(parent_id)
        changed = {}
        for change in changes:
            path = change["path"]
            if not self.editable(path, target):
                raise ValueError(f"Protected file: {path}")
            source = (
                safe_path(parent / "tree", path)
                if target == "executor"
                else parent / "strategy.json"
            )
            if file_hash(source) != change["before_sha256"]:
                raise ValueError(f"Patch base hash mismatch: {path}")
            original = source.read_text()
            text = changed.get(path, original)
            before, after = change["before"], change["after"]
            if not before or before == after or text.count(before) != 1:
                raise ValueError(f"Replacement must match exactly once and change content: {path}")
            changed[path] = text.replace(before, after, 1)
        for path, content in changed.items():
            if path.endswith(".py"):
                ast.parse(content, filename=path)
            elif path.endswith(".toml"):
                updated = tomllib.loads(content)
                original = tomllib.loads((parent / "tree" / path).read_text())
                method = self.config["executor"]
                mutable = "task_harness"
                if {k: v for k, v in updated.items() if k != mutable} != {
                    k: v for k, v in original.items() if k != mutable
                }:
                    raise ValueError(
                        "Benchmark, planner, policy, budget and method config are protected"
                    )
                if method == "task_harness":
                    from fds.task_harness.skills import validate_skill_changes

                    validate_skill_changes(
                        original.get("task_harness", {}).get("skills", []),
                        updated.get("task_harness", {}).get("skills", []),
                        [
                            read(safe_path(self.root / "evidence", eid + ".json"))
                            for eid in proposal["evidence_ids"]
                        ],
                    )
            else:
                strategy_valid(json.loads(content))
        version_id = identifier("e" if target == "executor" else "i")
        destination = self.root / "versions" / version_id
        if target == "executor":
            shutil.copytree(parent / "tree", destination / "tree")
            for path, content in changed.items():
                safe_path(destination / "tree", path).write_text(content)
            hashes = tree_hashes(destination / "tree")
            sha = digest(hashes)
        else:
            write(destination / "strategy.json", json.loads(changed[STRATEGY_PATH]))
            hashes, sha = None, digest(read(destination / "strategy.json"))
        write(
            destination / "version.json",
            {
                "id": version_id,
                "target": target,
                "parent": parent_id,
                "sha256": sha,
                "files": hashes,
            },
        )
        candidate_id = identifier("c")
        record = {
            **proposal,
            "id": candidate_id,
            "version": version_id,
            "version_sha256": sha,
            "base_sha256": parent_meta["sha256"],
        }
        write(self.root / "candidates" / candidate_id / "candidate.json", record)
        self.event("candidate_staged", candidate=candidate_id, target=target, version=version_id)
        return record

    def rollback(self, reason):
        if not reason.strip():
            raise ValueError("Rollback needs a reason")
        state = self.state
        if not state["history"]:
            raise ValueError("No promoted version to roll back")
        transition = state["history"].pop()
        self.version(transition["previous"])
        state[transition["target"]] = transition["previous"]
        state["generation"] += 1
        write(self.root / "state.json", state)
        self.event("rolled_back", reason=reason, **transition)
        return state
