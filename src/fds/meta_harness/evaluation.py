"""Fixed tests, paired evaluation and promotion gates outside the editable surface."""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from fds.meta_harness.store import (
    digest,
    identifier,
    promotion_target,
    read,
    safe_path,
    strategy_valid,
    suite_cases,
    suite_names,
    write,
)


def subprocess_check(command, directory, timeout, cwd):
    command = list(command)
    if "/" in command[0] and not Path(command[0]).is_absolute():
        command[0] = str(Path(command[0]).absolute())
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Loading unrelated pytest plugins can execute project-local hooks outside the snapshot.
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with (directory / "stdout.log").open("w") as out, (directory / "stderr.log").open("w") as err:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=out, stderr=err, env=environment, start_new_session=True
        )
        expired = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    result = {
        "command": command,
        "returncode": process.returncode,
        "timeout": expired,
        "seconds": time.monotonic() - started,
        "passed": not expired and process.returncode == 0,
    }
    write(directory / "process.json", result)
    return result


def run_episode(
    store, version, config, task, episode, output, *, allow_experiments=False, python=None
):
    store.assert_controller()
    if (
        config["benchmark"]["name"] in ("robomme", "robomme_plus")
        and config["benchmark"].get("split") != "train"
    ):
        raise ValueError("RSI workers may only run split='train'; no experiment was started")
    root, meta = store.version(version)
    job = {
        "tree": str(root / "tree"),
        "output": str(output),
        "config": config,
        "task": task,
        "episode": episode,
        "version": version,
        "source_sha256": meta["sha256"],
        "config_sha256": digest(config),
        "allow_experiments": allow_experiments,
        "workspace": store.session.get("asset_workspace", store.session["repo"]),
        "controller_sha256": store.session.get("controller_sha256"),
    }
    job_file = output.parent / (output.name + ".job.json")
    write(job_file, job)
    worker = Path(__file__).with_name("worker.py")
    result = subprocess_check(
        [python or sys.executable, "-I", str(worker), str(job_file)],
        output.parent / (output.name + ".process"),
        store.config["episode_timeout_seconds"],
        root / "tree",
    )
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
    else:
        receipt = {
            k: job[k] for k in ("version", "source_sha256", "config_sha256", "task", "episode")
        }
        receipt.update(
            success=False,
            official_status="error",
            error={
                "type": "WorkerFailure",
                "message": f"exit={result['returncode']}, timeout={result['timeout']}",
            },
            validation_only=config["benchmark"]["name"] == "mock",
            steps=0,
            planner_calls=0,
        )
    if not result["passed"]:
        receipt.update(
            success=False,
            official_status="error",
            error=receipt.get("error") or {"type": "WorkerFailure", "message": str(result)},
        )
    for key in ("version", "source_sha256", "config_sha256", "task", "episode"):
        if receipt.get(key) != job[key]:
            raise ValueError(f"Worker receipt identity mismatch: {key}")
    store.version(version)
    write(receipt_path, receipt)
    return receipt


def check(store, candidate_id, *, python=None):
    store.assert_controller()
    directory, record = store.candidate(candidate_id)
    store.current_parents(record)
    root, meta = store.version(record["version"])
    check_id = identifier("check")
    output = directory / check_id
    output.mkdir()
    results = []
    if record["target"] == "improver":
        strategy_valid(read(root / "strategy.json"))
        results.append({"name": "strategy_schema", "passed": True})
    else:
        tree = root / "tree"
        for path in sorted(tree.rglob("*.py")):
            ast.parse(path.read_text(), filename=str(path))
        results.append({"name": "python_syntax", "passed": True})
        baseline, _ = store.version(store.session["baseline_executor"])
        tests = baseline / "tree" / "tests"
        bootstrap = (
            "import sys; sys.dont_write_bytecode=True; "
            f"sys.path.insert(0, {str(tree / 'src')!r}); "
            "import pytest; "
            f"raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', {str(tests)!r}]))"
        )
        results.append(
            {
                "name": "frozen_regression_tests",
                **subprocess_check(
                    [python or sys.executable, "-I", "-c", bootstrap],
                    output / "pytest",
                    store.config["check_timeout_seconds"],
                    tree,
                ),
            }
        )
        # The offline check must not instantiate either benchmark. It executes
        # the candidate's contract smoke script inside its frozen source tree.
        smoke = tree / "scripts" / "smoke_fds.py"
        smoke_check = subprocess_check(
            [python or sys.executable, "-I", str(smoke)],
            output / "smoke",
            store.config["check_timeout_seconds"],
            tree,
        )
        results.append(
            {
                "name": "offline_contract_wiring",
                "passed": smoke_check["passed"],
            }
        )
    store.version(record["version"])
    report = {
        "id": check_id,
        "candidate": candidate_id,
        "source_sha256": meta["sha256"],
        "passed": all(r["passed"] for r in results),
        "validation_only": True,
        "checks": results,
    }
    write(output / "report.json", report)
    write(directory / "checks.json", report)
    store.event("candidate_checked", candidate=candidate_id, passed=report["passed"])
    return report


def require_checks(store, candidate_id):
    store.assert_controller()
    directory, record = store.candidate(candidate_id)
    checks = read(directory / "checks.json")
    if not checks["passed"] or checks["source_sha256"] != record["version_sha256"]:
        raise ValueError("Candidate must pass checks for its exact source version")
    return record


def aggregate_pair(parent, candidate, expected):
    def indexed(rows):
        result = {(r["task"], r["episode"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate evaluation case")
        return result

    left, right = indexed(parent), indexed(candidate)
    complete = set(left) == set(right) == set(expected)
    clean = complete and all(
        not r.get("error")
        and not r.get("validation_only", True)
        and r.get("official_status") in ("success", "fail", "ongoing", "timeout")
        and r.get("success") is (r.get("official_status") == "success")
        for r in [*parent, *candidate]
    )

    # Do not trust a success boolean if it disagrees with the independently recorded status.
    def success(r):
        return bool(
            r
            and r.get("success") is True
            and r.get("official_status") == "success"
            and not r.get("error")
        )

    gains = [list(k) for k in expected if success(right.get(k)) and not success(left.get(k))]
    regressions = [list(k) for k in expected if success(left.get(k)) and not success(right.get(k))]
    return {
        "planned": len(expected),
        "complete": complete,
        "clean_real_evaluation": clean,
        "parent_successes": sum(success(left.get(k)) for k in expected),
        "candidate_successes": sum(success(right.get(k)) for k in expected),
        "net_gain": len(gains) - len(regressions),
        "gains": gains,
        "regressions": regressions,
    }


def gate(config, suites):
    reasons = []
    for name in suite_names(config):
        result = suites[name]
        if not result["clean_real_evaluation"]:
            reasons.append(f"{name}: incomplete, errored or mock evaluation")
        if len(result["regressions"]) > config["gate"]["max_regressions"]:
            reasons.append(f"{name}: too many paired regressions")
        if result["net_gain"] < 0:
            reasons.append(f"{name}: total successes decreased")
    target, gain = promotion_target(config)
    if suites[target]["net_gain"] < gain:
        reasons.append(f"{target}: insufficient improvement")
    return {"passed": not reasons, "reasons": reasons}


def evaluate(store, candidate_id, *, allow_experiments=False, python=None):
    # This check precedes reading candidates, constructing models or launching any process.
    if not allow_experiments:
        raise ValueError("Evaluation requires --allow-experiments; no experiment was started")
    directory, record = store.candidate(candidate_id)
    store.current_parents(record)
    require_checks(store, candidate_id)
    if record["target"] != "executor":
        raise ValueError("Improver candidates require the meta-evaluation procedure")
    evaluation_id = identifier("eval")
    output = directory / evaluation_id
    output.mkdir()
    rows = {"parent": {}, "candidate": {}}
    for arm, version in (("parent", record["parent_executor"]), ("candidate", record["version"])):
        root, _ = store.version(version)
        config = tomllib.loads((root / "tree" / store.session["executor_config"]).read_text())
        if (
            config["benchmark"]["name"] == "mock"
            or config["planner"]["backend"] == "mock"
            or config["run"].get("validation_only")
        ):
            raise ValueError(
                "Mock/validation-only configurations cannot enter promotion evaluation"
            )
        for suite in suite_names(store.config):
            config = {
                **config,
                "benchmark": {
                    **config["benchmark"],
                    "split": store.config[suite].get("split", config["benchmark"].get("split")),
                },
            }
            rows[arm][suite] = []
            for task, episode in suite_cases(store.config, suite):
                # Same fixed config/budget/seed per paired case; mutable harness options may differ.
                receipt = run_episode(
                    store,
                    version,
                    config,
                    task,
                    episode,
                    output / arm / suite / task / f"episode_{episode:04d}",
                    allow_experiments=True,
                    python=python,
                )
                rows[arm][suite].append(receipt)
                write(output / "progress.json", rows)
    suites = {}
    for name in suite_names(store.config):
        expected = suite_cases(store.config, name)
        suites[name] = aggregate_pair(rows["parent"][name], rows["candidate"][name], expected)
    report = {
        "id": evaluation_id,
        "candidate": candidate_id,
        "target": "executor",
        "version_sha256": record["version_sha256"],
        "config_sha256": store.session["config_sha256"],
        "parent_executor": record["parent_executor"],
        "parent_improver": record["parent_improver"],
        "suites": suites,
        "gate": gate(store.config, suites),
        "rows": rows,
        "validation_only": False,
    }
    save_evaluation(store, directory, output, report)
    return report


def save_evaluation(store, directory, output, report):
    write(output / "report.json", report)
    write(
        directory / "evaluation.json",
        {"report": str(output.relative_to(directory) / "report.json"), "sha256": digest(report)},
    )
    store.event(
        "candidate_evaluated",
        candidate=report["candidate"],
        target=report["target"],
        gate=report["gate"],
    )


def load_evaluation(directory, reference=None):
    reference = read(directory / "evaluation.json") if reference is None else reference
    report = read(safe_path(directory, reference["report"]))
    if digest(report) != reference["sha256"]:
        raise ValueError("Evaluation report changed")
    return report, reference


def verify_executor_report(store, candidate_id, report):
    _, record = store.candidate(candidate_id)
    if (
        report["target"] != "executor"
        or report["candidate"] != candidate_id
        or report["version_sha256"] != record["version_sha256"]
        or report["config_sha256"] != store.session["config_sha256"]
        or report["parent_executor"] != record["parent_executor"]
        or report["parent_improver"] != record["parent_improver"]
        or report["validation_only"]
    ):
        raise ValueError("Executor evaluation identity mismatch")
    suites = {}
    for name in suite_names(store.config):
        expected = suite_cases(store.config, name)
        suites[name] = aggregate_pair(
            report["rows"]["parent"][name], report["rows"]["candidate"][name], expected
        )
    if suites != report["suites"]:
        raise ValueError("Raw evaluation receipts do not match the stored result")
    return suites


def promote(store, candidate_id):
    directory, record = store.candidate(candidate_id)
    store.current_parents(record)
    require_checks(store, candidate_id)
    report, reference = load_evaluation(directory)
    if (
        not report["gate"]["passed"]
        or report["validation_only"]
        or report["version_sha256"] != record["version_sha256"]
        or report["config_sha256"] != store.session["config_sha256"]
        or report["candidate"] != candidate_id
        or report["parent_executor"] != record["parent_executor"]
        or report["parent_improver"] != record["parent_improver"]
    ):
        raise ValueError("Candidate has no eligible, current, non-mock improvement evaluation")
    store.version(record["parent_executor"])
    store.version(record["parent_improver"])
    # Recompute the gate; a stored 'passed' flag alone is not sufficient.
    if record["target"] == "executor":
        suites = verify_executor_report(store, candidate_id, report)
        if not gate(store.config, suites)["passed"]:
            raise ValueError("Raw evaluation receipts do not support promotion")
    else:
        children = {}
        for arm in ("parent", "candidate"):
            cid = report["offspring"][arm]
            child_directory, child_record = store.candidate(cid)
            if any(child_record[k] != record[k] for k in ("parent_executor", "parent_improver")):
                raise ValueError("Meta-evaluation offspring used a different base version")
            child, _ = load_evaluation(child_directory, report["offspring_evaluations"][arm])
            verify_executor_report(store, cid, child)
            children[arm] = child
        suites = {}
        for name in suite_names(store.config):
            expected = suite_cases(store.config, name)
            suites[name] = aggregate_pair(
                children["parent"]["rows"]["candidate"][name],
                children["candidate"]["rows"]["candidate"][name],
                expected,
            )
        if (
            suites != report["suites"]
            or not gate(store.config, suites)["passed"]
            or not gate(store.config, children["candidate"]["suites"])["passed"]
        ):
            raise ValueError("Meta-evaluation offspring do not support promotion")
    state = store.state
    transition = {
        "target": record["target"],
        "previous": state[record["target"]],
        "version": record["version"],
        "candidate": candidate_id,
        "evaluation_sha256": reference["sha256"],
    }
    state[record["target"]] = record["version"]
    state["history"].append(transition)
    state["generation"] += 1
    write(store.root / "state.json", state)
    store.event("promoted", **transition)
    return state
