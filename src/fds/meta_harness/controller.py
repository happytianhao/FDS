"""Bounded improvement cycles and equal-budget improvement-of-improver trials."""

from __future__ import annotations

from fds.meta_harness import evaluation, evidence, improver
from fds.meta_harness.store import identifier, read, suite_cases, suite_names, write


def evaluate_improver(
    store, candidate_id, *, allow_model=False, allow_experiments=False, python=None, model=None
):
    if not allow_model or not allow_experiments:
        raise ValueError("Meta-evaluation requires both --allow-model and --allow-experiments")
    directory, record = store.candidate(candidate_id)
    store.current_parents(record)
    evaluation.require_checks(store, candidate_id)
    if record["target"] != "improver":
        raise ValueError("Meta-evaluation requires an improver candidate")
    evaluation_id = identifier("meta")
    output = directory / evaluation_id
    output.mkdir()
    # A fixed pool, independent of either strategy's ranking, gives both the same case budget.
    pool = sorted(p.stem for p in (store.root / "evidence").glob("*.json"))
    pool = pool[: store.config["max_evidence_cases"]]
    offspring, reports = {}, {}
    # Prepare both contexts before producing either offspring: the second improver
    # must not see the first one's proposal, checks or evaluation as extra feedback.
    prepared = {
        arm: improver.prepare(store, strategy_id=strategy, evidence_ids=pool)
        for arm, strategy in (
            ("parent", record["parent_improver"]),
            ("candidate", record["version"]),
        )
    }
    for arm in ("parent", "candidate"):
        request, request_directory = prepared[arm]
        child = improver.generate(store, request, request_directory, model=model)
        offspring[arm] = child["id"]
        check = evaluation.check(store, child["id"], python=python)
        if check["passed"]:
            reports[arm] = evaluation.evaluate(
                store, child["id"], allow_experiments=True, python=python
            )
        write(
            output / "progress.json",
            {"offspring": offspring, "completed_evaluations": list(reports)},
        )
    suites = {}
    if len(reports) == 2:
        for name in suite_names(store.config):
            expected = suite_cases(store.config, name)
            suites[name] = evaluation.aggregate_pair(
                reports["parent"]["rows"]["candidate"][name],
                reports["candidate"]["rows"]["candidate"][name],
                expected,
            )
        decision = evaluation.gate(store.config, suites)
        if not reports["candidate"]["gate"]["passed"]:
            decision["passed"] = False
            decision["reasons"].append("New improver's offspring did not improve the base executor")
    else:
        decision = {
            "passed": False,
            "reasons": [
                "Both improvers must produce valid, fully evaluated offspring under the same budget"
            ],
        }
    report = {
        "id": evaluation_id,
        "candidate": candidate_id,
        "target": "improver",
        "version_sha256": record["version_sha256"],
        "config_sha256": store.session["config_sha256"],
        "parent_executor": record["parent_executor"],
        "parent_improver": record["parent_improver"],
        "suites": suites,
        "gate": decision,
        "validation_only": False,
        "offspring": offspring,
        "fixed_evidence_ids": pool,
        "offspring_evaluations": {
            arm: read(store.root / "candidates" / offspring[arm] / "evaluation.json")
            for arm in reports
        },
        "budget": {
            "proposals_per_improver": 1,
            "same_base_executor": record["parent_executor"],
            "max_changed_files": store.config["max_changed_files"],
            "max_patch_bytes": store.config["max_patch_bytes"],
        },
        "note": "Promotes only the improver strategy, not either offspring executor.",
    }
    evaluation.save_evaluation(store, directory, output, report)
    return report


def cycle(
    store,
    *,
    target="executor",
    rounds=None,
    allow_model=False,
    allow_experiments=False,
    python=None,
):
    rounds = store.config["max_rounds"] if rounds is None else rounds
    if type(rounds) is not int or not 1 <= rounds <= store.config["max_rounds"]:
        raise ValueError("Requested rounds exceed the frozen cycle budget")
    results = []
    for _ in range(rounds):
        proposal = improver.propose(store, target, allow_model=allow_model)
        if proposal.get("status") == "awaiting_proposal":
            results.append(proposal)
            break
        cid = proposal["id"]
        checks = evaluation.check(store, cid, python=python)
        if not checks["passed"]:
            results.append({"candidate": cid, "status": "checks_failed"})
            continue
        if not allow_experiments:
            results.append({"candidate": cid, "status": "awaiting_evaluation"})
            break
        report = (
            evaluation.evaluate(store, cid, allow_experiments=True, python=python)
            if target == "executor"
            else evaluate_improver(
                store, cid, allow_model=allow_model, allow_experiments=True, python=python
            )
        )
        if report["gate"]["passed"]:
            evaluation.promote(store, cid)
        if target == "executor":
            # Development feedback can improve later rounds; validation trajectories never do.
            directory, _ = store.candidate(cid)
            evidence.collect(store, [directory / report["id"] / "candidate" / "development"])
        results.append(
            {
                "candidate": cid,
                "status": "promoted" if report["gate"]["passed"] else "not_promoted",
                "gate": report["gate"],
            }
        )
    store.event("cycle_finished", target=target, results=results)
    return results


def status(store):
    state = store.state
    versions = {kind: store.version(state[kind])[1] for kind in ("executor", "improver")}
    candidates = []
    for p in sorted((store.root / "candidates").glob("*/candidate.json")):
        record = read(p)
        checks = read(p.parent / "checks.json") if (p.parent / "checks.json").exists() else None
        reference = (
            read(p.parent / "evaluation.json") if (p.parent / "evaluation.json").exists() else None
        )
        candidates.append(
            {
                "id": record["id"],
                "target": record["target"],
                "checks_passed": checks["passed"] if checks else None,
                "evaluated": reference is not None,
            }
        )
    n = sum(len(suite_cases(store.config, k)) for k in suite_names(store.config))
    return {
        "workspace": str(store.root),
        "executor_method": store.config["executor"],
        "state": state,
        "benchmark": store.session["benchmark"],
        "suites": {name: store.config[name] for name in suite_names(store.config)},
        "active_hashes": {k: v["sha256"] for k, v in versions.items()},
        "development_evidence": len(list((store.root / "evidence").glob("*.json"))),
        "candidates": candidates,
        "planned_executor_comparison_episodes": 2 * n,
        "planned_improver_comparison_episodes": 4 * n,
        "default_behavior": "Prepare only; model requests and experiments require explicit flags.",
    }
