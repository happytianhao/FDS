"""Summarize logical episodes, retaining interrupted attempts separately."""
import argparse
import json
from pathlib import Path

TASKS = ("PatternLock", "VideoPlaceOrder", "SwingXtimes", "ButtonUnmask")
VARIANTS = ("full", "no_dynamics", "no_history", "sparse16")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(args.runs.glob("*/result.json")):
        result = json.loads(path.read_text())
        manifest = json.loads((path.parent / "manifest.json").read_text())
        records = [json.loads(x) for x in (path.parent / "exchange.jsonl").read_text().splitlines()]
        executions = [r["execution"] for r in records if "execution" in r]
        result.update(path=str(path.parent), resumed_from=manifest.get("resume_source"),
                      seed=manifest["seed"], difficulty=manifest["difficulty"],
                      execution_steps=executions[-1]["execution_control_steps"] if executions else 0,
                      failure_reason=executions[-1].get("evaluator_failure_reason") if executions else None)
        runs.append(result)
    selected = []
    for task in TASKS:
        for variant in VARIANTS:
            candidates = [r for r in runs if r["task"] == task and r["variant"] == variant]
            # Select the end of the recovery chain, never select according to success.
            continued = {r["resumed_from"] for r in candidates if r["resumed_from"]}
            endings = [r for r in candidates if str(Path(r["path"]).resolve()) not in continued]
            if len(endings) > 1:
                raise ValueError(f"Multiple independent runs for {task}/{variant}: manual review needed")
            if endings:
                selected.append(endings[0])
    document = {"design": "Four tasks x four interface conditions, train episode 0, paired seeds, one sample per cell",
                "planned": 16, "completed_usable": sum(r["success"] is not None for r in selected),
                "selected": selected, "exploratory": [r for r in runs if r["variant"] == "verified_progress"], "all_attempts": runs}
    usage = {}
    requests = 0
    for path in args.runs.glob("*/call_*/stats.json"):
        stats = json.loads(path.read_text())
        requests += 1
        for key, value in (stats.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    document["observed_inference_usage"] = {"attempts_with_stats": requests, **usage,
        "note": "Actual call directories only; resumed prefixes are not counted twice. Failed requests without usage have unknown token cost."}
    (args.out / "results.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    table = ["| Task | full | no_dynamics | no_history | sparse16 |", "|---|---|---|---|---|"]
    for task in TASKS:
        cells = []
        for variant in VARIANTS:
            matches = [r for r in selected if r["task"] == task and r["variant"] == variant]
            if not matches:
                cells.append("待完成")
            else:
                r = matches[0]
                label = "成功" if r["success"] is True else "失败" if r["success"] is False else "中断，未计分"
                cells.append(f"{label}；{r['calls']} 次 / {r['execution_steps']} 步")
        table.append("| " + task + " | " + " | ".join(cells) + " |")
    table.extend(["", "次数为总有效模型回答数（恢复前后合计），步数为执行控制步数；模型自行结束不视为环境成功。具体结局与失败原因见 results.json。", ""])
    (args.out / "results_table.md").write_text("\n".join(table))
    print(json.dumps({k: v for k, v in document.items() if k not in ("selected", "all_attempts", "exploratory")}))
    print("\n".join(table))


if __name__ == "__main__":
    main()
