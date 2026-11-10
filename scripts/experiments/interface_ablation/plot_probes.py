"""Plot the two post-hoc action interventions from recorded simulator outcomes."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    swing = args.runs / "full_SwingXtimes_ep0_resume1"
    place = args.runs / "full_VideoPlaceOrder_ep0_resume1"
    original = json.loads((swing / "diagnostic_state_replay/diagnostics.json").read_text())["steps"]
    lowered = json.loads((swing / "height_probe/result.json").read_text())["steps"]
    hold = json.loads((place / "retreat_probe/result.json").read_text())["steps"]
    baseline = [r["execution"] for r in map(json.loads, (place / "exchange.jsonl").read_text().splitlines()) if "execution" in r]
    evidence = {"swing_original": original, "swing_lowered": lowered,
                "placement_original": baseline, "placement_hold_then_retreat": hold,
                "note": "Post-hoc diagnostics; evaluator state never supplied to policy. Height intervention selected after inspecting evaluator."}
    verified = args.runs / "verified_progress_SwingXtimes_ep0/diagnostic_state_replay/diagnostics.json"
    if verified.exists():
        evidence["swing_with_generic_verification_prompt"] = json.loads(verified.read_text())["steps"]
    (args.out / "probe_evidence.json").write_text(json.dumps(evidence, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    ax = axes[0]
    ax.step([r["turn"] + 1 for r in original], [r["evaluator_only_hidden_state"]["swing_count"] for r in original], where="post", color="#b34b49", label="Original saved actions")
    ax.step([r["turn"] + 1 for r in lowered], [r["evaluator_only_swing_count"] for r in lowered], where="post", color="#287b64", label="Carry waypoints 2 cm lower")
    ax.axhline(12, color="#777777", linestyle=":", linewidth=1)
    ax.set(title="SwingXtimes: physical events vs. issued moves", xlabel="Saved action number", ylabel="Evaluator-recorded target arrivals", ylim=(-.5, 13), yticks=[0, 3, 6, 9, 12])
    ax.legend(loc="upper left")
    ax = axes[1]
    # Last release is action 6 of this fixed successful source trace.
    origin = baseline[5]["execution_control_steps"]
    for rows, label, color in [(baseline[5:], "Retreat immediately", "#4268ad"), (hold[5:], "Hold 40 steps, then retreat", "#b87923")]:
        ax.step([r["execution_control_steps"] - origin for r in rows], [int(r["status"] == "success") for r in rows], where="post", label=label, color=color)
        end = rows[-1]
        ax.scatter([end["execution_control_steps"] - origin], [int(end["status"] == "success")], color=color)
    ax.set(title="VideoPlaceOrder: waiting does not replace retreat", xlabel="Control steps after release (20 Hz)", ylabel="Environment status", ylim=(-.1, 1.3), yticks=[0, 1], yticklabels=["ongoing", "success"])
    ax.legend(loc="upper left")
    fig.suptitle("Fixed-action diagnostic interventions | train episode 0 | no model calls", fontsize=12)
    fig.savefig(args.out / "mechanism_probes.png", dpi=180)
    fig.savefig(args.out / "mechanism_probes.svg")


if __name__ == "__main__":
    main()
