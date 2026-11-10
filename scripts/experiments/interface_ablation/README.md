# Interface ablation pilot

The policy receives permitted RGB, robot state, calibration, task goal and public
execution feedback. It does not inspect task source, hidden scene state or oracle
subgoals. The source audit and evaluator-only diagnosis are separate from policy
inference. All episodes use the **training** split.

Historical results in `docs/robomme_interface_review/` used the earlier combined
user prompt. New runs use `codex_base_instructions_v1`: the interface rules
replace the Codex base through `model_instructions_file`, with
`skills.include_instructions=false`. Resume requires matching protocol metadata;
an old interrupted run cannot continue under the new protocol. These pilot
results are not results for the current FDS candidate.

The runner extracts the fixed upstream Astra interface prompt from:

```
.cache/robomme_repo_review_20261110/embodied-astra-robomme/harness/astra_waypoint_eval_v3.py
```

That checkout is pinned at `79dcc7f73f536948cb039103d3fc5e722d689207` in
`jackbjed/embodied-astra-robomme`. The installed RoboMME-Plus checkout for this
experiment is `69a2fe25cb476069943aa7dd754d19f0932d9ec3`.

Run from the FDS repository root using the existing simulator environment:

```bash
.venv/bin/python scripts/experiments/interface_ablation/run_one.py \
  --task VideoPlaceOrder --episode 0 --variant full \
  --output runs/my_interface_pilot/full_VideoPlaceOrder_ep0

.venv/bin/python scripts/experiments/interface_ablation/run_panel.py \
  --out runs/my_interface_pilot --workers 2

.venv/bin/python scripts/experiments/interface_ablation/summarize.py \
  runs/my_interface_pilot --out runs/my_interface_pilot/summary

.venv/bin/python scripts/experiments/interface_ablation/verify_artifacts.py \
  runs/my_interface_pilot --out runs/my_interface_pilot/summary/verification.json
```

Codex CLI authentication must already be available. The CLI is an ephemeral,
tool-disabled JSON inference transport, not an autonomous agent in the workspace.
The panel uses this machine's existing proxy on `127.0.0.1:21084`; adjust that
setting for another machine. Single-run inference inherits proxy settings.
Requests have bounded timeouts and no transport retries. A panel stops admitting
new jobs on infrastructure errors; the other in-flight jobs finish.

To explicitly recover an interrupted inference, use a new directory:

```bash
.venv/bin/python scripts/experiments/interface_ablation/run_one.py \
  --task VideoPlaceOrder --episode 0 --variant full \
  --resume-from runs/my_interface_pilot/full_VideoPlaceOrder_ep0 \
  --output runs/my_interface_pilot/full_VideoPlaceOrder_ep0_resume1
```

Recovery replays completed actions without model calls, and requires the exact
same interrupted robot state and PNG hashes. It retains the original attempt.
It never retries semantic policy failures. `run_panel.py --resume-errors` allows
one such recovery per interrupted case. Do not launch two panels that own the
same unfinished output directories.

Artifacts include source/runner hashes, fixed seed and difficulty, all prompts,
ordered image paths and hashes, raw CLI events, usage, validated answers,
executed actions and environment outcomes. Image files stay in the run folder;
they are not committed. The first live PatternLock smoke is named
`pilot_full_PatternLock_ep0`; the panel recognizes that result instead of repeating
it. `summarize.py` selects the end of each recovery chain, never the best score.

`replay_actions.py` makes no model calls. With `--evaluator-state`, it records
hidden state only for post-hoc diagnosis. `retreat_probe.py` and
`swing_height_probe.py` are explicitly **post-hoc action interventions**, not new
policy evaluation episodes. The swing intervention was selected after viewing
the evaluator; it must not be included in non-privileged success rates.

This is not an exact official reproduction: visible history is explicitly
replayed through fresh CLI requests, hidden reasoning is not preserved, and the
pilot has a 40-call cap rather than 150. One seed per cell and stochastic model
outputs permit mechanism exploration, not population-level causal estimates.

The main four-condition panel is reported separately from the two exploratory
`verified_progress` runs. `summarize.py` retains both and sums actual inference
directories across interrupted and resumed runs without double-counting replayed
prefixes. `plot_probes.py` exports PNG/SVG figures from completed intervention
artifacts; it requires matplotlib, already available in this workspace's venv.

Review: `docs/robomme_interface_review/review.md`.
