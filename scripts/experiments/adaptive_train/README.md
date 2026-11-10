# Adaptive train campaign

Protocol: [adaptive_train_protocol.md](../../../docs/adaptive_train_protocol.md).
These scripts run a finite, review-driven training queue, with one explicitly
authorized Codex code reviewer triggered by every failure, plus 30-minute
health checkpoints. They do not invoke the
project's Meta Harness and never run test episodes.

The active deployment is recorded under
`runs/robomme_plus/task_harness/e4_astra_medium_adaptive_train/`. Do not re-run
`init` or create a second campaign to reset attempt counters. `campaign.py run`
and `optimizer.py` take exclusive locks; the coordinator can adopt surviving
workers by PID, creation time and exact command. A dead coordinator is not proof
that its scenes stopped.

One-time setup, from the repository root:

```bash
.venv/bin/python scripts/experiments/adaptive_train/campaign.py init --out /absolute/new/output
```

Run `campaign.py run --out ...`, `optimizer.py --out ...`, and optionally
`widget_publisher.py --out ...` as detached processes with separate logs. The
optimizer immediately handles pending failures on startup and new failures
through file events; its health-check timer is 30 minutes. `--once` is a manual
AI checkpoint and can consume tokens; it is not a read-only status command.

Single read-only progress/collection commands:

```bash
.venv/bin/python scripts/experiments/adaptive_train/campaign.py status --out /absolute/output
.venv/bin/python scripts/experiments/adaptive_train/reporting.py --out /absolute/output
```

`campaign.json` contains 50 possible cases: 15 × 3 difficulties and five independent
MoveCube variants (episodes 0–4). Higher difficulty tiers stay locked until the prior tier officially
succeeds. `attempts/<case>/attempt_NN` owns its request, process identities, logs
and immutable output. Failed cases wait for diagnosis and a validated repair before a priority retry;
three total runs per case includes infrastructure failures and regression checks.

`versions/vNNN/src` is a complete source/resource snapshot; every run imports it
through `PYTHONPATH` and uses its frozen config. The project `.venv` and installed
benchmark assets are shared through `FDS_WORKSPACE`. Source changes in a running
version are forbidden. Optimizer workspaces are separate from the shared source
tree. Ordinary candidate changes require source-scope checks, unchanged
configuration/scripts/dependencies, the trusted original offline suite and the
candidate's own tests. No benchmark oracle, hidden state, task answers or skills
may be introduced. Model/effort/sandbox/approval/budgets stay fixed.

The reviewer writes a structured proposal; the host applies its bounded queue
decisions. Special proposals remain in `NEEDS_USER_DECISION.json`. They are not
automatically approved. `optimizer/*/usage.json` separates optimization tokens
from scene inference tokens; missing usage is explicitly unknown.

Request resilience uses three immediate retries, one 60-second cooldown while
holding the live simulator, and at most one more three-retry window. With the
initial request in each window, this is at most eight attempts per logical call.
Permanent auth/quota failures do not retry. Scene outcomes are success/failed;
failure categories and official unknown status preserve the difference between
task failure and infrastructure failure. A circuit breaker pauses new scenes
for shared account/network/dependency faults. Per-scene initialization errors
hold only that case, while independent cases continue.

All external changes to `control.json` must hold `state.lock`. `halt=true` stops
new dispatch and stops the optimizer at its next checkpoint, while running
scenes finish normally. It is not an immediate process kill. Infrastructure
clearance identifies exact diagnosed attempt IDs in
`cleared_infrastructure_attempts`; never clear a newly arrived or undiagnosed
failure. The reviewer cannot self-clear auth/quota without an external recovery
decision.

`REPORT.md`, `results.json` and `results.csv` preserve version-specific results.
`FINAL.json`/`FINAL.md` identify the selected frozen source and its verified,
failed and untested cases. A cross-version success union is not a success rate
for that source. Attempt exhaustion without comparable evaluation is reported
as budget-limited, not as proven convergence.

Widget credentials remain exclusively in
`~/.config/fds-monitor/widget-credentials.json` (0600). Only deployment IDs go in
`widget-target.json`. `--write-source` is needed only for a UI change; normal
publishing uses the data endpoint. The publisher stops after `FINAL.json`.

Concurrency follows cgroup CPU/memory limits and visible GPU pressure; the old
three-scene cap is removed. The deployed policy begins at six and ramps by two
up to ten, with resource telemetry every 30 seconds in the background. Scene
inference remains remote. This timer performs no AI calls and sends no chat
updates. Existing scenes retain their source/config; changes affect new
admissions only. Resource policy and thread caps are saved in each new request.

Each failed attempt is a separate optimizer focus. `optimizer/failure-reviews.json`
deduplicates notifications, records diagnosis outcomes and bounds failed review
recovery. Ordinary task/program failures cannot retry on unchanged source or
stop early with remaining attempts. A third failure is still diagnosed, but
never rerun. Special changes await the user; other scenes continue. The optimizer
no longer pauses the entire dispatch queue during a candidate review.
