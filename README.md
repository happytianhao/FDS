# FDS: Falsifiable Decision State for RoboMME

FDS is an open-source, reproducible task harness for long-horizon RGB robotic tasks. It keeps one actor, the **task harness**, and one training-only improvement loop, the **meta harness**. The task harness calls `gpt-6-astra` (the same Astra model used by the original Harness project), emits ordered environment or auxiliary tool calls, and attaches a falsifiable decision contract to every decision. The meta harness can later propose small, evidence-backed task-harness changes between training episodes.

The initial release supports only:

- **RoboMME-Plus** (`robomme_plus`), the primary benchmark;
- **RoboMME** (`robomme`).

There are no copied historical result directories, credentials, or dependency checkouts in this repository. A fresh clone can install FDS and fetch the pinned benchmark sources independently of the original Harness checkout.

## FDS protocol

Every Sys2 response contains `tool_batch`, `fds_contract`, `memory_changes`, and `keyframe_changes`. `fds_contract` records:

- the selected candidate action;
- assumptions and their evidence references;
- the expected outcome;
- evidence that would falsify the decision;
- required evidence still missing;
- risk and commitment state.

The host validates the contract before execution and writes it to `audit/call_XXXX/fds_contract.json`. After the batch, FDS records a separate verdict. A successful command receipt is never treated as proof of semantic task success: the verdict is `contradicted` for explicit public execution errors and `uncertain` when the command ran but the task outcome is not publicly verified. This makes mistakes traceable for later meta-harness diagnosis.

The first public configuration intentionally has `skills = []`. The interface is already present so skills can be added later from training evidence without changing the benchmark, model, or evaluator.

## Installation

FDS requires Python 3.11 or 3.12 and uses `uv` for a locked environment.

```bash
git clone <your-fds-repository-url>
cd FDS
uv sync --locked --extra dev --extra api
```

The project never stores API keys. Configure the same Astra access that your Codex installation already uses, or export the key expected by your local planner setup. Check the local environment without making a model call:

```bash
.venv/bin/fds doctor
```

Fetch the exact benchmark revisions recorded in `configs/dependencies.toml`:

```bash
.venv/bin/python scripts/setup_dependencies.py robomme_plus
# and, if you also need the original benchmark:
.venv/bin/python scripts/setup_dependencies.py robomme
```

Build the simulator environments inside FDS. The Plus environment is `.venv`; the original RoboMME environment is `.venv-robomme`. They are separate because both releases expose a package named `robomme` but must use different pinned source revisions.

```bash
.venv/bin/python scripts/setup_environment.py robomme_plus
.venv/bin/python scripts/setup_environment.py robomme
```

The setup script installs FDS and the selected local checkout into that environment, including the pinned ManiSkill, SAPIEN, PyTorch and computer-vision dependencies. It never reads or modifies Harness environments. The corresponding frozen dependency snapshots are under `requirements/`.

Verify the simulator without calling Astra:

```bash
.venv/bin/python scripts/verify_environment.py robomme_plus
.venv-robomme/bin/python scripts/verify_environment.py robomme
```

Each command resets `BinFill` episode 0 on the training split, checks RGB/state/action dimensions, and records the exact local benchmark commit. If a package is accidentally imported from another checkout, the adapter fails fast.

## Run a reproducible training episode

RoboMME-Plus is the default and should be used first. The checked-in configs use the training split, episode 0, process isolation, and no skills. Run Plus with `.venv`; run original RoboMME with `.venv-robomme`.

```bash
.venv/bin/fds run \
  --config configs/robomme_plus/task_harness.toml \
  --task BinFill \
  --episodes 0 \
  --output runs/robomme_plus/task_harness/binfill_ep0
```

Use the original benchmark after the Plus smoke run:

```bash
.venv-robomme/bin/fds run \
  --config configs/robomme/task_harness.toml \
  --task BinFill \
  --episodes 0 \
  --output runs/robomme/task_harness/binfill_ep0
```

For import, schema, and contract checks that do not require a simulator or model:

```bash
.venv/bin/python scripts/smoke_fds.py
.venv/bin/pytest
```

The `*_debug.toml` files disable episode process isolation for local debugging. They do not change the task-harness contract.

## Meta harness (training-only RSI)

The meta harness is deliberately separate from task execution. It reads development traces, prepares an evidence-bounded proposal, checks a candidate in an isolated snapshot, and can compare parent and candidate on the same training cases. It may edit `src/fds/task_harness/` and, later, `task_harness.skills`; it cannot edit the benchmark, Astra model, evaluator, budgets, or held-out split. Validation and test traces are rejected as proposal evidence.

Initialize a Plus training workspace:

```bash
.venv/bin/fds-meta init \
  --config configs/meta_harness/robomme_plus.toml \
  --workspace runs/meta_harness/robomme_plus
```

Preparation is offline by default. Explicit flags are required before model calls or scored experiments:

```bash
.venv/bin/fds-meta collect runs/robomme_plus/task_harness/binfill_ep0
.venv/bin/fds-meta propose --workspace runs/meta_harness/robomme_plus
.venv/bin/fds-meta cycle --workspace runs/meta_harness/robomme_plus \
  --allow-model --allow-experiments --rounds 1
```

The `robomme.toml` meta configuration has the same training-only policy for RoboMME. A promoted task-harness version should be evaluated on a fresh held-out run after the meta workspace is frozen.

## Visualization

FDS reuses the audit and visualization format from the migrated Harness implementation:

```bash
.venv/bin/fds-visualize runs/robomme_plus/task_harness/binfill_ep0
.venv/bin/python -m fds.serve --port 5011 --directory .
```

Open the generated `visualization/index.html` through the local server. The audit contains prompts, tool receipts, selected frames, contract JSON, verdicts, and benchmark feedback; images are never silently relabeled as fresh observations.

## Repository layout

```text
src/fds/task_harness/   single Sys2 actor, tools, memory, and FDS contracts
src/fds/meta_harness/   training-only evidence, proposal, check, and promotion loop
src/fds/benchmarks/     RoboMME and RoboMME-Plus adapters only
src/fds/common/         transport, recovery, geometry, and audit utilities
src/fds/support/        internal mechanical support; not another public executor
configs/                benchmark, debug, and meta-harness configurations
requirements/            pinned environment specifications
scripts/                dependency setup and offline smoke checks
```

Runs, third-party checkouts, local configuration overrides, and credentials are ignored by Git. Historical results should be published separately with their manifest, source hashes, benchmark revision, and exact configuration.

## Reproducibility notes

- The planner model is fixed in the public configs as `gpt-6-astra` with medium effort.
- Random seeds, package versions, source hashes, benchmark revision, and effective config are recorded in each run manifest.
- RoboMME-Plus is run on `train` for development; `test` is reserved for the frozen final harness.
- RoboMME supports `train`, `val`, and `test`; meta-harness proposals may use only `train`.
- An executed command is not an official success claim. Official benchmark feedback remains separate from the FDS decision verdict.

## License

FDS is released under the MIT License. The benchmark repositories and simulator packages retain their own licenses; review those licenses before redistribution.
