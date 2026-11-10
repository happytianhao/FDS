# Training-only harness/FDS improvement campaign

This campaign seeks one reproducible Task Harness + episodic memory + Full FDS
version, with `skills=[]`. It is engineering optimization on the training set,
not a held-out estimate or a paired F0/F1 ablation.

## Authorized scope and limits

- RoboMME-Plus only. Fifteen difficulty-labelled tasks start at train episode 0
  (easy), advance to episode 1 (medium) only after official easy success, then
  episode 2 (hard) only after official medium success.
- MoveCube train episodes 0–4 are five independent variants: hook, poke, scoop,
  ball and cue, with difficulty `null`. User clarified the full scope is 50
  cases (15 × 3 + 5). They are not relabelled easy/medium/hard, and one variant's
  failure does not block another. Existing attempt counters remain unchanged.
- At most **three total physical runs per task/episode in this new campaign**,
  including failures, infrastructure failures, and regression verification.
  Historical E1/E2/E3 runs remain separate. Request/decision retries inside a
  run retain their existing bounded limits and consume their existing budgets.
- Explicit user exception recorded on 2026-11-09 after the account was topped up:
  `PatternLock_ep1`, `PickXtimes_ep1`, and `VideoRepick_ep0` may each receive
  **one fourth physical attempt**, solely to replace their quota-interrupted
  third attempts. Keep all original attempts and usage; no fifth attempt and no
  expansion to other cases. The campaign manifest must link the exact per-case
  authorization in `user-decisions/20261109_quota_topup_three_makeup_runs.json`.
- No test split, original RoboMME, skill modification, or formal Meta Harness
  execution. No commit or push is part of this campaign.
- Fixed planner: gpt-6-astra, medium effort, workspace-write, approval never;
  project `.venv`, proxy `http://127.0.0.1:21084`, seed 7, 100 logical calls,
  400 tool steps per batch, existing official episode step limits.
- User subsequently authorized stronger API/Codex resilience: after three
  immediate transport retries, retain the live environment through one
  60-second cooldown and one additional bounded retry window. At most eight
  request attempts per logical decision; no replay after tools or completion.
  Shared auth/quota/network failures pause new dispatch; local initialization
  failures hold only their own case for diagnosis. Final success/failed outcomes retain
  infrastructure causes and never fabricate official task status.
- Scene concurrency adapts to the actual container CPU quota, memory and GPU
  headroom. The previous three-scene cap has been lifted by the user. Initial
  admission is six, increasing by two up to the resource ceiling (currently ten
  with twelve CPU cores and one visible A100). Active scenes are never killed
  when capacity drops. A run owns an immutable source snapshot and config;
  editing the working tree cannot change a running scene.
- All sixteen tasks' episode-0 entries are offered before higher difficulties;
  MoveCube's other four independent variants share the entry priority. Each
  failure immediately enters a single-writer diagnosis queue. A failed scene
  waits for an evidence-based generic harness/FDS improvement and offline
  validation before its next attempt; improved retries take dispatch priority. Exhausted cases block only
  their task's higher difficulties; other tasks continue.

## Review and changes

The user explicitly rejected exposing official completion status to the planner
on 2026-11-09. The model must judge completion from the already authorized
observations and may choose `finish`. Keep official status for external scoring
and experiment reports; do not inject it into planner input, gate a valid finish
on the official evaluator, or force execution until official success. A
`planner_finish` with unsuccessful external scoring is an outcome to diagnose,
not by itself proof of a harness bug or premature termination. Distinguish a
mistaken completion judgment from an explicit decision to stop without success
by reading the model's stated reason and public evidence. Improve evidence
presentation, memory and self-evaluation within the existing observation scope.
Do not propose the rejected official-status input change again without a new
user request.

After a run, record official status, public failure evidence, process failures,
calls, attempts, known input/output tokens, missing usage, environment steps,
contract/rejection/verdict/recovery metrics, source/config hashes and paths.
Each newly completed failure triggers its own review through file events,
including a third failure (diagnosis only; never a fourth run). Do not wait for
the health-check timer or spend a task-failure retry on unchanged source. A
previously validated improvement that the failed case has not tried may be used.
Only independently recovered transient transport faults may replay unchanged
source. Do not stop a task failure early while attempts remain. If no routine
repair is feasible, preserve the remaining attempts and request a concrete
user decision. Independent scenes continue during analysis; repairs publish
atomically and cannot affect already running scenes.

An agent review may propose at most one coherent generic harness/FDS change.
It must cite training traces, distinguish hypotheses from verified causes, and
run relevant offline tests before registering a candidate snapshot.

The 2026-11-09 implementation audit found that 19 of v001-v021 changed mainly
shared execution guidance, while execution_verdict did not compare the contract's
predictions/falsifiers with new observations. Avoid further repetitive guidance
unless a specific missing instruction is demonstrated. Prefer preserving and
testing concrete citation, evidence-presentation, memory-dependency or feedback
repairs. Do not claim mechanical preflight rejection is visual/semantic
falsification. Keep successful-case regression checks focused on a coherent
candidate; they spend the same three-run allowance and cannot be replenished.

Routine changes are allowed in generic harness/FDS feedback, validation,
bounded recovery and public-observation reasoning. Never use hidden object
states, oracle actions, benchmark scoring internals, episode answers or
task-name/seed-specific policies. Do not weaken success criteria or fabricate
semantic verification. Such changes do not meet this experiment's scope.

Ask the user before changing the model, effort, budgets, tools/capabilities,
observation privileges, benchmark rules, algorithm architecture, external
services, or introducing task-specific strategies. Preserve an explicit
`needs_user_decision` proposal with the concrete patch and impact. Unrelated
approved work may proceed while that proposal waits.

## Selection and stopping

Prioritize official successes on comparable cases, then fewer process errors,
then lower known token cost. Keep version-specific results separate from
cross-version 'ever succeeded' coverage. Never claim that a mixed-version
coverage count measures one version's performance.

Successful cases need not be repeated without a code change. Use remaining
attempts to check regressions and validate the final common candidate. These
checks also count toward the three-run cap. Do not reset counters for a new
version or create another campaign to bypass the cap.

Treat two comparable improvement rounds with no additional official successes
and no clear process improvement as a practical plateau. If attempt limits or
infrastructure prevent such comparison, report **budget-limited/inconclusive**,
not proven convergence. A final candidate with unverified cases must list them
and must not be described as fully validated.

## Long-running operation

The scene queue waits on process/file events, not conversation polling. The
widget uses its existing data endpoint and private credential file; no secrets
enter source, experiment prompts or logs. Account quota readings carry their
sampling time and are distinct from experiment token usage.

The user explicitly authorized a background Codex reviewer, then clarified
that every failure must trigger immediate analysis/improvement/retry. The
30-minute timer remains a health-check fallback; it does not delay failures.
Parallel failures are diagnosed successively by one writer to prevent
conflicting patches, while independent scenes keep running. It uses Astra medium, workspace-write and approval never in an
isolated candidate workspace, never starts a second queue, and shares the
immutable snapshot/attempt limits. It cannot grant itself approval for special
changes. The old platform Arrange token is not used; this is a local background
process, not a change to the platform's permanent schedule.
