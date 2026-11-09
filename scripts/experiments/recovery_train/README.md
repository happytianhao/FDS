# Train recovery campaign templates

These are the versioned copies of the helpers used for the F1 train/easy/episode-0 recovery campaign. They are not part of the `fds` public CLI and are not imported by a running experiment. Keep generated results, credentials and deployment configuration under ignored or private locations.

The helpers expect to be copied into **`runs/<benchmark>/<harness>/<campaign>/`** inside this repository. Their repository-root calculation depends on this layout. Use a new directory for a new experiment; never overwrite an existing result directory. `run_one.py` intentionally accepts only F1, tasks in the manifest and episode 0. It does not expand into test or medium/hard.

The runtime requires Linux, the project `.venv`, and `psutil` (already present with the simulation setup). If needed, install only this optional monitoring dependency with the project's uv/mirror workflow:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'psutil>=5.9'
```

Copy the helper files into the new output directory. Prepare a `campaign.json` there containing:

- `tasks`: an explicit ordered allowlist of train episode-0 tasks.
- `configs.f1`: the absolute path of an external temporary TOML config; `config_sha256` and `config-snapshot.toml` must match it.
- `fds_source_sha256`: relative paths under `src/fds` mapped to their SHA-256 digests. Do not edit sources during the campaign.
- `train_episode_metadata`: a task-to-list mapping with official train `episode`, `difficulty` and `seed`; use the installed benchmark metadata, not invented seeds.
- `execution_policy.max_concurrency`: integer 1–3. Three is the current upper limit.
- `old_results` and `previous_campaign`: optional historical references for reporting. The widget's prior-round comparison expects `previous_campaign` when enabled.

Keep a `source.patch` and record the source commit. The experiment configuration used here keeps Astra / medium, memory, empty skills, seed 7, 100 logical decision calls and 400 tool steps per batch. Explicitly set `[recovery] transport_retries = 3` and `pause_seconds = 0` for bounded automatic transport recovery without a manual recovery window. The runner retains the local proxy `http://127.0.0.1:21084` and Hugging Face mirror setting used by this environment.

Create `logs/`, then start `run_campaign.py` using the project's `.venv/bin/python` in a detached session. The coordinator adopts verified surviving runners and accounts for them in the cap; it never restarts completed scenes. `campaign.lock` prevents duplicate coordinators. Do not bypass a `needs_diagnosis` state: inspect runner PID/create_time, child worker identities, output and logs before resuming.

Run `monitor_once.py` when a check is needed. `collect_results.py` and `render_report.py` run after queue changes and at completion. The report text is specific to this recovery experiment; review scope labels if adapting it to another study. The initial frame/seed checks require the historical artifacts; absent references are not passing checks. Token totals exclude requests without usage and are lower bounds in that case.

For the optional widget, also supply an ignored `widget-target.json` with the deployment's `composer_id` and `widget_id`. Put its bearer token in `~/.config/fds-monitor/widget-credentials.json` as a `token` field, with directory permission 700 and file permission 600. Do not put a token into Vue, a manifest, shell history or Git. `FDS_CODEX_BIN` can select the already-authenticated Codex executable for quota reads; the default is `codex` on PATH. No credentials are copied by the quota reader.

`widget_publisher.py --write-source` publishes the Vue source once. The default invocation publishes changing data on file events and exits after the campaign completes. It must have only one active deployment publisher for the widget; a per-directory lock protects a single campaign directory. `--once` publishes a single snapshot. The publisher's `schedule-arrangement.json` is display-only scheduling state; it does not authorize or arrange a wake-up. Platform task scheduling is managed separately.

The process tests in `tests/test_campaign_dispatch.py` run short synthetic child processes; they do not invoke FDS, a simulator or a model. They are skipped if the optional `psutil` dependency is unavailable. See [the detailed change notes](../../../docs/recovery_and_experiment_changes.md) for behavior and limitations.
