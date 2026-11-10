"""Bounded, review-driven TRAIN queue. No optimizer or model runs during init."""
import argparse
import difflib
import json
import os
import selectors
import shutil
import subprocess
import tomllib
from pathlib import Path

import psutil
from common import FileEvents, atomic, hashes, identity_alive, lock, now, process_exit_fd, read, sha
from resources import DEFAULT_POLICY, ResourceController, ResourceTimer

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name('run_one.py').resolve()
AUTO_HOLD_PREFIX = 'Infrastructure diagnosis required: '
QUOTA_EXTENSION_CASES = frozenset({'PatternLock_ep1', 'PickXtimes_ep1', 'VideoRepick_ep0'})


def attempt_limit(manifest, case_id, past_attempts=None):
    """One additional run only for the three explicitly authorized quota cases."""
    override = manifest.get('attempt_limit_overrides', {}).get(case_id, {})
    auth = override.get('authorization', {})
    evidence = auth.get('evidence')
    permitted = (case_id in QUOTA_EXTENSION_CASES and override.get('attempt_limit') == 4
                 and override.get('interrupted_attempt') == f'{case_id}_attempt_03'
                 and override.get('interruption_category') == 'quota'
                 and auth.get('source') == 'user' and bool(str(auth.get('reason', '')).strip())
                 and isinstance(evidence, list) and bool(evidence)
                 and all(isinstance(p, str) and Path(p).is_absolute() and Path(p).is_file() for p in evidence))
    if permitted and past_attempts is not None and len(past_attempts) >= 3:
        third = past_attempts[2]
        permitted = (third.get('id') == f'{case_id}_attempt_03' and third.get('failure_category') == 'quota'
                     and third.get('state') == 'finished' and not third.get('success'))
    return 4 if permitted else 3


def requires_global_hold(row):
    """Shared service outages stop admissions; local scene failures await review."""
    if row.get('success') or row.get('state') in ('running', 'needs_diagnosis'):
        return False
    if row.get('failure_category') in ('auth', 'authentication', 'authorization', 'quota', 'network', 'transport'):
        return True
    # Dependency/import/config failures must be explicitly diagnosed as shared;
    # a generic RuntimeError or missing scene artifact alone proves no outage.
    return row.get('failure_scope') == 'global' and bool(row.get('shared_failure_evidence'))


def reconcile_global_hold(control, completed):
    """Refresh only our derived hold, preserving independent operator controls."""
    updated = dict(control)
    cleared = set(control.get('cleared_infrastructure_attempts', []))
    ids = sorted(row['id'] for row in completed if requires_global_hold(row) and row['id'] not in cleared)
    updated['global_failure_attempts'] = ids
    reason = control.get('hold_reason')
    if not reason or (isinstance(reason, str) and reason.startswith(AUTO_HOLD_PREFIX)):
        updated['hold_reason'] = AUTO_HOLD_PREFIX + ', '.join(ids) if ids else None
    return updated


def fixed_config(path):
    text = path.read_text()
    config = tomllib.loads(text)
    required = {'method': 'task_harness'}
    assert all(config[k] == v for k, v in required.items())
    for section, values in {
        'benchmark': {'name': 'robomme_plus', 'split': 'train'},
        'planner': {'backend': 'codex', 'model': 'gpt-6-astra', 'effort': 'medium'},
        'fds': {'enabled': True, 'contract_mode': 'full'},
        'task_harness': {'skills': []},
        'run': {'seed': 7, 'max_calls': 100, 'max_tool_steps': 400},
    }.items():
        if any(config[section].get(k) != v for k, v in values.items()):
            raise ValueError('Fixed protocol mismatch: ' + section)
    if 'recovery' in config:
        if config['recovery'].get('transport_retries') != 3 or config['recovery'].get('pause_seconds') != 0:
            raise ValueError('Recovery must specify retries=3, pause=0')
        if 'auto_resume_after_seconds' not in config['recovery']:
            text = text.replace('[recovery]', '[recovery]\nauto_resume_after_seconds = 60', 1)
        elif config['recovery']['auto_resume_after_seconds'] != 60:
            raise ValueError('Recovery auto-resume interval must be 60 seconds')
    else:
        text += '\n[recovery]\ntransport_retries = 3\npause_seconds = 0\nauto_resume_after_seconds = 60\n'
    return text


def allowed_change(name):
    return name in {'fds/decision.py', 'fds/common/retry.py', 'fds/common/recovery.py', 'fds/planners.py', 'fds/cli.py'} or (name.startswith('fds/task_harness/') and name != 'fds/task_harness/skills.py')


def verify_version(out, version, validation=True):
    if not version.startswith('v') or not version[1:].isdigit():
        raise ValueError('Invalid version')
    folder = out / 'versions' / version
    meta = read(folder / 'version.json')
    if not meta or hashes(folder / 'src') != meta['source_sha256']:
        raise ValueError('Snapshot source integrity mismatch')
    if sha(folder / 'config.toml') != meta['config_sha256'] or (folder / 'config.toml').read_bytes() != (out / 'versions/v000/config.toml').read_bytes():
        raise ValueError('Frozen configuration mismatch')
    baseline = read(out / 'versions/v000/version.json')['source_sha256']
    changed = [k for k in baseline.keys() | meta['source_sha256'].keys() if baseline.get(k) != meta['source_sha256'].get(k)]
    if any(not allowed_change(k) for k in changed):
        raise ValueError('Candidate modifies files outside harness/FDS scope')
    if validation and version != 'v000':
        result = read(folder / 'validation.json')
        if result.get('passed') is not True:
            raise ValueError('Candidate needs passing offline validation')
    return meta


def snapshot(out, reason, initial=False, source=None):
    campaign = read(out / 'campaign.json')
    root = Path(source or campaign['root']).resolve()
    versions = out / 'versions'
    versions.mkdir(exist_ok=True)
    index = max([int(p.name[1:]) for p in versions.glob('v[0-9]*')] or [-1]) + 1
    version = f'v{index:03d}'
    folder = versions / version
    current = hashes(root / 'src')
    if not initial:
        base = read(versions / 'v000/version.json')['source_sha256']
        if any(not allowed_change(k) for k in base.keys() | current.keys() if base.get(k) != current.get(k)):
            raise ValueError('Working tree changes outside harness/FDS scope')
    folder.mkdir()
    shutil.copytree(root / 'src', folder / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Entire resource tree is preserved; symlinks are materialized into this snapshot.
    config = fixed_config(Path(campaign['config_path'])) if initial else (versions / 'v000/config.toml').read_text()
    (folder / 'config.toml').write_text(config)
    repository = Path(campaign['root'])
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository, capture_output=True, text=True, check=True).stdout.strip()
    if initial:
        patch = subprocess.run(['git', 'diff', 'HEAD', '--', 'src'], cwd=repository, capture_output=True, check=True).stdout
        (folder / 'source.patch').write_bytes(patch)
    else:
        baseline = versions / 'v000/src'
        lines = []
        for name in sorted(hashes(baseline).keys() | current.keys()):
            old, new = baseline / name, folder / 'src' / name
            before = old.read_text(errors='replace').splitlines(keepends=True) if old.is_file() else []
            after = new.read_text(errors='replace').splitlines(keepends=True) if new.is_file() else []
            lines.extend(difflib.unified_diff(before, after, fromfile='baseline/src/' + name, tofile=version + '/src/' + name))
        (folder / 'source.patch').write_text(''.join(lines))
    atomic(folder / 'version.json', {'id': version, 'source_sha256': hashes(folder / 'src'),
           'config_sha256': sha(folder / 'config.toml'), 'reason': reason, 'source_commit': commit, 'created_utc': now()})
    return version


def initialize(out, root=ROOT):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'campaign.json').exists() or (out / 'versions').exists():
        raise ValueError('Campaign already exists; refusing to reset counters')
    metadata = root / 'third_party/robomme_plus/src/robomme_plus/env_metadata/train'
    cases = []
    for path in sorted(metadata.glob('record_dataset_*_metadata.json')):
        data = read(path)
        task = data['env_id']
        for record in data['records']:
            episode = record['episode']
            if episode not in (range(5) if task == 'MoveCube' else [0, 1, 2]):
                continue
            difficulty = None if task == 'MoveCube' else ['easy', 'medium', 'hard'][episode]
            if record['difficulty'] != difficulty:
                raise ValueError('Metadata difficulty mismatch')
            # Deliberate public identity allowlist; never copy oracle fields.
            cases.append(dict(id=f'{task}_ep{episode}', task=task, episode=episode,
                              difficulty=difficulty, seed=record['seed'], variant=record.get('variant')))
    tasks = sorted({c['task'] for c in cases})
    if len(tasks) != 16 or len(cases) != 50:
        raise ValueError('Expected 15 labelled tasks and five independent MoveCube variants')
    atomic(out / 'campaign.json', dict(root=str(root), output=str(out), tasks=tasks, cases=cases,
           max_attempts=3, max_concurrency=DEFAULT_POLICY['max_concurrency'], resource_policy=DEFAULT_POLICY,
           config_path=str(root / 'configs/robomme_plus/task_harness.toml')))
    (out / 'attempts').mkdir()
    (out / 'reviews').mkdir()
    version = snapshot(out, 'Initial frozen baseline', initial=True)
    atomic(out / 'control.json', dict(active_version=version, halt=False))
    return version


def runner_command(request):
    return [str(Path(request['root']) / '.venv/bin/python'), str(RUNNER), '--request',
            str(Path(request['output']).parent / 'request.json')]


def inspect_attempt(folder):
    req, state = read(folder / 'request.json'), read(folder / 'status.json')
    if not req:
        return {'id': folder.name, 'state': 'needs_diagnosis', 'errors': ['Missing request; preserved reservation']}
    row = dict(req)
    for role in ('runner', 'worker'):
        pid, created = state.get(role + '_pid'), state.get(role + '_create_time')
        if identity_alive(pid, created):
            expected = runner_command(req) if role == 'runner' else state.get('command')
            if not expected or not identity_alive(pid, created, expected):
                return dict(row, state='needs_diagnosis', errors=['Live process command mismatch'])
            if role == 'runner':
                return dict(row, state='running', pid=pid, create_time=created)
            return dict(row, state='needs_diagnosis', pid=pid, create_time=created,
                        errors=['Orphan worker alive; case halted'])
    # Cover wrapper death between spawn and identity publication conservatively.
    for proc in psutil.process_iter(['pid', 'cmdline', 'create_time', 'status']):
        argv = proc.info['cmdline'] or []
        if proc.info['status'] != psutil.STATUS_ZOMBIE and (argv == runner_command(req) or
                (str(req['output']) in argv and 'fds.cli' in argv)):
            return dict(row, state='needs_diagnosis', pid=proc.pid, create_time=proc.info['create_time'],
                        errors=['Unverified live attempt process; case halted'])
    summary = read(Path(req['output']) / req['task'] / f"episode_{req['episode']:04d}" / 'summary.json')
    metrics = read(Path(req['output']) / 'metrics.json')
    errors = []
    if state.get('status') != 'completed' or not summary or metrics.get('status') != 'completed' or metrics.get('errors'):
        errors.append(state.get('infrastructure_error') or 'Incomplete or failed scene')
    success = bool(summary.get('success') is True and summary.get('official_status') == 'success' and not errors)
    category = summary.get('failure_category') or ('infrastructure' if errors else (None if success else 'task'))
    if errors:
        from fds.common.retry import classify_request_failure
        # Classify actual errors only. Digits in token totals or scene logs are
        # not HTTP status codes, and a task failure is not a network failure.
        details = [summary.get('error')]
        infrastructure = state.get('infrastructure_error') or {}
        if isinstance(infrastructure, dict):
            details.append(infrastructure.get('error'))
        for detail in details:
            if not detail:
                continue
            classified = classify_request_failure(str(detail))
            if classified['failure_category'] in ('authentication', 'authorization'):
                category = 'auth'
                break
            if classified['failure_category'] == 'quota':
                category = 'quota'
                break
            if classified['retryable']:
                category = 'network'
    return dict(row, state='finished', success=success, outcome='success' if success else 'failed', failure_category=category,
                failure_scope=state.get('failure_scope', summary.get('failure_scope')),
                shared_failure_evidence=state.get('shared_failure_evidence', summary.get('shared_failure_evidence')),
                official_status=summary.get('official_status', 'unknown'), stop_reason=summary.get('stop_reason', 'missing_result'), errors=errors)


def build_state(out):
    campaign = read(out / 'campaign.json')
    rows, active, completed = [], [], []
    for case in campaign['cases']:
        past = [inspect_attempt(p) for p in sorted((out / 'attempts' / case['id']).glob('attempt_*')) if p.is_dir()]
        limit = attempt_limit(campaign, case['id'], past)
        row = dict(case, past_attempts=past, state='pending', attempt_limit=limit)
        if past:
            last = past[-1]
            decision = read(out / 'reviews' / (last['id'] + '.json'))
            if last['state'] in ('running', 'needs_diagnosis'):
                row['state'] = last['state']
            elif decision.get('action') == 'retry' and len(past) < limit:
                row.update(state='retry', retry_version=decision.get('version'))
            elif last.get('success'):
                row['state'] = 'success'
            elif decision.get('action') == 'stop' or len(past) >= limit:
                row['state'] = 'stopped'
            else:
                row['state'] = 'waiting_review'
            active.extend(p for p in past if p['state'] == 'running')
            completed.extend(p for p in past if p['state'] == 'finished')
        rows.append(row)
    byid = {r['id']: r for r in rows}
    entries_started = all(r['past_attempts'] for r in rows if r['episode'] == 0)
    for row in rows:
        if row['state'] in ('pending', 'retry') and row['episode'] > 0 and row['difficulty'] is not None:
            parent = byid[f"{row['task']}_ep{row['episode'] - 1}"]
            if parent['state'] == 'stopped':
                row['state'] = 'blocked'
            elif not entries_started or parent['state'] != 'success':
                row['state'] = 'locked'
    # Propagate exhausted/stopped prerequisites through every later tier.
    for row in sorted(rows, key=lambda r: r['episode']):
        if row['difficulty'] is not None and row['episode'] and row['state'] == 'locked' and byid[f"{row['task']}_ep{row['episode'] - 1}"]['state'] in ('blocked', 'stopped'):
            row['state'] = 'blocked'
    status = 'running' if active or any(r['state'] in ('pending', 'retry') for r in rows) else 'waiting_review'
    if any(r['state'] == 'needs_diagnosis' for r in rows):
        status = 'needs_diagnosis'
    elif all(r['state'] in ('success', 'stopped', 'blocked') for r in rows):
        status = 'completed'
    previous = read(out / 'status.json')
    adaptive = campaign.get('resource_policy', {}).get('enabled', False)
    return dict(status=status, max_concurrency=previous.get('max_concurrency', campaign['max_concurrency']) if adaptive else campaign['max_concurrency'],
                concurrency_ceiling=campaign['max_concurrency'], resources=previous.get('resources'), active_runs=active,
                completed=completed, cases=rows, updated_utc=now())


def review(out, attempt, action, reason, version=None, evidence=()):
    if not reason.strip():
        raise ValueError('A diagnosis/reason is required')
    state = build_state(out)
    matches = [(c, a) for c in state['cases'] for a in c['past_attempts'] if a['id'] == attempt]
    if len(matches) != 1:
        raise ValueError('Unknown attempt')
    case, row = matches[0]
    if row != case['past_attempts'][-1] or row['state'] != 'finished':
        raise ValueError('Only the latest finished attempt can be reviewed')
    target = out / 'reviews' / (attempt + '.json')
    previous = read(target)
    extension = case['attempt_limit'] == 4 and len(case['past_attempts']) == 3 and row.get('failure_category') == 'quota'
    if previous and not ((previous['action'] == 'accept' or (previous['action'] == 'stop' and extension)) and action == 'retry'):
        raise ValueError('Attempt already reviewed')
    if action == 'retry':
        if len(case['past_attempts']) >= case['attempt_limit']:
            raise ValueError(f"Attempt cap reached ({case['attempt_limit']})")
        version = version or read(out / 'control.json')['active_version']
        verify_version(out, version)
        if row.get('success') and ('regression' not in reason.lower() or version == row['version']):
            raise ValueError('Success retry needs regression reason and a different validated version')
    if action == 'accept' and not row.get('success'):
        raise ValueError('Cannot accept a non-success as official success')
    if previous:
        suffix = '.accepted.json' if previous['action'] == 'accept' else '.stop_before_authorized_extension.json'
        atomic(out / 'reviews' / (attempt + suffix), previous)
    atomic(target, dict(attempt=attempt, action=action, reason=reason, version=version,
                        evidence=[str(Path(p).resolve()) for p in evidence], created_utc=now()))


def dispatch(out, case, version):
    campaign = read(out / 'campaign.json')
    verify_version(out, version)
    index = len(case['past_attempts']) + 1
    limit = attempt_limit(campaign, case['id'], case['past_attempts'])
    if index > limit:
        raise ValueError('Attempt cap reached')
    folder = out / 'attempts' / case['id'] / f'attempt_{index:02d}'
    folder.mkdir(parents=True, exist_ok=False)
    req = {k: case[k] for k in ('task', 'episode', 'difficulty', 'seed')}
    req.update(id=f"{case['id']}_attempt_{index:02d}", case_id=case['id'], attempt=index, version=version,
               root=campaign['root'], out=str(out), campaign=str(out),
               source_root=str(out / 'versions' / version / 'src'),
               config=str(out / 'versions' / version / 'config.toml'), output=str(folder / 'output'))
    req['resource_policy'] = campaign.get('resource_policy', {})
    req['attempt_limit'] = limit
    if index > 3:
        req['attempt_limit_authorization'] = campaign['attempt_limit_overrides'][case['id']]
    atomic(folder / 'request.json', req)
    try:
        with (folder / 'runner.log').open('xb') as stream:
            child = subprocess.Popen(runner_command(req), cwd=campaign['root'], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as error:
        atomic(folder / 'status.json', dict(status='failed', infrastructure_error={'error_type': type(error).__name__, 'error': str(error)}, finished_utc=now()))
        return None
    atomic(folder / 'status.json', dict(status='starting', runner_pid=child.pid,
           runner_create_time=psutil.Process(child.pid).create_time(), started_utc=now()))
    return child


def run(out):
    with lock(out / 'campaign.lock', nonblocking=True):
        events = FileEvents()
        resources = ResourceController()
        timer = ResourceTimer(int(read(out / 'campaign.json').get('resource_policy', {}).get('sample_seconds', 30)))
        children = []
        try:
            # Only external control/review and attempt status directories wake us;
            # status.json publication itself must not cause a busy event loop.
            events.watch(out / 'reviews')
            events.watch(out, names={'control.json', 'campaign.json'})
            while True:
                with lock(out / 'state.lock'):
                    state = build_state(out)
                    control = read(out / 'control.json')
                    occupied = len(state['active_runs']) + sum(c['state'] == 'needs_diagnosis' for c in state['cases'])
                    resource_state = resources.sample(read(out / 'campaign.json'), state['active_runs'], occupied)
                    limit = resource_state['target']
                    # Local initialization/runtime failures remain waiting_review
                    # on their own case. Only diagnosed shared failures block peers.
                    updated_control = reconcile_global_hold(control, state['completed'])
                    if updated_control != control:
                        control = updated_control
                        atomic(out / 'control.json', control)
                    if not control.get('halt') and not control.get('reviewer_hold') and not control.get('hold_reason') and not control.get('hold_until'):
                        candidates = sorted((c for c in state['cases'] if c['state'] in ('pending', 'retry')), key=lambda c: (c['state'] != 'retry', c['episode'] if c.get('difficulty') is not None else 0, c['task'], c['episode']))
                        for case in candidates[:max(0, limit - occupied)]:
                            child = dispatch(out, case, case.get('retry_version') or control['active_version'])
                            if child is None:
                                break  # diagnose launch failure before spending more attempts
                            children.append(child)
                    state = build_state(out)
                    state.update(pid=os.getpid(), create_time=psutil.Process().create_time(),
                                 max_concurrency=limit, resources=resource_state)
                    atomic(out / 'status.json', state)
                    for c in state['cases']:
                        for a in c['past_attempts']:
                            if a.get('output'):
                                events.watch(Path(a['output']).parent, names={'status.json'})
                    # Drain our own atomic write events before blocking, under the
                    # same lock external mutators use to prevent lost wakeups.
                    events.drain()
                for child in children:
                    child.poll()
                children = [p for p in children if p.returncode is None]
                if state['status'] == 'completed' or (control.get('halt') and not state['active_runs']):
                    return
                with selectors.DefaultSelector() as selector:
                    selector.register(events.fd, selectors.EVENT_READ)
                    selector.register(timer.fd, selectors.EVENT_READ)
                    descriptors = []
                    raced = False
                    try:
                        for row in state['active_runs']:
                            try:
                                fd = process_exit_fd(row['pid'])
                            except ProcessLookupError:
                                raced = True
                                break
                            descriptors.append(fd)
                            selector.register(fd, selectors.EVENT_READ)
                        if not raced:
                            while True:
                                ready = selector.select()
                                meaningful = False
                                for event, _ in ready:
                                    if event.fd == events.fd:
                                        meaningful |= events.drain()
                                    elif event.fd == timer.fd:
                                        timer.drain()
                                        meaningful = True
                                    else:
                                        meaningful = True
                                if meaningful:
                                    break
                    finally:
                        for fd in descriptors:
                            os.close(fd)
        except BaseException as error:
            state = read(out / 'status.json')
            state.update(status='needs_diagnosis', dispatch_error=str(error), updated_utc=now())
            atomic(out / 'status.json', state)
            raise
        finally:
            events.close()
            timer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'snapshot', 'activate', 'run', 'status', 'review'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reason', default='')
    parser.add_argument('--version')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--attempt')
    parser.add_argument('--action', choices=['retry', 'stop', 'accept'])
    parser.add_argument('--evidence', action='append', default=[])
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error('--out must be absolute')
    out = args.out.resolve()
    if args.command == 'init':
        out.mkdir(parents=True, exist_ok=True)
    if args.command == 'run':
        run(out)
        return
    with lock(out / 'state.lock'):
        if args.command == 'init':
            print(initialize(out))
        elif args.command == 'snapshot':
            if not args.reason.strip():
                parser.error('--reason is required')
            print(snapshot(out, args.reason, source=args.source))
        elif args.command == 'activate':
            verify_version(out, args.version)
            control = read(out / 'control.json')
            control['active_version'] = args.version
            atomic(out / 'control.json', control)
        elif args.command == 'review':
            if not args.attempt or not args.action:
                parser.error('--attempt and --action are required')
            review(out, args.attempt, args.action, args.reason, args.version, args.evidence)
        else:
            state = build_state(out)
            previous = read(out / 'status.json')
            for key in ('pid', 'create_time'):
                state[key] = previous.get(key)
            atomic(out / 'status.json', state)
            print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
