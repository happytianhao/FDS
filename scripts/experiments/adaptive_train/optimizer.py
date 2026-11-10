"""Failure-triggered Codex reviewer plus 30-minute health checks; one writer."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import campaign
import psutil
from common import FileEvents, atomic, hashes, identity_alive, lock, now, read
from reporting import safe_read, write_report

INTERVAL = 1800
MODEL = 'gpt-6-astra'
EFFORT = 'medium'
TERMINAL = {'completed', 'budget_limited', 'plateau'}


def response_schema():
    def obj(fields):
        return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
    text = {'type': 'string'}
    return obj({
        'summary': text, 'change_kind': {'type': 'string', 'enum': ['none', 'routine', 'needs_confirmation']},
        'rationale': text, 'evidence': {'type': 'array', 'items': text},
        'decisions': {'type': 'array', 'items': obj({
            'attempt_id': text, 'action': {'type': 'string', 'enum': ['retry', 'stop', 'accept']},
            'reason': text,
        })},
        'regression_cases': {'type': 'array', 'items': text},
        'clear_infrastructure_hold': {'type': 'boolean'},
        'recommended_version': text,
        'finish_campaign': {'type': 'boolean'},
        'stop_basis': {'type': 'string', 'enum': ['continue', 'plateau', 'budget_limited', 'all_success', 'needs_user']},
        'user_question': text,
    })


def changed_sources(base, work):
    a, b = hashes(base / 'src'), hashes(work / 'src')
    return [name for name in sorted(a.keys() | b.keys()) if a.get(name) != b.get(name)]


def validate_candidate_scope(base, work, tasks):
    changed = changed_sources(base, work)
    for name in changed:
        if not campaign.allowed_change(name):
            raise ValueError('Special change outside routine source scope: ' + name)
        path = work / 'src' / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Source deletion/symlink requires review: ' + name)
        old = (base / 'src' / name).read_text() if (base / 'src' / name).exists() else ''
        new = path.read_text()
        additions = '\n'.join(line[1:] for line in difflib.unified_diff(old.splitlines(), new.splitlines())
                              if line.startswith('+') and not line.startswith('+++'))
        if any(task in additions for task in tasks):
            raise ValueError('Task-name-specific source change requires confirmation: ' + name)
    return changed


def ancillary_hashes(work):
    """Only source and tests are mutable in a reviewer workspace."""
    result = {}
    for name in ('scripts', 'configs', 'docs'):
        result.update({name + '/' + k: v for k, v in hashes(work / name).items()})
    for name in ('pyproject.toml', 'uv.lock'):
        result[name] = hashlib.sha256((work / name).read_bytes()).hexdigest()
    return result


def version_evidence(result):
    """Select only evaluated versions; promotions require comparable coverage."""
    maps = {}
    for row in result['rows']:
        if row['terminal']:
            maps.setdefault(row['version'], {})[row['case_id']] = row
    best = 'v000'
    promotions = []
    for version in sorted(maps):
        if version == best:
            continue
        current, candidate = maps.get(best, {}), maps[version]
        shared = set(current) & set(candidate)
        old_successes = {c for c, r in current.items() if r['success']}
        regressed = [c for c in shared if current[c]['success'] and not candidate[c]['success']]
        gained = [c for c in shared if not current[c]['success'] and candidate[c]['success']]
        covered = old_successes <= set(candidate)
        if shared and covered and not regressed and gained:
            promotions.append({'from': best, 'to': version, 'gained': gained, 'regressions': regressed})
            best = version
    chosen = maps.get(best, {})
    ever_success = {r['case_id'] for r in result['rows'] if r['terminal'] and r['success']}
    unverified = sorted(ever_success - set(chosen))
    regressed = sorted(c for c in ever_success & set(chosen) if not chosen[c]['success'])
    compared = []
    for version in sorted(maps)[-3:]:
        comparable = maps[version]
        if comparable and all(not r.get('failure_category') or r['failure_category'] == 'task' for r in comparable.values()):
            compared.append(comparable)
    plateau = (len(compared) == 3 and set(compared[0]) == set(compared[1]) == set(compared[2])
               and sum(r['success'] for r in compared[0].values()) >= sum(r['success'] for r in compared[1].values())
               >= sum(r['success'] for r in compared[2].values()))
    # Success counts alone are insufficient: recovery/process improvements also prevent a plateau claim.
    if plateau:
        for before, after in zip(compared, compared[1:], strict=False):
            if sum(r.get('recoveries', 0) for r in after.values()) > sum(r.get('recoveries', 0) for r in before.values()):
                plateau = False
            if sum(r.get('decision_rejections', 0) for r in after.values()) < sum(r.get('decision_rejections', 0) for r in before.values()):
                plateau = False
            if sum(r.get('contradicted', 0) for r in after.values()) > sum(r.get('contradicted', 0) for r in before.values()):
                plateau = False
            old_tokens = sum(r.get('total_tokens', 0) for r in before.values())
            new_tokens = sum(r.get('total_tokens', 0) for r in after.values())
            if old_tokens and new_tokens < old_tokens * .95:
                plateau = False
    return {'best_version': best, 'promotions': promotions, 'unverified_success_cases': unverified,
            'known_failures_despite_other_version_success': regressed, 'comparable_plateau_evidence': plateau,
            'evaluated_cases': sorted(chosen), 'successful_cases': sorted(c for c, r in chosen.items() if r['success'])}


def reviewer_environment(root, work):
    env = os.environ.copy()
    for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL'):
        env.pop(key, None)
    env['PATH'] = str(Path.home() / '.local/bin') + os.pathsep + env.get('PATH', '')
    env['FDS_WORKSPACE'] = str(root)
    env['PYTHONPATH'] = str(work / 'src')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = 'http://127.0.0.1:21084'
    env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost,::1'
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    return env


def make_workspace(root, out, directory, version):
    work = directory / 'workspace'
    work.mkdir()
    shutil.copytree(out / 'versions' / version / 'src', work / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('tests', 'scripts', 'configs'):
        shutil.copytree(root / name, work / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copyfile(root / name, work / name)
    (work / '.venv').symlink_to(root / '.venv', target_is_directory=True)
    (work / 'docs').mkdir()
    shutil.copyfile(root / 'docs/adaptive_train_protocol.md', work / 'docs/adaptive_train_protocol.md')
    return work


def reviewer_prompt(root, out, work, result, version, focus_attempt=None):
    authorized_limits = {c['id']: c.get('attempt_limit', 3) for c in result.get('cases', []) if c.get('attempt_limit', 3) != 3}
    return f'''你是用户明确授权的 FDS 后台优化代理。本次只检查一次，分析完成后退出，不要轮询、sleep、再启动代理或实验。
工作目录 {work} 是独立候选源码副本。正式仓库 {root} 及所有运行中的源码快照不可写。
当前活动版本 {version}。先读 docs/adaptive_train_protocol.md；用户最新补充要求请求失败可恢复，最终成功/失败必须保留基础设施原因，不能伪造官方评分。
用户明确要求模型自主决定 finish；不得把官方 success/ongoing/fail、隐藏评分或其派生完成信号提供给规划器，也不得用官方状态替代或强制模型的终止决定。官方结果仅用于运行后的离线评估。
范围 RoboMME-Plus TRAIN，共50场景：15任务 easy ep0→medium ep1→hard ep2，前一难度官方成功才升级；MoveCube ep0–4 是五个独立变体（hook/poke/scoop/ball/cue），没有难度递进依赖，一个失败不能阻塞其他变体。每个task/epi本campaign总共最多3次，包括异常与回归检查，不能清零。场景由宿主按实际资源并行调度，不再限制为3路；你仍是唯一候选代码写入者。Astra medium，workspace-write，approval never，memory=true，skills=[]，seed7，100逻辑调用，400工具批次步数，官方步数不改。
用户最新要求：每遇到一次失败就分析→改进通用harness/FDS→离线验证→优先重跑同一task/episode；成功才升级，累计3次仍失败才停止该case。失败由事件立即触发，不等待30分钟。当前重点失败：{json.dumps(focus_attempt, ensure_ascii=False)}。本次优先闭环这个失败，其他任务继续并行。如果重点失败为null，则本次仅做健康汇总、成功回归或最终版本选择，不处理暂缓/等待用户/未指定的失败，不据此修改源码；这些失败由各自事件单独处理。若并发期间还有新失败，留给后续逐个分析，不要抢先决定其重试。若同一失败已耗尽3次，仍说明原因，但不能启动第4次。
只允许修改当前工作目录 src/fds/task_harness 下通用实现（skills.py除外）、fds/decision.py，或已有 common/retry.py、common/recovery.py、planners.py、cli.py 的可靠性部分；可补相关离线tests。不得修改配置、预算、数据/benchmark/评分、隐藏状态访问、skill、依赖、调度脚本或正式Meta Harness。不得git提交推送，不得访问凭据，不得发送消息/上传数据，不得调用额外模型或直接运行fds benchmark。不得将任务名、seed、答案、训练轨迹坐标固化进源码。不能把执行成功改成语义成功。
模型/预算/新工具/观察权限/架构/任务专用策略属于特殊改动：只在 rationale 中给出具体方案及影响，change_kind=needs_confirmation，user_question写清需要用户决定的内容；不要实施。普通改动最多一组有训练失败证据的通用修复，先证明原因再改。禁止只为表现“有进展”而无依据加长提示。
实验数据 {out}/results.json 和 REPORT.md；所有逐attempt的summary/audit/public帧路径均在其中。历史训练复盘可读 {root}/runs/retrospective_20261109/REPORT.md。只能分析公开观测、请求日志和官方结果，不要读取 third_party 的oracle策略/隐藏物体状态/测试数据。不要把训练成功描述为泛化。
最新汇总：{json.dumps({'totals': result['totals'], 'versions': result['versions'], 'status': result['status']}, ensure_ascii=False)}
用户明确特批的唯一次数例外（优先于本提示中默认3次/禁止第4次的表述）：{json.dumps(authorized_limits, ensure_ascii=False)}。只有此处列出的case可因第3次quota中断补跑第4次，第5次一律禁止；其他case仍最多3次。须有manifest逐case授权证据且宿主已登记额度恢复，不得由模型自行扩大授权。已恢复额度的这一次补跑不强求无关源码改动。
按需读取 {out}/status.json、control.json、reviews/ 与 optimizer/ 下过去结论，避免重复处理。重点失败必须有原因、证据及针对原因的通用改进；既有代码缺陷和提示/状态表示缺陷都可分析，不要求先证明是程序异常才优化。不得为碰运气直接重跑同代码的任务失败，也不要在尚有次数时用stop提前放弃。若当前活动版本已含有针对该失败的未试过修复，可用它重跑并说明对应差异；否则实现有证据支持的常规改进后retry。若确无可实施常规改进，给出具体阻碍，change_kind=needs_confirmation，并等待用户，不消耗剩余场景机会。纯瞬时网络错误只有已确认恢复才能建议原版本retry。永久鉴权/额度问题不能靠重跑解决，不清除全局hold；等待用户或额度恢复。对没有修改理由的成功场景无需重跑。不同代码版本上的成功集合不能合并成新版成功率。
如果修改源码，运行 .venv/bin/python -m pytest tests -q（仅离线单元测试，不是benchmark测试集），并记录具体效果假设。宿主还会独立运行测试、哈希检查和设置检查后注册新版本。不要自己调用campaign.py更改队列；通过最终结构化decisions提出retry/stop/accept。重点失败尚有次数时必须提出retry；已耗尽次数时提出stop。已有accept的成功可以在新版本需要回归时列到regression_cases，回归也占3次机会。改进后优先验证重点失败，再检查已有成功是否回退；没有回归证据不得建议替换best。不能为保留共同版本验证机会而拒绝失败case的第3次，明确剩余验证限制即可。
如果新版本让已成功任务回退，recommended_version建议回到此前有证据的版本；不得通过改变测试范围掩盖回退。只有相同cases的两轮候选均未增加官方成功且没有过程改善时可说plateau。若受3次上限限制、网络故障或覆盖不足，明确budget_limited/inconclusive，不能声称收敛。仍有未执行可达case或运行中的scene时 finish_campaign 必须false。
最终只返回schema规定JSON；evidence用本地真实路径，decisions只引用最新已完成attemptid。regression_cases引用case_id（例如PatternLock_ep0），不限具体任务但必须有已成功证据和剩余次数。recommended_version填已存在且有依据的版本ID；若本次新改动尚未注册留空，宿主会给新版本ID。clear_infrastructure_hold仅在你已核对故障解除或属于已修复局部问题时true。给出所有失败原因及本轮修改/不修改理由。\n'''


def stop_tree(process):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def run_reviewer(root, out, directory, work, prompt):
    schema = directory / 'response-schema.json'
    atomic(schema, response_schema())
    (directory / 'prompt.md').write_text(prompt)
    command = ['codex', 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
               '--sandbox', 'workspace-write', '--json', '--model', MODEL, '--cd', str(work),
               '--output-schema', str(schema), '-o', str(work / 'review.json'),
               '-c', 'approval_policy="never"', '-c', 'model_reasoning_effort="medium"',
               '-c', 'forced_login_method="chatgpt"', '-c', 'model_provider="openai"',
               '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
               '-c', 'skills.include_instructions=false']
    for feature in ('multi_agent', 'apps', 'plugins', 'browser_use', 'computer_use', 'image_generation',
                    'hooks', 'memories', 'skill_search', 'unbounded_connection_retries'):
        command.extend(['--disable', feature])
    command.extend(['--', '-'])
    atomic(directory / 'request.json', {'model': MODEL, 'effort': EFFORT, 'sandbox': 'workspace-write',
                                      'skills_include_instructions': False,
                                      'approval': 'never', 'command': command, 'started_at': now()})
    with (directory / 'events.jsonl').open('xb') as events, (directory / 'stderr.log').open('xb') as errors:
        proc = subprocess.Popen(command, cwd=work, env=reviewer_environment(root, work),
                                stdin=subprocess.PIPE, stdout=events, stderr=errors, text=True, start_new_session=True)
        atomic(directory / 'process.json', {'pid': proc.pid, 'create_time': psutil.Process(proc.pid).create_time()})
        try:
            proc.communicate(prompt, timeout=1500)
        except BaseException:
            stop_tree(proc)
            raise
    usage = {'input_tokens': 0, 'output_tokens': 0, 'completed_turns': 0, 'usage_missing': False}
    for line in (directory / 'events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'turn.completed':
            usage['completed_turns'] += 1
            for key in ('input_tokens', 'output_tokens'):
                value = (event.get('usage') or {}).get(key)
                usage['usage_missing'] |= value is None
                usage[key] += value or 0
    usage['usage_missing'] |= usage['completed_turns'] == 0
    atomic(directory / 'usage.json', usage)
    if proc.returncode or not (work / 'review.json').exists():
        raise RuntimeError(f'Optimizer exit={proc.returncode}; evidence retained at {directory}')
    proposal = read(work / 'review.json')
    import jsonschema
    jsonschema.validate(proposal, response_schema())
    atomic(directory / 'proposal.json', proposal)
    return proposal


def ensure_coordinator(root, out):
    state = safe_read(out / 'status.json')
    command = [str(root / '.venv/bin/python'), str(Path(__file__).with_name('campaign.py')),
               'run', '--out', str(out)]
    if identity_alive(state.get('pid'), state.get('create_time'), command):
        return
    if identity_alive(state.get('pid'), state.get('create_time')):
        raise RuntimeError('Coordinator identity command mismatch; refusing duplicate launch')
    if campaign.build_state(out)['status'] == 'completed':
        return
    # campaign.run takes an exclusive singleton lock and adopts verified existing workers.
    with (out / 'logs/coordinator.log').open('ab') as stream:
        subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def validate_failure_decisions(out, state, proposal, target_version, changed, focus_attempt, candidate_sha=None):
    manifest = read(out / 'campaign.json')
    """A task retry must evaluate a repair, never spend a slot on a blind replay."""
    latest = {c['past_attempts'][-1]['id']: c for c in state['cases'] if c['past_attempts']}
    focus_decisions = [d for d in proposal['decisions'] if d['attempt_id'] == focus_attempt]
    if focus_attempt and len(focus_decisions) != 1:
        raise ValueError('The focused failure requires exactly one explicit decision')
    for decision in proposal['decisions']:
        case = latest.get(decision['attempt_id'])
        if not case:
            raise ValueError('Decision must reference a latest known attempt')
        attempt = case['past_attempts'][-1]
        if attempt.get('success'):
            continue
        if not focus_attempt or attempt['id'] != focus_attempt:
            raise ValueError('Other failures require their own focused diagnosis')
        limit = campaign.attempt_limit(manifest, case['id'], case['past_attempts'])
        if len(case['past_attempts']) >= limit:
            if decision['action'] != 'stop':
                raise ValueError(f'Exhausted failure must stop; cap {limit}, no fourth attempt without named quota authorization and never a fifth')
            continue
        if decision['action'] != 'retry':
            raise ValueError('A failed case with remaining attempts must improve/retry or await confirmation')
        quota_extension = (limit == 4 and len(case['past_attempts']) == 3 and attempt.get('failure_category') == 'quota'
                           and focus_attempt in read(out / 'control.json').get('cleared_infrastructure_attempts', []))
        if attempt.get('failure_category') in ('auth', 'quota') and not quota_extension:
            raise ValueError('Account recovery requires independent confirmation')
        tried = [read(out / 'versions' / a['version'] / 'version.json').get('source_sha256')
                 for a in case['past_attempts']]
        after = candidate_sha if changed else read(out / 'versions' / target_version / 'version.json').get('source_sha256')
        different_source = bool(after and all(before and before != after for before in tried))
        recovered_network = (attempt.get('failure_category') in ('network', 'transport')
                             and proposal.get('clear_infrastructure_hold'))
        if not different_source and not recovered_network and not quota_extension:
            raise ValueError('Task/program failure retry requires an improved source version not previously tried')


def apply_review(root, out, directory, work, proposal, version, focus_attempt=None):
    base = out / 'versions' / version
    manifest = read(out / 'campaign.json')
    if ancillary_hashes(work) != read(directory / 'ancillary-hashes.json'):
        raise ValueError('Reviewer changed configuration/scripts/dependencies outside authorized scope')
    changed = validate_candidate_scope(base, work, manifest['tasks'])
    if proposal['change_kind'] == 'needs_confirmation':
        atomic(out / 'NEEDS_USER_DECISION.json', {'proposal': str(directory / 'proposal.json'),
               'attempt_id': focus_attempt, 'question': proposal['user_question'],
               'rationale': proposal['rationale'], 'created_at': now()})
        return {'status': 'needs_user_decision', 'changed_files': changed}
    if changed and proposal['change_kind'] != 'routine':
        raise ValueError('Unreported source edits in reviewer workspace')
    if changed and not focus_attempt:
        raise ValueError('A source repair requires a focused failure; health checks cannot bypass a case hold')
    candidate_sha = hashes(work / 'src')
    ancillary_sha = ancillary_hashes(work)
    parent_meta = campaign.verify_version(out, version)
    recommendation = proposal['recommended_version']
    # A case-local fallback is a retry target, never a global policy promotion.
    target_version = recommendation if focus_attempt and not changed and recommendation else version
    if target_version != version:
        campaign.verify_version(out, target_version)
    validate_failure_decisions(out, campaign.build_state(out), proposal,
                              target_version,
                              changed, focus_attempt, candidate_sha)
    if changed:
        with (directory / 'validation.log').open('xb') as log:
            validation = subprocess.run([str(root / '.venv/bin/python'), '-m', 'pytest', str(root / 'tests'), '-q'],
                cwd=work, env=reviewer_environment(root, work), stdout=log, stderr=subprocess.STDOUT, timeout=240)
            if validation.returncode == 0:
                validation = subprocess.run([str(root / '.venv/bin/python'), '-m', 'pytest', 'tests', '-q'],
                    cwd=work, env=reviewer_environment(root, work), stdout=log, stderr=subprocess.STDOUT, timeout=240)
        if validation.returncode:
            return {'status': 'candidate_rejected', 'reason': 'offline_tests_failed', 'changed_files': changed}
        if hashes(work / 'src') != candidate_sha or ancillary_hashes(work) != ancillary_sha:
            raise ValueError('Candidate bytes changed during offline validation')
        with lock(out / 'state.lock'):
            control = read(out / 'control.json')
            if control['active_version'] != version:
                raise ValueError('Active source changed during review; candidate must be rebased and revalidated')
            if campaign.verify_version(out, version)['source_sha256'] != parent_meta['source_sha256']:
                raise ValueError('Parent source changed during review')
            target_version = campaign.snapshot(out, proposal['rationale'], source=work)
            metadata_path = out / 'versions' / target_version / 'version.json'
            metadata = read(metadata_path)
            if metadata['source_sha256'] != candidate_sha:
                raise ValueError('Registered snapshot differs from validated candidate')
            metadata.update(parent_version=version, parent_source_sha256=parent_meta['source_sha256'],
                            lineage=[*parent_meta.get('lineage', [version]), target_version])
            atomic(metadata_path, metadata)
            atomic(out / 'versions' / target_version / 'validation.json', {
                'passed': True, 'kind': 'offline_unit_tests', 'source_sha256': candidate_sha,
                'parent_version': version, 'parent_source_sha256': parent_meta['source_sha256'],
                'log': str(directory / 'validation.log'), 'proposal': str(directory / 'proposal.json'),
                'performance_validated': False, 'created_at': now(),
            })
            control['active_version'] = target_version
            control['candidate_status'] = 'training_evaluation_pending'
            atomic(out / 'control.json', control)
    decisions = []
    with lock(out / 'state.lock'):
        state = campaign.build_state(out)
        latest = {c['id']: c['past_attempts'][-1] for c in state['cases'] if c['past_attempts']}
        requests = list(proposal['decisions'])
        for case_id in proposal['regression_cases'] if changed else []:
            previous = latest.get(case_id, {})
            if previous.get('success') and previous.get('version') != target_version:
                requests.append({'attempt_id': previous['id'], 'action': 'retry', 'reason': 'regression: candidate validation'})
        for decision in requests:
            try:
                attempt = next((a for a in latest.values() if a['id'] == decision['attempt_id']), {})
                if decision['action'] == 'retry' and attempt.get('success') and not changed:
                    raise ValueError('Success regression requires a newly validated coherent candidate')
                campaign.review(out, decision['attempt_id'], decision['action'], decision['reason'],
                                target_version, proposal['evidence'])
                decisions.append({**decision, 'applied': True})
            except ValueError as error:
                decisions.append({**decision, 'applied': False, 'error': str(error)})
        control = read(out / 'control.json')
        if proposal['clear_infrastructure_hold']:
            reviewed_ids = {focus_attempt} if focus_attempt else set()
            ids = set(control.get('cleared_infrastructure_attempts', []))
            # Auth/quota/permissions cannot be cleared by a model assertion; they require
            # an independently recorded user/account recovery decision.
            ids.update(a['id'] for a in state['completed'] if a['id'] in reviewed_ids
                       and a.get('failure_category') in ('network', 'transport', 'infrastructure'))
            uncleared = [a for a in state['completed'] if campaign.requires_global_hold(a) and a['id'] not in ids]
            existing_hold = control.get('hold_reason')
            if existing_hold and existing_hold.startswith('Infrastructure diagnosis required:'):
                existing_hold = None
            control.update(hold_reason=('Infrastructure diagnosis required: ' + ', '.join(a['id'] for a in uncleared)) if uncleared else existing_hold,
                           hold_until=None, cleared_infrastructure_attempts=sorted(ids))
        atomic(out / 'control.json', control)
    result = {'status': 'reviewed', 'version': target_version, 'changed_files': changed, 'decisions': decisions,
              'recommendation': {'version': recommendation or None,
                                 'scope': 'focused_case_only' if focus_attempt and not changed else 'advisory_only',
                                 'attempt_id': focus_attempt, 'global_activation': bool(changed)},
              'skipped_regression_cases': proposal['regression_cases'] if not changed else []}
    state = campaign.build_state(out)
    if proposal['finish_campaign'] and not state['active_runs']:
        reachable = [c for c in state['cases'] if c['state'] in ('pending', 'retry', 'running', 'waiting_review', 'needs_diagnosis')]
        if not reachable and proposal['stop_basis'] in ('all_success', 'budget_limited', 'plateau'):
            evidence = version_evidence(write_report(out))
            target_version = evidence['best_version']
            campaign.verify_version(out, target_version)
            all_success = all(c['state'] == 'success' for c in state['cases']) and len(evidence['successful_cases']) == len(state['cases'])
            stop_basis = 'all_success' if all_success else 'plateau' if evidence['comparable_plateau_evidence'] else 'budget_limited'
            final = {**result, 'status': 'completed',
                     'recommended_version': target_version, 'proposal': str(directory / 'proposal.json'),
                     'finished_at': now(), 'source_path': str(out / 'versions' / target_version / 'src'),
                     'validation_scope': 'See version-specific results; never combine mixed-version successes.',
                     'stop_basis': stop_basis, 'selection_evidence': evidence}
            atomic(out / 'FINAL.json', final)
            (out / 'FINAL.md').write_text('# Training optimization result\n\n' + proposal['summary'] + '\n\n' +
                proposal['rationale'] + '\n\nSource: ' + final['source_path'] + '\n\nVerified stop basis: ' + stop_basis +
                '\n\nHost-verified selection/coverage:```json\n' + json.dumps(evidence, ensure_ascii=False, indent=2) + '\n```\n')
            return final
    return result


def _checkpoint(out, focus_attempt_id=None):
    root = Path(read(out / 'campaign.json')['root'])
    with lock(out / 'state.lock'):
        state = campaign.build_state(out)
        old = safe_read(out / 'status.json')
        for key in ('pid', 'create_time', 'max_concurrency', 'resources'):
            if key in old:
                state[key] = old[key]
        atomic(out / 'status.json', state)
    result = write_report(out)
    focus_attempt = next((r for r in state['completed'] if r['id'] == focus_attempt_id), None)
    if focus_attempt_id and (not focus_attempt or focus_attempt.get('success')):
        raise ValueError('Failure-triggered checkpoint needs a completed unsuccessful attempt')
    control = read(out / 'control.json')
    if control.get('halt'):
        return {'status': 'paused_by_user'}
    cleared = set(control.get('cleared_infrastructure_attempts', []))
    permanent = [r for r in state['completed'] if r.get('failure_category') in ('auth', 'quota') and r['id'] not in cleared]
    if permanent:
        decision = {'created_at': now(), 'kind': 'account_recovery',
                    'attempt_ids': [r['id'] for r in permanent],
                    'question': '账户鉴权或额度阻止请求；请恢复账户后确认继续。现有结果与剩余场景机会已保留。'}
        atomic(out / 'NEEDS_USER_DECISION.json', decision)
        return {'status': 'needs_user_decision', 'reason': 'account_recovery_required'}
    owner = control.get('reviewer_owner') or {}
    if identity_alive(owner.get('pid'), owner.get('create_time')) and owner.get('pid') != os.getpid():
        raise RuntimeError('Another optimizer owns the review hold')
    for path in (out / 'optimizer').glob('*/process.json'):
        process = read(path)
        if identity_alive(process.get('pid'), process.get('create_time')):
            raise RuntimeError('Surviving reviewer process must be diagnosed before another review')
    completed = sorted(r['id'] for r in result['rows'] if r['terminal'])
    fingerprint = hashlib.sha256(json.dumps({'completed': completed, 'control': control}, sort_keys=True).encode()).hexdigest()
    previous = safe_read(out / 'optimizer/last-checkpoint.json')
    if not completed or (not focus_attempt_id and previous.get('fingerprint') == fingerprint
                         and state['status'] != 'completed'):
        ensure_coordinator(root, out)
        return {'status': 'no_new_results', 'completed': len(completed)}
    directory = out / 'optimizer' / datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True)
    version = control['active_version']
    work = make_workspace(root, out, directory, version)
    atomic(directory / 'ancillary-hashes.json', ancillary_hashes(work))
    atomic(directory / 'reviewed-attempts.json', {'ids': completed})
    atomic(directory / 'trigger.json', {'kind': 'failure' if focus_attempt_id else 'health_check',
                                       'attempt_id': focus_attempt_id, 'created_at': now()})
    with lock(out / 'state.lock'):
        control = read(out / 'control.json')
        # Source snapshots isolate active scenes. Only the failed case is held;
        # independent scenes keep using the current version while we diagnose.
        control['reviewer_hold'] = False
        control['reviewer_owner'] = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(), 'directory': str(directory)}
        atomic(out / 'control.json', control)
    try:
        prompt = reviewer_prompt(root, out, work, result, version, focus_attempt)
        proposal = run_reviewer(root, out, directory, work, prompt)
        outcome = apply_review(root, out, directory, work, proposal, version, focus_attempt_id)
        outcome['directory'] = str(directory)
        outcome['focus_attempt_id'] = focus_attempt_id
        atomic(directory / 'applied.json', outcome)
        atomic(out / 'optimizer/last-checkpoint.json', {'fingerprint': fingerprint, 'directory': str(directory),
                                                       'outcome': outcome, 'finished_at': now()})
        return outcome
    finally:
        with lock(out / 'state.lock'):
            control = read(out / 'control.json')
            if (control.get('reviewer_owner') or {}).get('directory') == str(directory):
                control['reviewer_hold'] = False
                control.pop('reviewer_owner', None)
                atomic(out / 'control.json', control)
        ensure_coordinator(root, out)
        write_report(out)


def checkpoint(out, focus_attempt_id=None):
    (out / 'optimizer').mkdir(exist_ok=True)
    with lock(out / 'optimizer/checkpoint.lock', nonblocking=True):
        return _checkpoint(out, focus_attempt_id)


def pending_failures(out, state, ledger, timestamp):
    """Latest completed failures only; repeated file notifications never repeat AI work."""
    pending = []
    for case in state.get('cases', []):
        past = case.get('past_attempts', [])
        if not past:
            continue
        attempt = past[-1]
        if attempt.get('state') != 'finished' or attempt.get('success'):
            continue
        if (out / 'reviews' / (attempt['id'] + '.json')).exists():
            continue
        record = ledger.get(attempt['id'], {})
        if record.get('acknowledged') or record.get('needs_user'):
            continue
        if record.get('retry_after', 0) > timestamp:
            continue
        pending.append(attempt)
    return sorted(pending, key=lambda a: (a.get('failure_category') not in ('auth', 'quota'),
                                         a.get('failure_category') == 'task'))


def record_failure_review(out, attempt_id, outcome):
    path = out / 'optimizer/failure-reviews.json'
    ledger = read(path)
    previous = ledger.get(attempt_id, {})
    acknowledged = any(d.get('attempt_id') == attempt_id and d.get('applied')
                       for d in outcome.get('decisions', []))
    failures = 0 if acknowledged else previous.get('analysis_failures', 0) + 1
    record = {'acknowledged': acknowledged, 'analysis_failures': failures,
              'outcome': outcome, 'updated_at': now()}
    if not acknowledged:
        # A rejected patch or failed optimizer request does not spend a scene
        # attempt. Bound reviewer recovery, too; never spin on the same event.
        record['retry_after'] = time.time() + INTERVAL
        record['needs_user'] = outcome['status'] == 'needs_user_decision' or failures >= 3
    ledger[attempt_id] = record
    atomic(path, ledger)
    if failures >= 3 and outcome['status'] != 'needs_user_decision':
        atomic(out / 'NEEDS_USER_DECISION.json', {
            'kind': 'optimizer_recovery', 'attempt_id': attempt_id, 'created_at': now(),
            'question': '同一失败的分析/候选验证连续三次未完成；已保留场景机会，请检查分析日志后恢复。',
            'evidence': str(path),
        })


def serve(out):
    (out / 'optimizer').mkdir(exist_ok=True)
    with lock(out / 'optimizer/optimizer.lock', nonblocking=True):
        identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time(),
                    'interval_seconds': INTERVAL, 'trigger_mode': 'failure_event_plus_health_check'}
        events = FileEvents()
        events.watch(out, names={'status.json', 'control.json', 'FINAL.json'})
        events.watch(out / 'reviews')
        events.watch(out / 'optimizer', names={'failure-reviews.json'})
        previous_due = safe_read(out / 'optimizer/status.json').get('next_check_at')
        first_delay = INTERVAL
        if previous_due:
            try:
                first_delay = max(0, min(INTERVAL, (datetime.fromisoformat(previous_due) - datetime.now(UTC)).total_seconds()))
            except (ValueError, TypeError):
                pass
        deadline = time.monotonic() + first_delay
        failures = 0
        try:
            while True:
                if safe_read(out / 'FINAL.json'):
                    return
                control = read(out / 'control.json')
                if control.get('halt'):
                    atomic(out / 'optimizer/status.json', {**identity, 'status': 'paused_by_user', 'updated_at': now()})
                    return
                # The coordinator publishes only durable, PID-verified results.
                state = safe_read(out / 'status.json')
                pending = pending_failures(out, state, read(out / 'optimizer/failure-reviews.json'), time.time())
                focus = pending[0]['id'] if pending else None
                health_due = time.monotonic() >= deadline
                if not focus and not health_due:
                    delay = max(0, deadline - time.monotonic())
                    previous = safe_read(out / 'optimizer/status.json')
                    atomic(out / 'optimizer/status.json', {**identity,
                           'status': 'needs_user_decision' if previous.get('status') == 'needs_user_decision' else 'waiting',
                           'last_result': previous.get('last_result', previous.get('status')),
                           'next_check_at': (datetime.now(UTC) + timedelta(seconds=delay)).isoformat(),
                           'updated_at': now()})
                    # inotify is the failure trigger; the timeout is only the
                    # authorized 30-minute health check, with no sleep polling.
                    while time.monotonic() < deadline:
                        ready = select.select([events.fd], [], [], max(0, deadline - time.monotonic()))[0]
                        if not ready or events.drain():
                            break
                    continue
                atomic(out / 'optimizer/status.json', {**identity, 'status': 'reviewing',
                       'trigger': 'failure' if focus else 'health_check', 'focus_attempt_id': focus,
                       'next_check_at': None, 'updated_at': now()})
                try:
                    outcome = checkpoint(out, focus)
                    failures = 0
                except Exception as error:
                    failures += 1
                    outcome = {'status': 'review_failed', 'error_type': type(error).__name__,
                               'error': str(error), 'consecutive_failures': failures}
                if focus:
                    record_failure_review(out, focus, outcome)
                atomic(out / 'optimizer/status.json', {**identity, **outcome,
                       'last_result': outcome['status'], 'updated_at': now()})
                if health_due:
                    deadline = time.monotonic() + INTERVAL
                if outcome['status'] in TERMINAL or outcome['status'] == 'paused_by_user' or failures >= 3:
                    return
        finally:
            events.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--once', action='store_true', help='One authorized analysis checkpoint; may call Codex.')
    args = parser.parse_args()
    if args.once:
        output = args.out.resolve()
        failures = pending_failures(output, campaign.build_state(output),
                                    read(output / 'optimizer/failure-reviews.json'), time.time())
        print(json.dumps(checkpoint(output, failures[0]['id'] if failures else None)))
    else:
        serve(args.out.resolve())
