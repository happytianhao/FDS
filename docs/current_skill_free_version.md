# 当前 harness + memory + FDS 无 skill 版本

## 版本定位

推荐入口为 [configs/robomme_plus/task_harness.toml](../configs/robomme_plus/task_harness.toml)。
本次将 v029 的通用恢复修复、新的执行后核对和 Codex 指令隔离整理进同一份正式源码。
用本文件所在的 Git 提交标识代码版本；它**不是历史不可变 v029 快照本身**。

这是目前整理出的工程候选，尚不能称为训练集 50 项上已验证的最佳版本。
当前 E4 汇总记录覆盖 29/50 个不同案例，共 80 次尝试，21 次成功、59 次未成功，
其中 13 个不同案例曾经成功。这是多个版本与重复尝试的历史汇总，
不能作为本候选的成功率；未成功还包含基础设施错误和状态未知，不能全部解释为策略失败。
历史 v029 没有实际场景尝试，最终统一版本的验证记录仍为空。

历史证据源为本地
`runs/robomme_plus/task_harness/e4_astra_medium_adaptive_train/results.json`、
`control.json` 与 `status.json`；结果目录保留本地，不随代码发布。
本次整理没有启动新训练、修改历史版本、重置尝试次数或更新后台实验配置。
独立的 [接口机制小样本实验](robomme_interface_review/results.md) 也不是本候选的成绩。

## 默认配置

| 项目 | 当前值 |
|---|---|
| Benchmark / split | RoboMME-Plus / train |
| Planner | Codex，gpt-6-astra，medium |
| 权限 | workspace-write，approval never；禁用 shell 与外部工具 |
| Memory | episode 内持续结构化记忆，记录来源、revision、执行事件 |
| FDS | enabled=true，contract_mode=full，postcheck_enabled=true |
| Skills | task_harness.skills=[]；Codex skills.include_instructions=false |
| Policy / examples | 均关闭 |
| Seed / episode | seed=7；默认 episode 0 |
| 预算 | 每集 max_calls=100；每个工具 batch 的 max_tool_steps=400 |
| 无效决策纠正 | 每组最多 2 次，消耗原调用预算 |
| 请求恢复 | transport_retries=3；最多一次 60 秒冷却；最多两个请求窗口 |
| 环境 | 项目内 .venv；Python 3.11 或 3.12 |

Memory 是 task harness 的固有组件，不需要额外的 `memory=true` 配置项。
`max_tool_steps` 限制每个工具 batch 的环境步数，不是工具调用数量或整集总步数；
整集环境步数另有 max_steps=10000 上限。
每个逻辑 planner 调用最多有 8 次传输请求，全部已知 token 单独计入用量，
并不把重试解释为免费调用。认证、额度等永久错误不会无限重试。

50 个训练案例指 15 个任务各 easy/medium/hard 三档，加 MoveCube 的 5 个独立变体。
自适应调度器仍按前一难度成功才升级、每案例合计最多 3 次组织；
既有的 3 个额度中断补跑例外单独记录，不扩展为一般第四次授权。
调度器是独立脚本，不会随 `fds run` 自动启动。

## 本次整理的关键改动

1. **请求失败可恢复且可审计。** 区分瞬时网络/限流与认证/额度/非法请求，
   保留每次请求、退避与冷却记录。已有完整响应或执行工具后禁止重放。
   缺失 token 保持未知；恢复耗尽仍保留真实失败原因，不能保证所有异常消失。
2. **拒绝反馈可用于纠正。** 返回不可用帧引用、记忆容量、被拒提案等具体原因；
   拒绝不执行工具、不污染有效 memory。保留最近物理反馈，避免辅助查询覆盖它。
3. **证据供应更准确。** 同帧同相机去重，区分当前供应、保留关键帧和历史文字；
   历史文字不会自动恢复旧图片或赋予新引用权限。
4. **执行后逐项核对。** 将候选与实际工具索引/参数哈希关联，给 expected outcome
   和 falsifier 稳定编号，下一次正常调用核对 met / violated / unknown。
   新反馈修正执行事件时校验来源、revision、目标与记忆更新的一致性，支持延迟反馈。
5. **Codex 指令隔离。** 通过 `model_instructions_file` 替换 coding base；
   任务规则与动态 user 观测拆开，禁用 Codex 技能说明注入，并记录提示哈希。
   本机验证使用 Codex 0.163.0；其 Astra 模板将 base 编码为 developer 消息。
6. **可复现的实验辅助工具。** 纳入不可变版本队列、资源调度、逐失败分析入口、
   结果汇总及小组件发布器。新增 `psutil` 开发依赖；uv 锁文件改用清华镜像，
   原有依赖版本保持不变。账号凭据、原始轨迹、虚拟环境和第三方 checkout 不入库。

新接口仍由模型判断画面含义。宿主可以检查结构、可见来源和显式依赖，
不能证明图片确实支持某个假设，也不能自动发现全部漏报依赖。
工具参数绑定是执行关联审计，不是候选自然语言与坐标正确性的证明。
官方 success、机械 verdict 和模型的 outcome review 始终分开。

## 使用与验证

首次安装：`uv sync --locked --extra dev --extra api`。
已有模拟器环境不要直接运行会清理额外包的 `uv sync`；按
[README](../README.md#installation) 的环境安装步骤维护项目 `.venv`。
Codex 认证在环境外配置；如本机需要代理，设置
`HTTP_PROXY=http://127.0.0.1:21084` 和 `HTTPS_PROXY=http://127.0.0.1:21084`。

```bash
.venv/bin/fds doctor
.venv/bin/fds run \
  --config configs/robomme_plus/task_harness.toml \
  --task BinFill --episodes 0 \
  --output runs/robomme_plus/task_harness/skill_free_candidate/binfill_ep0
```

运行前选择新的输出目录，保留旧版本结果。评估需冻结当前提交和配置，
先遵循既有训练案例与尝试上限；训练选择完成前，不把跨版本成功并集称为最优成绩。
同代码的旧 FDS 核对协议对照可用 `postcheck_enabled=false`；F0 需
`enabled=false, contract_mode="off"`，其 memory、模型和预算保持一致。

离线检查：

```bash
.venv/bin/python -m pytest tests
.venv/bin/python scripts/smoke_fds.py
.venv/bin/python scripts/diagnostics/verify_codex_prompt_isolation.py
```

本候选全量离线测试 139 项通过。smoke 覆盖 full/off 契约；提示隔离诊断使用
临时无认证环境和 localhost 拒绝服务，无真实模型请求。新执行后核对协议尚未进行
机器人训练性能验证，不能从离线测试推断成功率提升。

用量读取 `metrics.json`、episode `summary.json` 和逐请求 `stats.json`；
缺失部分必须单独列出。费用仅用实际单价计算：

```text
estimated_cost = prompt_tokens / 1e6 * input_price
               + completion_tokens / 1e6 * output_price
```

详细说明：[执行后核对](fds_postcheck_implementation.md)、
[请求恢复](request_reliability.md)、[提示注入与技能隔离](robomme_interface_review/prompt_injection.md)、
[自适应训练约定](adaptive_train_protocol.md)。
