# 决策恢复、运动反馈与实验调度改动说明

本次改动解决两类工程问题：模型返回不合法决策会直接结束 episode；实验外层脚本会把第一次可重试的请求错误升级为整场终止。同时，针对 train 失败记录，补充了动作过程反馈与决策核验提示。它们帮助模型发现问题，但不构成新的自动视觉判定器。

## 1. 非法决策变为有预算的纠正流程

涉及 `src/fds/decision.py`、`task_harness/protocol.py`、`task_harness/runtime.py` 和三个真实 planner 后端。

之前，JSON、schema 或 FDS contract 校验异常会直接向上抛出，episode 结束。现在模型输出错误使用 `DecisionValidationError`，包含 `code`、`path`、`message` 和原始 `rejected_response`。覆盖无效 JSON、非有限数值、schema、批次结构、重复 review frame ID，以及 contract 引用/前置条件错误。未来采样字段 `observation_steps` 已移除，继续输出该字段也会被 schema 拒绝。无效 schema 本身、普通程序错误和 API 故障不伪装为决策错误。

执行顺序是：

1. 获取响应并严格校验。
2. 若被拒绝，写入 `audit/call_XXXX/decision_rejection.json` 及 trace。
3. 保持当前环境、观测窗口、已接受 memory 和 keyframes；不执行被拒绝响应中的工具或 memory 修改。
4. 下次正常 planner 调用通过 `decision_recovery.previous_rejection` 收到错误与原响应。纠正轮只自动输入已保留关键帧，不重放未保留的旧观测；需要旧画面时，可先调用 `review_frames`，收到图片后再生成完整替代决策。环境与已接受的记忆仍保持不变。
5. 纠正通过后恢复正常执行，并记录 `decision_recovery.json`；连续纠正耗尽则结束。

`task_harness.max_decision_corrections` 默认 2，允许 0–5。默认情况下，一次初始非法响应后最多再发两次纠正请求；总计连续三次非法响应后 `stop_reason=decision_recovery_exhausted`。所有纠正都占原有 `max_calls`，若先耗尽总调用预算则为 `budget_exhausted`。有效决策会重置连续错误计数。

`summary.json.decision_recovery` 记录上限、拒绝数、纠正调用数、恢复成功序列数和最后一次拒绝。恢复成功只代表决策通过校验，不代表任务成功。F0 与 F1 共用此流程，F0 不加入 FDS contract 指令。

## 2. 区分决策纠正和传输重试，修正 usage 统计

`src/fds/common/retry.py` 将无效模型决策标记为 `decision_rejected`，避免把它当成传输故障再次发送同一请求。已有传输重试仍只针对允许重试的故障，重试期间环境保持不变，逐 attempt 保存记录。

此前 `failed_attempt_usage_unknown` 把所有未接受响应都计为未知用量。现在检查是否实际收到输入和输出 usage；即使响应因 contract 不合法被拒绝，若 usage 已返回，也不会计入未知。缺失 usage 仍明确标记，已知 token 合计不能充当完整计费数。

预算区别：`max_calls=100` 是逻辑决策调用上限，包含决策纠正；传输重试是该调用内部的额外请求，必须另计 attempts 和 token。实验模板沿用每调用最多 3 次传输重试，并将 `pause_seconds` 设为 0 以关闭人工恢复等待。它不是“最多 100 个网络请求”的保证。

## 3. 从最后一个终点反馈扩展到逐段实测反馈

`task_harness/tools.py` 在每个原始 waypoint 的最后一个内部子段执行完、且有返回观测时，生成 `target_observations`，包括：

- 对应的原始 target / resolved target 编号、实际执行步偏移与 frame。
- 通过实测关节计算的 TCP pose、位置误差和旋转误差。
- 原始目标是否已完整下发；`semantic_success` 仍为 `unverified`。

提前终止、未到达或没有观测的后续端点不会被补造。此改动没有调整 IK、运动插值、动作时序或步数。原来的最终 tracking 反馈保留。

`runtime.py` 向 planner 提供 `recent_motion_history`，保留最近六次 `move_ee` 请求和反馈，跨辅助调用持续可见，帮助识别反复微调却没有进展的情况。历史 frame ID 不恢复图片，不扩展 memory 的合法引用范围。

代价是 prompt 可能变长；历史有六次调用的上限，但每次调用仍可能包含多个目标。是否降低总调用/token 消耗需要实验验证。

## 4. 提示明确前置证据、事件顺序与语义边界

`task_harness/contracts.py` 增补通用说明：

- 所有 candidate（包括未选中的候选）的 `requires` 都必须指向已有 assumption ID。
- `required_evidence` 表示所选批次的执行前置条件；本批次打算获取的信息应写在 `expected_outcome`。
- 接近按钮、发出命令、出现高亮或经过一段时间，不能单独证明事件已完成。
- 对视频中的“第几次”“之前/之后”要求，检查关键转换的前后帧，区分经过、接触和释放；目标移动时跟踪身份，不能把旧坐标当身份。
- TCP 到达与物体到达不同；释放前核对目标身份、物体位置与抓持偏移。
- 停滞时结合近期实测反馈重新区分接触、姿态、抓持偏移等假设，避免无新证据的重复微调。

这些内容是任务无关的 harness 指导，未加入答案、隐藏物体状态或评分阈值，没有新增 skill。FDS 严格校验及 `execution_verdict` 判定未放宽：命令执行成功不等于 semantic success。

## 5. Codex 执行设置与审计

`src/fds/planners.py` 按实验约定将 Codex 从 `read-only` 改为 `workspace-write`，显式设置 `approval_policy="never"`。仍在临时推理工作目录中运行，保留已有 ephemeral、结构化输出及外部工具禁用设置。每次 stats 写入 `sandbox_mode` 和 `approval_policy`，便于检查实际请求。

Codex、Responses、Chat 三个后端都使用 `Decision.parse_json`，因此 JSON 解码错误可进入同一纠正机制。此次没有修改正式模型配置；实验继续使用 `gpt-6-astra`、effort `medium`。

## 6. 实验运行脚本：等待有限恢复、最多三个场景并行

运行时副本仍保存在忽略的 `runs/` 下；可复用模板见 `scripts/experiments/recovery_train/`。本次提交不改变已经启动的运行副本。

| 文件 | 作用 |
|---|---|
| `run_one.py` | 隔离单个 train ep0 场景，保留错误、PID/create_time 和输出。等待 worker 自己完成有限传输重试，不因中间 `model_request_error` 立刻杀掉场景；超时/终止时清理其进程树。 |
| `run_campaign.py` | 使用独占锁管理队列，最多三个场景；接管身份匹配的现有 runner，跳过终态任务，不覆盖已有输出；基于 Linux 进程退出事件补位。执行前核对源码/配置快照；runner 消失但 worker 仍存活时停止调度并报告。 |
| `collect_results.py` | 汇总已终结场景，逐 attempt 累加 token，排除 canonical 副本重复计数；保留缺失 usage、缺失步骤、错误及请求设置问题。 |
| `render_report.py` | 生成结果表、contract/verdict/recovery 统计、历史配对检查和输出路径；历史参考缺失时不宣称配对通过。 |
| `progress.py` / `monitor_once.py` | 单次进度与协调器存活检查，可同时列出最多三个活跃任务；不启动实验。 |
| `widget_publisher.py` / `Widget.vue` | 文件变更触发数据推送，合并窗口约 15 秒、60 秒心跳；展示多场景进度、用量和额度。变化值通过 data 通道传输。 |
| `read_quota.py` | 通过 Codex app-server 的 `account/rateLimits/read` 读取额度与时间；不创建模型 turn。不可读时返回 unknown。 |

模板中的小组件目标由部署时的 `widget-target.json` 提供；凭据留在权限为 600 的用户配置文件中，不进入 Git。组件背景透明并使用主题变量。模板不会调用定时任务 Arrange API，也不会创建 AI 轮询任务；30 分钟唤醒由平台单独管理。

并发限制针对 scene runner，单场景内部仍按自身调用/动作预算执行。并行会改变墙钟时间、共享资源和请求重叠，不应把切换前后的 wall time 当作纯算法加速比。

## 7. 验证与结论边界

新增测试覆盖：

- 无效 JSON/schema/batch/contract/provenance 的拒绝和纠正，拒绝前后状态与图片不变。
- 连续错误上限、总调用预算、恢复后重置连续计数、F0 无 FDS 指令污染。
- 其他错误继续传播；有 usage 的非法响应正确计费；非法决策不触发传输重试。
- 每段原始 waypoint 的实测反馈，提前终止不补造后续端点，辅助调用保留历史。
- 传输重试不推进环境且在上限处停止。
- 进程接管、并发上限三、不重复启动，以及一个子进程失败后继续处理队列。

本地验证不调用模型、不执行 benchmark test split。E2 的六项新增成功来自官方判定；该轮没有发生在线 decision correction，不能声称成功来自纠正被实际触发。训练集失败驱动的修改与原 F0 属于不同版本，不构成完整同版本消融；E3 尚在运行时不对最终收益作保证。

本次提交前验证：`.venv/bin/python -m pytest tests` 为 **23 passed**；涉及的 Python 源码、测试和实验模板通过 ruff；`git diff --check` 通过。Vue 的并行展示已通过平台源码发布及 data 读回核验，未声称进行了浏览器端自动化测试。
