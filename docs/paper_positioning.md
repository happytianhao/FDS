# FDS paper positioning

这份文档记录论文的工作定位、核心 idea 和 contribution 草稿。它是写作工作笔记，不替代最终论文中的正式定义、实验结果和引用。

对应的分层实验方案见 [`docs/experiment_design.md`](experiment_design.md)。

## 一句话定位

FDS（Falsifiable Decision State）把具身智能中的每次决策显式表示为一个可由 host 检查、带有证据来源、预期结果和反证条件的决策契约，使系统能够区分“动作已经执行”与“任务语义上已经成功”，并为纠错、审计和后续改进提供共同接口。

## 核心 idea

具身任务中的模型输出不是普通的文本回答，而是在部分可观测环境中基于有限证据选择下一步行动。FDS 因此要求每次决策同时表达：

- 选择了哪个候选行动，以及它依赖的假设；
- 每个假设对应哪些当前可用的观测或保留帧；
- 预期会发生什么；
- 哪些新证据或执行反馈会证伪这个判断；
- 当前风险和承诺状态是什么。

这些内容组成 `fds_contract`。host 在执行前检查契约结构、证据来源和承诺条件；执行后单独记录公开执行反馈和 FDS verdict。执行收据只说明命令是否被接受或发生了公开错误，不自动被解释为任务语义成功。

FDS 的关键边界是：环境工具负责改变或推进环境，辅助工具只读取已有观测或计算证据；辅助工具不会因为产生了几何结果或图像检查结果而获得环境执行权限。这样，决策、证据、行动和反馈可以在同一条审计链中连接，但各自保持清晰的因果边界。

## 五个正交因素

论文中应明确把下列因素分开。FDS 是其中的决策状态因子，而不是其他因素的总称。

| 因素 | 解决的问题 | 论文中的角色 |
| --- | --- | --- |
| **FDS factor** | 模型是否显式声明假设、证据、预期结果和 falsifier | 决策表示与 host-side verification 协议 |
| **Harness** | 如何组织 actor、工具、环境、预算、隔离和反馈 | 执行基础设施 |
| **Memory** | 哪些历史观测、事件和 keyframe 可以被后续决策引用 | 信息保留与证据边界 |
| **Skill** | 可复用的任务指导或策略如何注入 actor | 行为先验/策略指导 |
| **RSI** | 如何根据训练失败提出、检查和评估 harness 或 skill 的改动 | 训练期改进循环 |

因此，论文可以分别研究：在同一个 harness、memory、skill 和 RSI 设置下打开或关闭 FDS；也可以研究 memory、skill 或 RSI 的变化，而不把这些变化误称为 FDS 的必要组成部分。当前代码中 `contract_mode = "off"` 提供了 matched contract-off baseline；skill 接口和 training-only RSI 已保留独立边界。

## Contribution 草稿

### 1. 决策表示范式

我们提出 Falsifiable Decision State（FDS），一种面向具身任务的显式决策表示范式。FDS 将每次行动选择表示为带有候选行动、可追溯假设、证据引用、预期结果、风险和 falsifier 的结构化契约，从而把通常隐含在模型内部的“我为什么现在采取这个动作”转化为可检查、可审计、可被后续反馈推翻的状态。

可用的英文表述：

> We introduce Falsifiable Decision State (FDS), an explicit decision representation for embodied agents that binds each selected action to traceable assumptions, evidence references, expected outcomes, risks, and falsifiers.

### 2. 契约驱动的执行与纠错系统

我们实现了一套与决策表示配套的 host-validated execution protocol。系统在行动前检查契约和证据 provenance，在行动后区分机械执行反馈与语义任务结果，并把环境执行工具、只读辅助工具、episodic memory、keyframe retention 和 audit/verdict 组织成可恢复的闭环。该设计允许模型先补充证据、检查几何或重新查看帧，再决定是否继续行动；它也使公开错误、未验证结果和真正的语义验证保持不同状态。

可用的英文表述：

> We build a contract-driven execution and correction system that validates decision provenance before acting, separates environment actions from read-only auxiliary analysis, and records execution feedback without conflating command completion with semantic success.

这里要强调 FDS 与 harness 的关系：harness 提供执行和审计容器，FDS 提供其中可独立开关的决策状态协议。Memory、skill 和 RSI 可以接入同一容器，但不构成 FDS 的定义。

### 3. 因子化评测与具身实验

我们在 RoboMME 和 RoboMME-Plus 上建立可复现的具身评测，使用相同模型、观测、工具、预算和 benchmark 条件比较 FDS 与 contract-off baseline，并进一步把 memory、skill 和 training-only RSI 作为独立因素进行消融或组合实验。除最终任务成功率外，评测应报告：

- 语义任务成功与机械执行成功的区别；
- 契约有效率、证据 provenance 错误和 host 拒绝率；
- 纠错/恢复次数、辅助调用成本和环境步数；
- 未验证或被反驳决策的比例；
- RSI 在训练 split 上产生的改动能否在冻结的 validation/test 条件上保持收益而不引入回归。

可用的英文表述：

> We provide a reproducible, factorized evaluation on RoboMME and RoboMME-Plus, comparing FDS against a matched contract-off baseline and isolating the effects of decision contracts, memory, skills, and training-only recursive self-improvement. The evaluation measures both task outcomes and the quality, provenance, cost, and recoverability of the decision process.

## 写作时应避免的表述

- 不要说“FDS 保证任务成功”；FDS 约束的是决策声明、证据边界和可证伪性。
- 不要把一次 command receipt 直接当作 semantic success；代码中的 verdict 明确保留 `uncertain`。
- 不要说 auxiliary tools 执行环境动作；它们是 read-only review/calculation tools。
- 不要把 memory、skill、harness 或 RSI 写成 FDS 的组成定义；它们是可以与 FDS 正交组合的系统因素。
- 不要只报告最终成功率；FDS 的价值还体现在证据 provenance、错误暴露、纠错路径和审计能力上。

## 论文叙事顺序

建议按以下顺序组织正文：

1. 先说明具身决策的核心困难是“基于不完整证据采取行动”，而不是单纯生成动作；
2. 定义 FDS contract 及其 host-side invariants；
3. 说明 contract 如何嵌入 task harness，并与 memory、skill、RSI 解耦；
4. 展示环境工具与辅助工具的因果边界、执行反馈和纠错闭环；
5. 给出 contract-off 和其他正交因素的因子化实验；
6. 最后讨论决策过程质量、可审计性和语义成功验证，而不只讨论动作执行率。
