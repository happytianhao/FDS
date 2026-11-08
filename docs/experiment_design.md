# FDS experiment design

这份设计围绕论文的核心主张：FDS 是一个可以独立插入具身 task harness 的决策状态因子。实验首先测量 FDS 本身的因果效果，再研究它和 harness、memory、skill、RSI 的正交组合。所有结果都应区分任务语义成功、机械执行反馈和决策过程质量。

## 1. 核心因果问题

主问题不是“加入更多提示词后分数是否提高”，而是：

> 在相同模型、观测、工具、预算和 benchmark 条件下，显式且可由 host 校验的 FDS contract 是否能减少早期不可逆错误、提高错误后的恢复能力，并让失败原因可追踪？

对应的假设是：

- **H1：可靠性。** FDS 降低 first irreversible error 和 false commitment，并提高最终任务成功率。
- **H2：可恢复性。** FDS 的 falsifier、公开执行反馈和 post-action verdict 让系统更快发现矛盾，并提高 contradiction 后的 recovery rate。
- **H3：过程质量。** FDS 改善 evidence provenance、assumption support 和 contradiction detection，而收益不能只用“模型输出了更多文字”解释。
- **H4：正交性。** FDS 的收益在固定 memory、skill 和 RSI 设置下仍然存在；memory、skill、RSI 的变化可以单独测量，不应被写成 FDS 的定义。
- **H5：泛化。** FDS 在 RoboMME-Plus 上的效果可以迁移到原始 RoboMME，并且不依赖某一个任务或某一类工具。

## 2. 实验因素和不变量

用 `F, H, M, S, R` 表示五个因素：

| 符号 | 因素 | 实验定义 |
| --- | --- | --- |
| `F` | FDS factor | `F0`: contract off；`F1`: 完整 FDS contract、host validation 和 verdict |
| `H` | Harness | actor、工具编排、环境隔离、预算和反馈管线；主实验中固定为同一个 Task Harness |
| `M` | Memory | 观测历史、文字 memory 和 retained keyframe 的可引用范围 |
| `S` | Skill | 注入 actor 的可复用指导；主实验使用空 skill，扩展实验使用冻结的 train-derived skill set |
| `R` | RSI | `R0`: 固定 executor；`R1`: 仅用 train traces 驱动的 training-only Meta Harness |

主实验固定 `H=1, M=1, S=0, R=0`，只改变 `F`。这样主表回答的是 FDS 是否有效，而不是几个系统改动叠加后是否有效。`Astra-Sys2` 或原始 Harness 的公开结果可以作为外部参考，不能替代这个 matched comparison。

所有成对方法必须固定：

- 同一个 `gpt-6-astra`、effort、prompt 预算和模型调用超时；
- 同一批 RGB、robot state、环境工具和 auxiliary tools；
- 同一 `max_calls`、environment step budget、observation budget 和 keyframe 上限；
- 同一 task、episode、benchmark revision 和随机种子；
- 同一 process isolation、校准和 simulator 配置。

## 3. 实验分层

### E0：协议和公平性检查

在正式跑 benchmark 之前，先用 mock/离线 contract 测试确认：

1. `F0` 的模型输入不包含 FDS contract 字段或 FDS 指令；
2. `F1` 的 contract 会经过 schema、evidence provenance、commitment gate 和 verdict 流程；
3. 两个条件的 RGB、robot state、工具集合、memory、预算和 evaluator 完全一致；
4. auxiliary batch 不会推进环境，environment batch 才会产生环境步；
5. command receipt 不会被写成 semantic success；
6. audit 中能关联 observation、contract、tool receipt、verdict 和后续决策。

这一层是实验有效性的前置条件，不计入任务成功率。

### E1：FDS 主对比

主对比只保留两个条件：

| 条件 | `F` | `H` | `M` | `S` | `R` | 作用 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Task Harness | 0 | 1 | 1 | 0 | 0 | matched contract-off baseline |
| Task Harness + FDS | 1 | 1 | 1 | 0 | 0 | FDS 的主处理条件 |

先在 RoboMME-Plus 上做主结果，再用完全冻结的配置在 RoboMME 上做泛化验证。开发只使用 `train`；最终报告至少包含冻结后的 `validation` 或 `test`，不能让 Meta Harness 或人工调参读取最终 test 轨迹。

主表建议报告：

| Method | Success | First irreversible error | Recovery | Planner calls | Prompt/completion tokens | Environment steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task Harness (`F0`) |  |  |  |  |  |  |
| Task Harness + FDS (`F1`) |  |  |  |  |  |  |

主结论应优先使用 paired episode difference，而不是只比较两个 aggregate success rate。

### E2：FDS 机制消融

这一组回答“收益来自显式决策状态，还是来自格式和额外文字”。最小可行的机制路径是：

| 条件 | contract | host gate | post-action verdict | 解释 |
| --- | --- | --- | --- | --- |
| `F0` Contract Off | 无 | 无 | 无 | 基线 |
| `F-log` Log Only | 有 | 只记录，不拒绝 | 不反馈或只记录 | 测量格式化日志本身的收益 |
| `F-validate` Validated FDS | 有 | schema + provenance + commitment gate | 有 | 测量 host 约束的收益 |
| `F1` Full FDS | 有 | 完整校验 | 有，作为下一轮决策输入 | 测量可证伪反馈闭环 |

`F-log` 和 `F-validate` 需要在当前 `off/full` 之外增加配置模式；如果时间有限，论文主结果可以先只做 `F0` vs `F1`，并把 `F-log` 作为后续实现或附录实验。

在完整 FDS 内，再做少量有明确机制含义的字段消融：

- **w/o evidence references：** 假设仍然存在，但不能引用具体 delivered/retained frame；
- **w/o falsifiers：** 保留预期结果，但不要求列出会推翻决策的证据；
- **w/o commitment gate：** 未解决或 unsupported assumption 仍允许 committed；
- **w/o post-action verdict：** 执行后的 contradiction 不进入下一轮 context；
- **w/o candidate comparison：** 只保留单一候选行动，不列 alternatives。

每次只去掉一个机制，其他 contract 字段和预算保持不变。论文正文重点放 `F0`、`F-log`、`F-validate`、`F1` 四档，字段级消融放机制表或附录。

### E3：FDS 与 memory、skill 的正交性

不建议直接跑完整 `2^5` 全因子设计，成本高且难解释。用两个 2×2 交互实验即可证明主要正交关系。

#### E3a：FDS × Memory

| 条件 | FDS | Memory |
| --- | ---: | ---: |
| A | 0 | 0 |
| B | 1 | 0 |
| C | 0 | 1 |
| D | 1 | 1 |

报告 FDS 的边际收益 `B-A`、`D-C`，以及 interaction `((D-C) - (B-A))`。Memory-off 必须是真正的 stateless/无 retained keyframe 条件，不能只把 `max_keyframes` 改成一个名字上为零但仍保留文字 memory 的伪消融。

#### E3b：FDS × Skill

| 条件 | FDS | Skill |
| --- | ---: | ---: |
| A | 0 | 0 |
| B | 1 | 0 |
| C | 0 | 1 |
| D | 1 | 1 |

Skill set 必须在 train 上冻结，validation/test 期间不可新增或修改。Skill 只能提供指导，不能覆盖 FDS contract、预算、身份和 provenance 规则。报告 `B-A`、`D-C` 以及 skill 是否放大或削弱 FDS 的 interaction。

这两组实验的意义不是证明所有因素完全独立，而是证明 FDS 的主要效果不依赖某一个 memory 或 skill 配置，并且可以和它们组合。

### E4：风险自适应和成本

风险自适应是 FDS 的执行策略扩展，不是 FDS 定义本身。应放在主结果之后，回答“可靠性收益是否必须付出同等的调用开销”。

| 条件 | 行为 |
| --- | --- |
| Full Verify | 每次决策都使用完整 contract 和 gate |
| Risk-Adaptive FDS | contract 仍存在；低风险动作使用较轻的检查，高风险动作使用完整 evidence/commitment gate |
| No Fast Path | 所有动作都走完整 FDS，作为效率对照 |

建议的风险规则：

- `wait` 或已由当前证据支持的自由空间移动属于低风险；
- 抓取、释放、接触、按钮操作、遮挡后行动属于高风险；
- 物体身份、颜色语义、平面假设或工具用途未解决时禁止 committed；
- 发生 contradiction 后，下一次决策自动回到完整验证路径。

低风险快速路径不能删除 FDS 的审计字段，否则它会变成另一个 contract-off 条件。报告 success、first irreversible error、calls、prompt/completion tokens、environment steps、auxiliary calls 和 cost per successful episode。

### E5：training-only RSI / Meta Harness

RSI 是跨 episode 的改进过程，应该单独评估，不能和 E1 的单 episode FDS 因果效果混在同一主表里。

| 条件 | 说明 |
| --- | --- |
| `R0` Static | 固定初始 Task Harness、memory 和 skill |
| `R1-task` | Meta Harness 只能修改 task harness |
| `R1-skill` | Meta Harness 只能新增或修改 skill |
| `R1-full` | 按项目规则允许修改 task harness 和 skill |

每一轮 RSI 使用相同的 train cases 和相同的 parent/candidate paired evaluator：

```text
Round 0: 初始 FDS executor
Round 1..K: 只读 train failure traces → proposal → isolated check → paired train evaluation
Freeze: 固定 promoted executor、skills、source hash 和 config hash
Final: 在未参与 proposal 的 validation/test 上一次性评估
```

必须报告：

- train success gain；
- held-out validation/test gain；
- regression 数量；
- 修改文件数、patch 大小和新增/修改 skill 数量；
- 每个 proposal 引用的 train evidence；
- 每轮的 planner calls、steps 和 token 成本。

RSI 的成功标准是 train 有改善、held-out 不退化，并且提升能被结构化 failure trace 解释。只在 train 上变好不能作为最终结论。

## 4. 指标定义

### 任务结果

1. **Official task success rate：** 以 benchmark/host 的官方状态为准，不使用 actor 自报成功替代。
2. **Per-task success rate：** 对每个 task 单独报告，再做 macro average，避免 episode 多的 task 支配结果。
3. **Timeout/error rate：** 区分 budget exhausted、环境错误、planner 错误和进程错误。

### 决策可靠性

1. **First irreversible error rate：** 第一次不可逆错误发生在动作序列中的比例。若 benchmark 没有自动标注，需要由独立 trace annotator 根据公开反馈和帧进行标注。
2. **False commitment rate：** committed 决策中存在 unsupported、tentative 或 unresolved assumption 的比例。
3. **Assumption support rate：** committed assumption 被当前 delivered/retained evidence 支持的比例。
4. **Falsifier coverage：** 后续真实失败是否已经出现在该决策声明的 falsifier 中。
5. **Contradiction detection rate：** 公开 action error、trial reset 或 rejected receipt 被正确标记为 `contradicted` 的比例。
6. **Recovery rate：** 发生 contradiction 或 first failure 后最终仍成功完成任务的比例。
7. **Recovery cost：** 从 contradiction 到恢复或终止额外消耗的 calls、tokens 和 environment steps。

### 过程和资源

- contract schema validity rate；
- evidence provenance error 和 host rejection rate；
- auxiliary batch 次数与 environment batch 次数；
- planner calls、prompt tokens、completion tokens；
- environment steps、每个成功 episode 的平均成本；
- 未验证 `uncertain`、被反驳 `contradicted` 和已验证结果的比例。

当前代码已经记录 planner calls、prompt/completion tokens、steps、contract 和 verdict；first irreversible error、falsifier coverage、人工 evidence support 需要从 audit trace 增加离线标注或分析脚本。

## 5. 统计和报告协议

- 所有主比较使用相同 episode ID 的 paired evaluation；
- 以 task macro average 和 episode micro average 同时报告；
- 使用 paired bootstrap 95% confidence interval，必要时报告 episode-level permutation test；
- 如果模型或环境存在随机性，至少使用多个 seed，且每个方法使用相同 seed 集合；
- 报告完整失败分布，不能只挑成功案例展示；
- qualitative trace 选择规则必须在看结果前固定，展示成功、早期错误、contradiction recovery 和误承诺各类案例。

## 6. 论文主表和图

建议正文使用四个结果单元：

1. **Main table：** `Task Harness (F0)` vs `Task Harness + FDS (F1)`，覆盖 RoboMME-Plus 和 RoboMME。
2. **Mechanism table：** `F0`、`F-log`、`F-validate`、`F1` 及字段级消融。
3. **Process figure：** first irreversible error、contradiction detection、recovery rate 和 recovery cost 的 paired 分布。
4. **Orthogonality/RSI figure：** FDS×Memory、FDS×Skill 的 2×2 interaction，以及 RSI 各轮的 train/held-out 曲线。

主表列建议为：

| Method | Benchmark | Success | First irreversible error | Recovery | Contract validity | Calls | Tokens | Steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task Harness (`F0`) | RoboMME-Plus |  |  |  | — |  |  |  |
| Task Harness + FDS (`F1`) | RoboMME-Plus |  |  |  |  |  |  |  |
| Task Harness (`F0`) | RoboMME |  |  |  | — |  |  |  |
| Task Harness + FDS (`F1`) | RoboMME |  |  |  |  |  |  |  |

## 7. 当前实现与实验前需要补的接口

当前仓库已经支持：

- `contract_mode = "off"` 与 `contract_mode = "full"`；
- contract schema、evidence provenance、commitment checks 和 post-action verdict；
- auxiliary/environment tool separation；
- bounded episodic memory、retained keyframes 和 frozen per-episode skills；
- training-only Meta Harness、development evidence 和 paired promotion gate。

为了完整跑完上述设计，建议按以下优先级补接口：

1. 增加 `log_only` 或 `soft` contract mode，用于区分记录收益与 enforcement 收益；
2. 增加字段级 ablation 开关，或在离线 replay 中生成受控缺字段条件；
3. 增加显式 `memory_mode = "stateless" | "episodic"`，避免用 keyframe 上限冒充 memory-off；
4. 增加 `risk_policy = "full" | "risk_adaptive"`，但保持所有路径产生可审计记录；
5. 增加 metrics/export 脚本，从 audit 统一汇总 contract、verdict、receipt、calls、tokens 和 steps；
6. 增加人工 trace annotation schema，用于 first irreversible error、evidence support 和 falsifier coverage。

如果暂时不扩展接口，论文第一版仍可以先完成 E0、E1、E3b 的空 skill 条件和 E5 的 R0/R1；不要在没有真实开关的情况下声称完成 `log-only`、memory-off 或 risk-adaptive 消融。

## 8. 预期论文结论

论文最终应围绕以下四句话组织，而不是只强调一个成功率数字：

1. 在 matched embodied execution 条件下，FDS 减少了基于未解决假设的早期不可逆决策；
2. Falsifier、host-side evidence gate 和 post-action verdict 是恢复收益的主要机制；
3. FDS 的收益来自改变决策、证据和反馈之间的关系，而不是单纯增加输出长度或 token；
4. Memory、Skill 和 training-only RSI 可以与 FDS 组合，且 RSI 能利用 FDS 产生的结构化 failure trace 在 held-out 条件上改进 executor。
