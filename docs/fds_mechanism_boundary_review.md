# FDS 机制边界核查与建议

核查日期：2026-11-09。核对正式src及实验v028；以下为代码分析与离线探针，不是新机制的实现或性能结论。本轮未修改运行中的实验源码或评分。

## 结论

如果目标是声称“证据→假设→行动依赖→反馈后的判断修正”构成可执行闭环，需要补代码。现有结构可以记录关系、检查引用合法性，并按模型自报承诺状态做一致性约束；尚未完整实现语义支持核验、候选与具体行动的强绑定、逐项事后核对及依赖失效传播。

## 已确认事实

1. `task_harness/fds.py:24` 的supported是模型输出。`validate_contract`检查引用属于可用集合，不读取图像语义。supported且evidence_refs为空也能通过；空引用并不必然错误（某些判断可来自工具回执），但协议目前缺少区分这些证据类型的结构。
2. `fds.py:112`仅在commitment=committed时检查未决依赖。tentative+unknown假设+闭合夹爪能通过协议解析。tentative原本可以允许有界试探，不能简单禁止所有tentative动作；但实际不可逆操作不能只靠模型把承诺标签改成tentative来豁免约束。
3. CANDIDATE仅有description/requires/expected_outcome/falsifiers/risk；没有具体工具参数绑定。`protocol.py`分别校验tool_batch和contract；`tools.py:114`直接执行tool_batch。候选描述“只回看”而实际工具是wait，仍通过解析。
4. `execution_verdict(contract, receipts)`未使用contract中的预期结果及falsifiers。只替换这些文本而保持回执不变，verdict不变。v027新增scope=mechanical_execution、semantic_status=unassessed，修正了解释范围，没有增加视觉判定。
5. v027保留last_environment_fds以跨辅助调用传递原动作契约；v028保留未应用的被拒记忆提案。它们改善信息传递，但不等同于逐项假设修正或语义验证。
6. CLI仍只支持off/full。docs/experiment_design.md所写F-log/F-validate是设计，不能当成已实施的受控比较。

离线探针只调用schema/parse/verdict，没有调用模型或环境：四项结果分别为supported空引用接受、tentative未知前提下闭合夹爪接受、候选/实际工具不一致接受、改变预测与falsifier后verdict不变。

实际例子：VideoRepick_ep0 attempt04，audit/call_0024/decision.json中identity=tentative、historical_identity=unresolved、commitment=tentative；随后move_ee close+hold，官方报告Picked up the wrong object. 这是未决身份下执行动作的真实轨迹，不证明所有tentative操作都应禁止。

## 最小实现顺序（提案，未实施）

1. **绑定具体动作**：提供一份权威action plan，每条工具调用有ID；selected candidate引用明确tool IDs及其所依赖的假设版本。宿主在校验和执行之间保存规范化参数及哈希，记录requested/resolved/executed差异。参数一致性可机械验证，但这仍不独立证明某位置对应画面中的正确对象。
2. **明确执行约束策略**：把结构/来源合法与行动准入分开，策略显式区分取证探测与承诺行动。使用工具效果、实际参数及声明依赖检查，不能仅信任模型的low-risk/tentative标签。部分物理风险无法由现有公开信息可靠判定，应保留unknown，不能冒充宿主证明。
3. **逐项后验声明核对**：expected_outcome/falsifier具有稳定ID；下一次正常planner调用先针对上次动作输出met/violated/unknown及新证据、关联receipt和需修正的assumption/memory IDs，再规划下一步。宿主校验来源和时序。可计算的条件用确定性检查，视觉关系保留model_assessed标签；没有新证据时unknown，不以执行回执充当成功。默认复用已有调用预算，不新增免费评审调用。
4. **依赖失效传播**：假设有稳定ID和revision，候选引用具体revision；假设被反驳后，使其依赖的未执行计划失效，要求重算计数/身份结论。宿主维护关系、失效标记和原始历史；模型依据证据修正语义，不能由宿主凭空重写物体事实。
5. **可独立开关的机制层**：off、log_only、precondition_enforced、postcheck_enabled。log_only中契约错误只记录，仍使用各组共享的工具安全/格式校验；否则它就不是仅记录条件。另设同预算的普通条件检查/验证重规划控制，避免把所有验证收益归于FDS表示。

增加执行准入和语义核对会改变算法行为，按已约定的训练协议应作为独立候选明确确认后实现和评估，不能暗中改变当前补跑。

## 评测与论文措辞

- 固定同一harness代码、模型/effort、memory、空skills、工具、episode/seed和预算上限；比较阶段冻结版本，不能边跑边优化不同组。
- 仅记录契约也会改变模型输出与计算量，不是完全被动的日志；应报告真实prompt/completion token及额外评审成本，并提供成本匹配对照。
- “结构性未决前提却承诺”与“模型标为supported但独立标注认为无支持”必须分开。后者需要基于公开证据的盲审/标注及unknown类别，不能用模型自己的supported字段自证。
- 机械错误检测和视觉/事件语义矛盾检测分别计数；恢复定义为检测后的有证据修正/后续任务成功，不能只计JSON纠正成功。
- 独立模型复核仍是模型判断，不能称视觉证明；官方状态仅用于外部评分，不注入planner，也不剥夺模型finish权。
- 当前可以声称结构化契约、引用合法性、声明一致性检查、机械回执及有限纠错、跨调用证据审计。暂不能声称视觉假设已被独立证明、falsifier逐项执行、动作选择受完备语义约束或性能收益已由上述机制消融确定。
