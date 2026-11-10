# FDS 执行后核对实现

日期：2026-11-10。状态：当前代码候选，待训练验证。没有增加任何 episode 尝试，不将离线测试称为性能提升，也没有替换 E4 正在引用的不可变 v029 快照。

## 基线与范围

本次先将 v029 的通用修复合入正式源码：同帧同相机输入去重、精确的可用引用范围、引用/容量拒绝提案反馈、上次物理契约跨辅助调用传递、机械 verdict 的范围标签。随后增加下述后验核对协议，并修正 Codex 的指令层级与技能说明隔离。

已有重试/网络可靠性工作保留，不属于本次重新设计。模型、effort、工具集合、图像数量上限、每组两次决策纠正、总调用预算、训练集范围和每案例总尝试上限保持原约定。官方 success 只用于评分，没有注入 planner。

## 新的执行流程

1. 模型输出原有 contract、tool_batch 和 memory_changes，额外输出 `action_binding` 与 `outcome_review`。
2. `action_binding` 指向 selected candidate、按顺序列出实际工具索引、绑定需要确认的执行事件，并列出后续动作依赖的事件 ID / revision。
3. 宿主从契约及选中候选提取 expected outcome / falsifier，去除完全重复项，分配包含动作调用号的稳定 check ID。记录工具参数的规范化 SHA256、实际回执及原动作起止帧。
4. 下一次正常 planner 调用必须逐项核对，返回 met / violated / unknown、证据来源与解释。对于 falsifier，met 表示反证出现，不能当作成功。
5. 先在临时内存副本中应用提案，校验后验核对、事件修正与依赖，再原子接受。失败进入现有有限纠正；拒绝前不执行工具，不修改有效记忆或核对状态。
6. 新动作执行后记录下一组待核对项。辅助工具不会覆盖它们；有未决事件的旧动作可跨等待、退离等后续动作继续核对。

入口为 [postcheck.py](../src/fds/task_harness/postcheck.py)，运行整合为 [runtime.py](../src/fds/task_harness/runtime.py)，动态输出结构为 [protocol.py](../src/fds/task_harness/protocol.py)。

## 证据与事件语义

- 视觉评估必须引用当前供应或保留的、晚于原动作开始的 episode 帧。原动作起止帧不随延迟观察被篡改；后续已供应帧可用来判断先前结果，模型必须区分延迟反馈与后来产生的新事件。
- public_feedback 需要引用该 check 所属动作的回执；另一次动作的回执不能替代。执行回执本身不允许确认 expected outcome。零步拒绝可据公开回执报告未完成，不需要伪造新图像。
- 已接受的 check 可以逐字段原样沿用 last_assessment，作为历史文本判断，不是新视觉核验；改变判断仍需当前可用证据。沿用不会恢复旧图片或赋予新记忆条目的额外引用权限。
- 事件必须显式关联动作。新增或改变执行事件的 succeeded / failed 状态，需要对应后验核对；不能先标成功再执行动作。
- 每个绑定事件引用该动作的全部 predicates。成功要求所有 expectation 为 met、所有 falsifier 为 violated；失败需至少一项否定期望或命中反证；证据不足保持 unknown。第一版有意使用保守规则，尚未提供更复杂的事件子条件表达式。
- 事件状态更新必须与接受的 memory upsert 一致，保留原 objective 和 success_criteria。不能改写目标来让失败变成功。
- 未决事件可保留自己的原始 predicates 跨下一次物理观察；不能将它重新绑定成另一个动作。删除需要显式 unknown resolution + delete，不把删除当作成功/失败。已确认或已删除事件不在辅助调用中被反复要求 upsert。
- 事件依赖必须引用当前 revision，且在本次修正后仍是 succeeded 的 execution 事件；tentative 标签不能豁免这个显式依赖检查。memory 被修订后旧 revision 不再有效。
- 未知结果不会禁止所有动作。模型可以选择不依赖该事件已发生的辨别性动作，也可以选择 finish。finish 不代表官方成功。

未决来源数量受现有事件内存上限约束。模型上下文只保留仍有未决依赖的旧来源及最近动作；完整历史留在 audit，不能通过它恢复图片。每次仍需核对待处理项，可能增加输出 token，必须在后续实验中测量。

## 能声称与不能声称

已实现：逐项模型核对、来源/时序检查、动作参数绑定审计、执行事件状态修正、显式事件依赖版本检查、拒绝的状态隔离与有限纠正。

仍不能声称：独立视觉证明、自动发现所有漏报依赖、识别任意自然语言计数、准确判定不可逆操作、完整对象身份/全部假设的依赖图。候选描述与参数在语义上是否相符仍由模型判断；工具索引与哈希只确保审计关联到实际动作，不证明某个坐标就是正确对象。

模型也可能提出过弱的预期条件、误读图片、遗漏 event_ids/dependencies。因此这是可验证的结构改进，不是完备的语义安全保证。隐藏计数、碰撞/接触阈值或任务答案没有进入在线判断。

历史 [机制边界核查](fds_mechanism_boundary_review.md) 描述的是修改前的 v028；本次补上其中一部分，不能把整份缺口都说成已解决。

## 配置与审计

`full` 默认启用新协议，解析后的配置显式记录开关。需要同一代码下的旧协议对照时设置：

```toml
[fds]
enabled = true
contract_mode = "full"
postcheck_enabled = false
```

`contract_mode="off"` 自动关闭 postcheck，响应不包含新字段。旧不可变实验快照按原行为执行，不会因为当前工作区改变而被替换。

每个接受的调用记录 `audit/call_XXXX/outcome_review.json`；物理请求另有 `action_outcomes.json`，包括来源、参数哈希、回执、原始帧区间和 check IDs。`summary.json` 记录 `postcheck_enabled` 与 `outcome_tracking`。这些计数是接受的核对条目数，包括 unknown 的多次重审；不能把它们当作官方成功数或独立视觉发现数。

环境结束或预算耗尽时不额外增加 verifier 调用，summary 保留最后动作尚未核对的状态。成本按正常请求的已知 token 累计。

Codex base override、skills 隔离及其他仓库的注入渠道见 [prompt_injection.md](robomme_interface_review/prompt_injection.md)。新旧提示协议不同，不能将旧结果改标为新版成绩。

## 验证与下一步

离线测试覆盖正常确认、未知状态、跨辅助/物理观察的延迟反馈、过期图片的原样历史判断、删除/确认后的清理、零步拒绝、不可用/旧帧、回执错误、遗漏与矛盾 predicates、过期/未知依赖、记忆提案被拒后的原子性，以及 finish/预算行为。提示测试检查真实命令构造和 system/user 内容分离。

使用项目 `.venv` 执行 `python -m pytest tests`；没有为验证代码而调用真实模型、运行机器人训练或测试集。下一次训练评估应冻结本工作区代码，单独报告提示隔离和 postcheck 的影响，继续遵守既有尝试次数上限；本次没有自动启动或扩展实验。

本次全量离线测试 **139 项通过**，其中新增后验核对测试 26 项、提示注入测试 3 项。另用本地拒绝 provider 验证 Codex 真正的请求序列化，三组对照均按预期完成。相关新增/修改代码的静态检查与 `git diff --check` 通过。
