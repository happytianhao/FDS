# Prompt 注入渠道与 Codex 隔离核查

日期：2026-11-10。用户要求子 agent 独立核查；依据为本地固定提交源码和本机 Codex 0.163.0。没有调用真实模型或新增机器人实验，没有加载、删除或修改任何 OpenAI skill。

## 六个仓库的实际做法

| 方法 | 规则在哪里注入 | 动态信息与其他条件 |
|---|---|---|
| Embodied-Astra | Responses `instructions=system_prompt(task)` | goal、图片、TCP、标定在 user；后续 `previous_response_id` 延续历史 |
| 3-tier Astra | 高层 planner 将每任务 Markdown、goal、completed/issued 都放在 user 内容；并非显式 system | 独立本地 Qwen3-VL + adapter 完成 monitor 使用 system/user；动作由训练后的 VLA 执行 |
| MEMI Act | 固定角色、机器人和周期语义在 Responses instructions | 优化后的 memory/action 规则在 user cached prefix；本轮图像、状态、memory 和关键帧在 user |
| PonderPounce | 本地 VLM user token 序列与图像 | 持续 KV 上下文，追加 subgoal/cognition；动作网络还接收 cognition、age_ms、proprio 张量 |
| SimpleMemVLA | 本地 user 多模态消息：视频、机器人参数、频率、时域、task | chat template/processor 编码，动作能力也依赖 checkpoint 与 action head |
| FrameSamp+Modul | goal 作为 VLA prompt | 历史图像经编码和采样，以 memory features 调制训练过的动作网络 |

具体源码位置（链接固定到审查时的提交）：

- [Embodied-Astra 调用](https://github.com/jackbjed/embodied-astra-robomme/blob/79dcc7f73f536948cb039103d3fc5e722d689207/harness/astra_waypoint_eval_v3.py)：规则定义约 66 行，instructions 225 行，首轮 user 265 行，历史关联 304 行。
- [3-tier planner](https://github.com/bingaochen/Astra-on-RoboMME/blob/4c3fd6a8667e7a219e547fe1a533a4b9726fc6db/examples/champ/core.py)：约 96 行组装规则；[API payload](https://github.com/bingaochen/Astra-on-RoboMME/blob/4c3fd6a8667e7a219e547fe1a533a4b9726fc6db/examples/champ/api_client.py) 约 79 行；[monitor 输入](https://github.com/bingaochen/Astra-on-RoboMME/blob/4c3fd6a8667e7a219e547fe1a533a4b9726fc6db/examples/champ/input_contract.py) 约 20 行。
- [MEMI Act 前缀及本轮内容](https://github.com/memi-robot/memi/blob/bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0/memi/act/policy_turn.py) 约 23 行；[Responses payload](https://github.com/memi-robot/memi/blob/bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0/memi/llm/responses.py) 约 137 行。
- [PonderPounce 上下文](https://github.com/worv-ai/PonderPounce/blob/723df35762bb641e1d520e4fa9359b98644adc21/ponderpounce/inference/append_context.py) 约 109 行；[动作网络条件](https://github.com/worv-ai/PonderPounce/blob/723df35762bb641e1d520e4fa9359b98644adc21/ponderpounce/inference/local.py) 约 168 行。
- [SimpleMemVLA 消息](https://github.com/OpenBMB/SimpleMemVLA/blob/453d6ec293e63318bbc2bce545d0b9256ab1bc51/simplememvla/data/messages.py) 约 73 行；[编码与推理](https://github.com/OpenBMB/SimpleMemVLA/blob/453d6ec293e63318bbc2bce545d0b9256ab1bc51/robomme_sim/policy.py) 约 262 行。
- [FrameSamp 输入](https://github.com/RoboMME/robomme_policy_learning/blob/ac846a9a4aec841e483b1dc5d620dfe426bde853/challenge_interface/policy.py) 约 69 行；[历史条件化](https://github.com/RoboMME/robomme_policy_learning/blob/ac846a9a4aec841e483b1dc5d620dfe426bde853/src/mme_vla_suite/models/integration/history_pi0.py) 约 616 行。

这不支持“只要改成 system prompt 就更强”的结论；上下文内容、闭环职责、代码验证和训练权重同样重要。3-tier 的任务规则和 MEMI 离线优化规则是额外任务知识；MEMI `used_fields` 是模型自报使用，不能作为因果证明。MEMI cached prefix 不等于自动携带全部在线对话。

## 原来的本机接口实验

[run_one.py](../../scripts/experiments/interface_ablation/run_one.py) 原来从上游提取 SYSTEM/MODEL_NOTES/LR_NOTE，拼入 Codex stdin，再显式拼接历史图片说明和公开回答。文件名虽为 `system_prompt.txt`，实际属于用户输入，没有替换原生 Codex coding base。16 类 task contract 草案也没有被它读取。

本次代码修正后的新请求使用真正的 base override。历史实验记录仍属于旧注入方式，不能事后改称新实现的结果。新 manifest 标记 `prompt_protocol=codex_base_instructions_v1`、`skills_include_instructions=false`；恢复运行要求这两个字段一致，拒绝把旧中断试验接到新注入协议上。

## Codex app / CLI 的真实覆盖接口

在临时 CODEX_HOME、无 auth.json、无真实 key 的环境中，子 agent 使用只监听 127.0.0.1 的假 provider 捕获请求序列化；服务返回本地 400 后停止，不转发请求。结果：

| 入口 | 设置 | 本机行为 |
|---|---|---|
| CLI | `-c model_instructions_file="/absolute/system_prompt.txt"` | 替换原生 coding base |
| CLI | `-c base_instructions="..."` | 本次版本未生效，不能代替上述文件开关 |
| CLI | `-c developer_instructions="..."` | 追加内容，保留原生 coding base |
| App-server thread/start | `baseInstructions` | 替换原生 coding base |
| App-server thread/start | `developerInstructions` | 追加 developer 内容 |

该 Astra 模型模板把基础指令序列化为 input 中的 developer 消息，并非字面 `role=system` 或顶层 instructions。真正替换 base 即满足本次目标，不需要迁移到 app-server。

`codex debug prompt-input` 不展示完整 base，不能仅凭该输出找不到自定义文本判断替换失败。它可用于检查附加技能清单；基础指令覆盖用本地假 provider 补验。测试约 21,420 字符的默认 coding base 被约 92 字符测试指令替换；这只是该本机版本的观测，不是通用长度承诺。

可复验脚本：[verify_codex_prompt_isolation.py](../../scripts/diagnostics/verify_codex_prompt_isolation.py)。运行 `.venv/bin/python scripts/diagnostics/verify_codex_prompt_isolation.py`；它使用真实项目提示构造函数、临时无认证目录及 localhost 拒绝服务，仅输出布尔值、数量和哈希，不输出完整请求或认证头。3 个配置条件都不消耗真实模型 token。本机 launcher 会重设账号目录，脚本因此安全解析其 native binary 路径后直调原生程序；正式 planner 的命令构造另有离线单元测试覆盖。

## Skill 隔离

FDS `skills=[]` 控制自己的冻结技能列表；`--disable skill_search` 禁搜索工具。二者都不足以单独证明 Codex 内置技能说明没有进入提示。

本次显式设置 `-c skills.include_instructions=false`，并保留禁用 skill_search/plugins/memories/外部工具等设置。离线检查确认技能目录说明（含 openai-docs 等）消失。配合替换原生 coding base，也移除了原生基础提示中的通用技能使用规则。没有删除共享技能目录，也不声称磁盘扫描一定停止。

## 我们的新请求布局

1. **`system_prompt.txt`**：harness 职责及宿主提供的工具、执行、记忆、FDS、后验核对和坐标协议。部分工具上限随剩余预算更新，不能把整份文件声称为逐次字节不变。
2. **`prompt.txt`（stdin user）**：当前 goal、观测说明、memory、上次动作及反馈、待核对事件、剩余预算。规则不在这里重复拼接。
3. **图片**：仍使用 `--image`；没有增加帧数或恢复已过期图片。
4. **`schema.json`**：仍通过 `--output-schema` 约束响应。
5. **`prompt_manifest.json`**：记录两部分文本哈希、协议版本和请求的技能隔离设置。普通运行记录是请求配置审计，不等于每次都截获远端实际请求。

模型仍为 gpt-6-astra，effort medium，workspace-write、approval never。Codex 的权限/环境等附加上下文不因 base override 自动消失；本次不声称完整删除所有宿主消息。

修改位置：[CodexPlanner](../../src/fds/planners.py)、[提示拆分](../../src/fds/task_harness/protocol.py)、[接口实验入口](../../scripts/experiments/interface_ablation/run_one.py)。后台代码优化器新启动的调用也禁用 skill 说明；它保留编码职责，不使用机器人控制的 base。已经运行的进程与历史快照没有被热替换。
