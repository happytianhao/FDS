# RoboMME / RoboMME-Plus：模型怎样获得任务规则、世界状态和执行语义

审查日期：2026-11-10。源码版本记录在 [source_inventory.json](source_inventory.json)。本次拉取了 Embodied-Astra、3-tier Astra、PonderPounce、FrameSamp+Modul、MEMI、SimpleMemVLA 六个公开仓库，并检查本机官方 Plus 适配器。前三个为 Git checkout，后三个为固定 commit 的源码归档；未下载或运行各 VLA 的大模型权重。

本次环境版本见 [environment.json](environment.json)。本报告只发布源码审查与实验记录，不包含原对话的恢复记录。

这份报告区分三种证据：**源码事实、作者公开实验、本机训练集实验**。源码能证明某个信息被提供给模型，不能独自证明它是成功的必要条件。本机小样本消融用于发现具体机制和反例，不能替代多种子统计实验。

## 1. 最值得借鉴的设计

不能仅用“任务描述写得详细”概括这些方法。有效接口至少需要回答六个问题：

1. **目标和完成事件**：到底累计什么事件，何时算一次，最终怎样结束？
2. **动作含义**：输出的是世界坐标 waypoint、像素指向的子目标，还是关节动作序列？夹爪命令何时生效？
3. **时间和动力学**：模型思考期间世界是否继续运动？一次动作耗时多少？等待怎样实现？
4. **可观测信息**：哪些图像属于示范、当前执行、动作中间过程？坐标和时间戳如何解释？
5. **记忆更新**：哪些事实会消失，哪些必须保留；事件有没有发生，证据在哪里？
6. **验证和恢复**：命令发出不等于动作完成。如何从新观测检查夹持、到位、释放、接触、次数和身份，失败后怎样修正？

Embodied-Astra 在通用物理接口上写得清楚；3-tier 把完成检查独立出来；MEMI 把记忆写入、保留、使用组织成任务专用规则；PonderPounce、SimpleMemVLA、FrameSamp+Modul 则把相当部分执行知识放在训练数据和动作网络中。因此，“提示词没有写某条规则”不能直接解释为该方法完全不知道这条规则。

| 方法 | 如何告知任务与世界规则 | 保留/提供什么观测 | 如何落实为动作 |
|---|---|---|---|
| Embodied-Astra | 通用物理接口说明 + 本集 goal | 稠密 front 历史、当前 wrist、TCP/手指状态、标定、时间 | 世界坐标 waypoint，由官方规划器执行 |
| One-shot Astra | 上述接口 + 同任务成功训练视频、阶段与时间标识 | 当前 episode 历史 + 训练例的示范/成功执行图像 | 同一 waypoint 接口，从范例归纳操作语义 |
| 3-tier Astra | 每任务模板、子目标语法、顺序/计数规则 | 当前图、执行起点、示范/执行拼图、完成记录 | 像素指向的子目标 → 学习型动作器；另有完成监测 |
| MEMI | 离线优化的记忆字段说明与行动/更新规则 | 当前采样帧、必要关键帧引用、结构化记忆 | Act 直接动作/等待；Plan 调用 oracle 技能，须分开比较 |
| PonderPounce | 持续追加任务、示范和认知；训练学习执行知识 | 因果图像上下文、当前子目标及认知新旧程度 | 子目标/隐表示条件化 π0.5 |
| SimpleMemVLA | 机器人、频率、动作段长度、目标、当前子任务提示 | 带时间戳的历史视频 + 当前另一视角 | 子任务及其隐状态条件化动作头 |
| FrameSamp+Modul | 任务语言 + 训练示例中的动作监督 | 采样历史的池化图像 token 与时间/位置编码 | 记忆特征调制动作网络 |

## 2. 榜单口径

使用官方 [Plus 动态数据](https://robomme.github.io/data/plus.json) 的快照，数据更新时间为 2026-10-28，读取日期为 2026-11-10。每个方法 800 个测试 episode。仓库里旧版 submission Markdown 的 one-shot 数字与当前动态数据不同，以下使用当前数据。

| 方法 | Plus 成功数 | 成功率 | 本次覆盖 |
|---|---:|---:|---|
| One-shot Embodied-Astra | 282/800 | 35.25% | 官方适配器、示例组织和榜单逐任务数据 |
| Embodied-Astra | 181/800 | 22.625% | 原仓库、Plus 适配器、本机统一控制接口消融 |
| 3-tier Embodied-Astra | 76/800 | 9.5% | 任务提示、规划器、完成监测输入 |
| PonderPounce | 23/800 | 2.875% | 上下文追加、子目标语言、动作条件化 |
| FrameSamp+Modul | 22/800 | 2.75% | 记忆采样、配置、训练接口 |
| SimpleMemVLA | 16/800 | 2.0% | 视频消息构造、时间戳、子任务和动作接口 |

MEMI 是原版 RoboMME 的相关方法，不在这个 Plus 快照中。其作者报告 Astra-Act 97.0%、Astra-Plan 99.4%；Plan 使用 oracle 技能执行器，不能与直接 waypoint 或学习型动作控制器混为一组。原版高分迁移到 Plus 后大幅下降，说明需要检查任务分布、执行前提和控制器能力，不能只看模型名称。

## 3. 各仓库具体向模型提供什么

### Embodied-Astra：通用物理接口 + 完整可见历史

源码：[astra_waypoint_eval_v3.py](https://github.com/jackbjed/embodied-astra-robomme/blob/79dcc7f73f536948cb039103d3fc5e722d689207/harness/astra_waypoint_eval_v3.py)。Plus 适配器在本机 `third_party/robomme_plus/scripts/plus_eval/embodied-astra/`。

输入包括任务目标、动作过程中 front 相机的稠密采样、最新 wrist 图、实际 TCP 姿态、手指位置、剩余步数，以及相机内外参。输出一个六维世界坐标姿态和离散夹爪命令。提示明确机器人基座位置、朝向、机器人左右与正面相机镜像关系、向下抓取姿态、夹爪状态持续性，以及规划器先到位再执行夹爪命令。

世界时间按 20 Hz 仿真推进；模型等待网络推理期间世界暂停，动作执行期间继续运动。提示给出不同位移对应的大致动作耗时，提醒模型不能靠“思考几秒”等待移动物体。原接口用 Responses 会话保留先前可见消息；Plus 用 execution control steps 区分示范时间和实际操作时间，初始图像超限时均匀采样并保留端点。

优点是模型能把像素、空间、动作耗时连起来。欠缺是各任务的事件判据大多依赖模型自行从 goal 和示范推断。还存在版本假设：原提示称完全相同姿态有规划风险，而本机 Plus waypoint 接口已有到位后的 hold 分支。不能把旧提示当作跨版本不变的物理定律。

原仓库 LR 修正有此前失败分析背景，因此应准确称为“公开固定接口提示”，不把其设计过程说成从未观察过任何评估现象。

### One-shot Embodied-Astra：用成功视频补足任务语义

核心文件：Plus 适配器中的 `eval_plus_fs.py`、`one_shot_examples.py`、`one_shot_images_prompt.txt`。每个任务通常提供一集 medium 训练例，InsertPeg 采用 episode 1，MoveCube 没有对应示例。示例区分示范阶段与机器人成功执行阶段，配对展示 front/wrist，标帧号和时间，重复示例的目标，要求迁移规则而非复制坐标。

这种设计实际补充了“什么事件算成功”“示范和执行如何对应”“应如何结束”的可观察证据。当前 Plus 表中 SwingXtimes 从 28% 到 98%，ButtonUnmask 从 34% 到 94%，但 BinFill 从 30% 到 28%、VideoRepick 从 18% 到 14%、RouteStick 从 12% 到 8%。这说明成功示例有时帮助很大，也可能无益；不能推导出所有任务都应无条件追加更多视频。上述是作者榜单结果，并非本机成对随机实验。

### 3-tier Astra：任务规则 + 子目标语法 + 完成监测

源码：[任务提示目录](https://github.com/bingaochen/Astra-on-RoboMME/tree/4c3fd6a8667e7a219e547fe1a533a4b9726fc6db/examples/champ/prompts)、`examples/champ/core.py`、`examples/champ/input_contract.py`。

规划器接收 goal、已完成子目标、已发出命令、当前 front、固定 execution-start 参考图、示范及执行 contact sheet。输出面向 VLA 的子目标，定位使用 `<y,x>` 像素顺序和 0–255 范围；这与 Astra 的世界坐标 waypoint 完全不同。任务模板还解释示范/执行阶段、对象身份/当前位置、抓起/放下/按钮终止等语义。

完成监测器单独看当前子目标、最近图像、命令开始时 front 和当前 wrist，返回完成判断；规划器不能仅按“已经发出命令”累计成功。例如 PickXtimes 把提起和放下作为不同阶段，验证一对阶段后才累计次数。InsertPeg 区分抓取端与插入端，以及相对机器人坐标的近远关系。

记忆不是所有任务都用统一低频截图：按钮触发、短暂显色、StopCube 等采用更密的保留策略。StopCube 还专门区分穿越计数与按键准备/触发。可借鉴的是“记忆采样由事件信息决定”和“控制闭环有明确判据”；其具体 lead steps 和原环境假设需要在 Plus 重新验证。完成监测器本身也是学习模型，并非真值。

### MEMI：任务专用结构化记忆 + 因果更新 + 使用检查

源码：[MEMI](https://github.com/memi-robot/memi/tree/bb6c37ce6c14bd1a204c86a91cc0cb36f565a1f0)，重点为 `memi/act/prompts/context.py`、`policy.py`、`memory.py`，以及 `artifacts/act/runs/opt_astra_act/<Task>/texts.iteration5.json`。字段名列表见本报告的 source_inventory.json，原描述见对应源码文件；下方逐任务表归纳其用途。

离线优化过程区分看过完整轨迹的标注器与在线只能看过去和现在的策略。它检查三件事：记忆加当前观测是否足以选动作；记忆能否从当时已有信息更新；字段是否真的会被动作使用。在线输出包括动作、变化的记忆字段、`used_fields`。需要视觉细节时保存关键帧引用，而不是强迫所有信息变成一句摘要。

Act 接口能显式 wait，通过保持关节目标推进仿真；它只提供选中的近期动作帧、当前 wrist、标定，以及从存储恢复的关键帧，不隐含保留全部历史。记忆字段往往是事件计数、阶段、目标身份、位置和关键证据帧的组合。

作者原版 ablation 表为：Astra-Act Full History 85.3 / 初始化 88.3 / MEMI 97.0；Astra-Plan 91.3 / 92.4 / 99.4。它支持“记忆结构值得优化”的判断，但不等于本机已经证明某个单独字段必要：Act/Plan、帧采样、推理强度、控制器和调用预算都需要对齐。

需要警惕优化后提示中的环境专用先验。实际文件里，PatternLock 假设从第一个节点起步并采用固定理想网格坐标；SwingXtimes 的规则以右侧先行和奇偶次序组织；StopCube 有固定周期模型和经验提前量。Plus 可以改变起始状态、起始侧、轨迹和接触条件。Plan 文件还保留训练例 ID 与示例坐标，说明提示相当具体；这不能据此指控测试泄漏，但足以要求检查迁移适用范围。

### PonderPounce：持续上下文 + 子目标/隐向量连接动作网络

源码：[PonderPounce](https://github.com/worv-ai/PonderPounce/tree/723df35762bb641e1d520e4fa9359b98644adc21)，重点 `ponderpounce/inference/append_context.py`、`ponderpounce/pi05/prompt.py`。

前缀是任务与示范图像，后续追加执行观测和认知输出。动作网络可接收任务、当前子目标、认知隐表示及其年龄。动作细节主要由训练出的 π0.5 执行，不能把子目标文字误解为模型唯一掌握的世界知识。该框架的异步认知延迟与 Astra 的推理期间仿真暂停不是同一个时间机制。Plus 官方评估还限制上下文长度，并对旧执行历史裁剪重建缓存。

值得借鉴：持续提供阶段语义，同时显式表达认知结果的新旧程度。需要验证：旧子目标在物体移动或执行落后时是否仍然有效；被裁剪的是无用旧帧还是唯一关键事件。

### SimpleMemVLA：带时间的视频上下文 + 当前子任务

源码：[SimpleMemVLA](https://github.com/OpenBMB/SimpleMemVLA/tree/453d6ec293e63318bbc2bce545d0b9256ab1bc51)，重点 `simplememvla/data/messages.py`、`simplememvla/benchmarks/robomme.py`。

消息构造器写入机器人身份、控制频率、下一段动作长度、带时间戳的历史视频、当前其他相机视图和目标，并要求辨认当前子任务。RoboMME 配置为 Panda、20 Hz、八维动作、front 历史加当前 wrist；训练/推理共用消息格式和条件化跨度。子任务语言及其隐藏状态供动作头使用，精确操作知识主要在训练中获得。

这提供了“过去发生什么、现在执行到哪一步”的统一时间语义，但没有像 Astra 一样靠文本完整交代标定几何和夹爪物理接口。原版 88.3% 与当前 Plus 2.0% 的差异不能只归咎于记忆长度；次数、分布、动作执行和任务终止规则都可能参与。

### FrameSamp+Modul：采样与特征调制

源码：[robomme_policy_learning](https://github.com/RoboMME/robomme_policy_learning/tree/ac846a9a4aec841e483b1dc5d620dfe426bde853)，重点 `src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml`、`shared/mem_buffer.py` 和训练配置。

它通过采样、图像 token 池化、位置/时间信息及记忆特征调制连接策略网络。检查的配置使用 512 的记忆 token 预算、每图 16 tokens、单视角；Plus 适配器对长示范做帧数限制。任务规则和动作语义主要从训练示例进入参数。对照其他方法时，应同时看“哪些帧被保留”和“信息通过哪条路径进入动作网络”，不能只比较 prompt 长短。

## 4. 16 类任务需要交代的内容

以下“关键条件”是从公开接口、训练提示和可见任务现象整理出的设计要求，未全部经本机消融证明为统计意义上的必要条件。避免把评价器内部阈值、隐藏对象状态或 oracle 子目标提供给待评估模型。

| 任务 | 模型需要的任务/世界信息 | MEMI 的具体记忆字段 | 执行中应验证的事件 |
|---|---|---|---|
| BinFill | 目标颜色向量、每色数量、容器、终止按钮；动态来料与干扰物 | deposited_blue/green/red、held_color、bin_center | 实际投入对应容器后更新颜色计数；抓住不等于投入；达到目标后结束 |
| PickXtimes | 一次完整提放怎样定义，释放和离开物体的作用 | placements_completed、milestone、button_center_xy | 提起→放下→释放→离开；从新观测确认阶段，不能按命令数累计 |
| SwingXtimes | 本集先到哪侧，往返与单次到达的区别，目标是否移动 | target_visits、right_target_center、button_center | 确认每次到达及次序，再累计周期；最后放下、按键；不能固定右侧先行 |
| StopCube | 穿越事件、计数方向、动作耗时、按键时世界仍运动 | motion_reference | 基于连续帧计数和估计速度；提前准备，再依据预计接触时刻触发 |
| VideoUnmask | 示范揭示的颜色/容器对应与请求顺序，执行后仍遮挡 | demonstration_reference、completed_lifts | 按身份找当前容器，确认提起事件，再推进顺序 |
| ButtonUnmask | 如何触发显色，证据出现窗口，随后重新遮挡 | uncovered_scene、button_pressed、completed_pickups | 确认按钮触发，及时保存显色帧，按身份顺序操作 |
| VideoUnmaskSwap | 初始身份映射与后续交换轨迹 | target_locations、completed_lifts、final_layout_keyframe | 身份随容器运动更新；最终位置不能沿用最初位置 |
| ButtonUnmaskSwap | 按键显色与交换阶段的因果顺序 | target_locations、tracking_frame、interaction_state、pickups_completed | 先记录身份，再连续跟踪交换；当前静态图不足以恢复隐藏颜色 |
| PickHighlight | 亮起的是哪个对象、顺序和证据时间点 | highlight_frame、button_pressed、completed_targets | 在提示消失前记录身份；当前外观相同也不能任意选择 |
| VideoRepick | 示范选择的对象身份、需要重复次数及终止条件 | target_location、completed_repetitions、button_location | 每次完成物理事件后计数；位置变化时继续跟踪同一对象 |
| VideoPlaceButton | 哪个按钮事件之后的放置才有效 | post_button_target | 按事件对齐示范时间；不能选到按钮前相似位置 |
| VideoPlaceOrder | 第 k 次完整放置事件，目标身份和后续移动 | selected_placement | 区分经过目标与实际放置；记录目标身份并定位其现在位置 |
| MoveCube | 示范方法及关键物体关系；是否有对应训练范例 | method、initial_scene | 保持方法所需约束，确认目标结果；五种方法不宜套一个固定模板 |
| InsertPeg | 抓取端/插入端、示范关系、工具长度与姿态 | demonstration | 抓持稳定后对齐入口，闭环修正并确认实际插入；TCP 到位不等于孔内接触 |
| PatternLock | 有序节点、连线轨迹、起始状态、接触高度与笔尖偏移 | pattern、reached_index | 先从当前姿态到首节点；确认实际触达后推进，避免漏节点或穿越错误节点 |
| RouteStick | 示范路线、障碍关系、棒长/倾角变化 | demonstration、completed_legs | 用当前工具几何验证路线可行；确认每段通过，不能照抄 TCP 轨迹 |

按任务族归纳：计数任务需要**可观察的事件定义**；遮挡任务需要**身份与位置分开记忆**；历史引用任务需要**事件时间与当前目标对应**；模仿任务需要**保留轨迹约束并适配当前工具与初始姿态**。

## 5. 本机消融设计及边界

脚本：`scripts/experiments/interface_ablation/run_one.py`、`run_panel.py`、`replay_actions.py`。结果目录：`runs/interface_ablation_20261110/`。

使用本机 A100、已安装 RoboMME-Plus commit `69a2fe25cb476069943aa7dd754d19f0932d9ec3`。四个任务各使用 train episode 0：PatternLock、VideoPlaceOrder、SwingXtimes、ButtonUnmask。固定同一任务种子、目标、官方步数预算、waypoint 控制器、gpt-6-astra medium，比较以下四组：

| 组别 | 唯一设计变化 | 检验问题 |
|---|---|---|
| full | 固定上游通用提示、front stride 3、当前 wrist、状态/标定、可见历史 | 本地统一接口基线 |
| no_dynamics | 删除整段 MODEL_NOTES；动作格式和 LR 提示仍保留 | 动力学/执行提示块的作用；不是单条物理规则消融 |
| no_history | 每轮仅最新观测组，仍保留目标和系统提示 | 显式视觉历史及此前回答的作用；同时移除过去推理文本 |
| sparse16 | 动作和示范 front 采样 stride 从 3 改为 16 | 稀疏观察是否漏掉事件；同时减少图像预算与推理负担 |

这是同一 waypoint harness 内的组件消融，**不是六个完整系统重跑**。VLA 权重、MEMI 离线优化和 oracle Plan 不在本地对照中。尚未执行的 `no_calibration`、`no_lr` 只是可用变体，不能写成已有结果。另在 PatternLock 和 SwingXtimes 上启动 `verified_progress` 两条探索性实验：在 full 提示后追加跨任务的可观察事件验证、身份/位置区分和接触时机提醒。该段提示预先写在 runner 中，在看到基线失败后选择这两个任务运行；它不包含诊断读出的评价器阈值。结果应独立于主消融表报告。

推理采用关闭工具的 Codex CLI，每次新请求显式重放保留的图像和可见回答。它没有原版 Responses 的 `previous_response_id`，也不保留模型隐藏推理；调用上限 40 次而非上游 150 次。各组一致，但结果不能冒充官方榜单复现。初始图片上限 750，保留图片上限 1500。模型只接收公开 RGB、自身状态、标定、目标及执行反馈；不接收评价器隐藏对象状态、任务实现或 oracle 动作。

每次请求保存 prompt、图像顺序和 SHA256、JSON schema、原始 CLI 事件、回答、用量、动作与仿真时间。推理中断记为 `success=null`，不算任务失败。恢复只对基础设施中断进行：在同种子仿真中重放已完成动作，必须得到完全一致的状态及 PNG SHA256 后才能从中断处继续；原始失败尝试保留。这样不会重采样一条成功路径来替换已有模型错误。

每个条件目前仅一集，模型采样未固定，不能由某组一个成功/失败证明必要性；也不能用四个任务代表全部 16 类。可信的因果下一步是：以本次定位的机制为假设，在多个训练种子上做单因素配对，再锁定规则评估未参与调试的任务实例。

本机结果及逐例分析见 [results.md](results.md)，原始结果汇总见 [results.json](results.json)。

另有两个不调用模型的事后动作干预，用于检验失败机制，单独记录，不计入策略成功率。VideoPlaceOrder 在释放后保持原位 40 个控制步，再执行原来的抬离；SwingXtimes 固定已有动作的其他分量，仅降低搬运阶段高度 2 cm。后者的干预是在读取评价器后选择的，属于有特权信息的诊断，不能包装成不依赖特权信息的新策略结果。

## 6. 建议落地的接口格式

每个任务族用同一种外壳，任务规则由允许的训练示例归纳：

```text
Goal: 本集目标参数、顺序/次数、结束动作
Action contract: 坐标系、单位、控制器行为、夹爪/等待语义
Clock: 帧时间、执行步数、模型推理期间是否推进、动作耗时
Observation: 示范与当前阶段、相机/标定、最新证据帧号
Memory: 事实值、对象身份、当前估计位置、证据帧、更新事件
Progress: 已验证事件数、当前阶段、尚未证实的动作结果
Next action: 一步动作、预期可观察结果、下一轮验证点
```

不要在 prompt 里只写“已经完成了 3 次”。应让计数对应证据，例如“第三次放置：已释放且新帧可见物体留在目标区域；已后退”。不要只写“蓝色在左边”。应记录“容器 A 曾显露蓝色；最新跟踪到右前位置；证据来自第 n 帧”。这些表达能让后续动作检查记忆是否过时。

最优先值得进一步验证的是三组机制：事件驱动的关键帧保留、完成后才更新的结构化进度、任务示范中可观察的成功判据。使用 Plus 时，固定方向、固定网格/工具长度、固定周期和“上一命令必然完成”等先验都需要在训练场景重新检查。

16 个任务的具体接口草案见 [task_contracts.md](task_contracts.md)。其中的目标参数、记忆、执行顺序、完成与恢复条目可作为后续从训练样例归纳规则的外壳；它们未作为本轮策略输入，也尚未验证为性能改进。
