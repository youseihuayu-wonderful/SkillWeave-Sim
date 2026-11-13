# SkillWeave Sim：G1 VLA 基线第一阶段执行计划

## 0. VLA 路线的目标；人工技能与 planner 支线见第 11—12 节

**让冻结的 G1 VLA 在真实 MuJoCo 物理与完整拒绝检查下，完成它已经训练过的叠方块任务，先建立 Adaptive-OFF 基线。**

这里的“真实 MuJoCo”指本地物理仿真计算，不是真机。最初计划分支只交付证据核对和观察器；第 10 节记录离线首目标诊断，第 11 节记录人工技能控制器，第 12 节记录真实 planner 的受限调用与结果反馈。它们均不代表已经完成真实 VLA 闭环，也不允许向真机发送动作。

本阶段不增加手机、不训练新模型、不接人类视频、不进行速度优化或真机运动。每个后续阶段必须以上一阶段的证据为输入，而不是以网页能显示机器人为通过条件。

## 1. 项目与分支边界

- 新项目：`youseihuayu-wonderful/SkillWeave-Sim`。
- `main` 和 `work/skillweave-sim` 保留已验证的 **Panda/MuJoCo** 参考项目，不将其结果改称 G1 结果。
- `plan/g1-baseline` 保存本计划和 G1 场景观察工具。它不是已经迁移完成的 G1 VLA 运行版本。
- `work/g1-baseline` 当前包含固定 G1 子模块、首目标诊断、独立人工技能基线及真实 planner 接入；不覆盖原 G1 仓库，不混用 Panda 与 G1 接口，尚非完整神经闭环。
- G1 来源：`youseihuayu-wonderful/g1-vla-control`，核对时本地工作区干净，提交为 `a5e40a25c5c080ae114a925dae94215c7cb6d793`。
- SkillWeave 计划分支起点：`cbaa518af63e0e18fd80bc159e7189087c74391c`。

## 2. 冻结的机器人、模型和数据契约

| 项目 | 第一阶段固定内容 |
|---|---|
| 机器人 | Unitree G1 Education、左右 Dex1-1、固定站姿双臂操作 |
| 控制范围 | 不由 VLA 控制腿、行走、躯干或平衡 |
| 策略 | `LGG100/stack-cube-eef-24k`，配置 `pi05_g1_eef` |
| Checkpoint revision | `cced7a7ff7b454fdcac555457a1a2a3dc262ac77` |
| 数据 revision | `LGG100/Stack-the-cubes@c468ef259ff8bfad1b3b0e7b2c1f45efbb0b30ba` |
| 作者数据变换 | `leihao100/openpi-fintune@29030046fd6a2810201db67b9804f243e0af3218` |
| 作者部署参考 | `leihao100/g1-client@1422e8d6ef674aa047cfb2878bc7dae54b118fbe`；只引用和审计，不直接运行其 publisher |
| 契约 | `g1_edu_dual_dex1_eef_v1`，版本 `1.3.1` |
| 三相机 | `observation/cam_left_high`、`observation/cam_left_wrist`、`observation/cam_right_wrist` |
| 图像预处理 | 480×640 uint8 RGB，保持比例补零缩放到 224×224，不换成中心裁剪 |
| State/action | 16-D，pelvis frame，绝对 EEF 目标；左右手 XYZ、xyzw 四元数、左右夹爪角 |
| EEF 定义 | 各侧 `wrist_yaw_link` 局部 +X 方向 50 mm |
| Dex1 | 0 rad 闭合、5.5 rad 打开 |
| 动作部署节拍 | 15 Hz；action horizon 32；`discrete_state_input=False` |
| MuJoCo 步长 | 0.002 s |
| 自适应速度 | OFF，不改变动作点或其原始时间分配来制造基线通过 |
| 规范化 | 只按已核对的作者 consumer 边界归一化非零四元数；保留 raw/canonical 两份 hash，不偷偷修改 XYZ 或夹爪值 |
| 归一化统计 | 使用该 checkpoint 配套统计并记录 hash，不以公开原始关节数据的统计代替 EEF policy 统计 |

15 Hz 是这里冻结的动作部署节拍，不表示必须每秒重新运行神经网络 15 次，也不等于硬件底层伺服/DDS 发布频率。推理、预取、时间对齐和提交须分别记录。

任务沿用已有 `stack_scene.TASK_PROMPT`，不临时换成拿手机：

> Stack the blocks by color: put the red block in the center, then stack the blue block on the red block, then stack the yellow block on the blue block.

公开数据 `LGG100/Stack-the-cubes` 的审计记录为 100 episodes、95,966 帧、约 30 Hz，原始 16 维是 14 个手臂关节位置加两个夹爪角，**不是直接的 EEF-16**。原始数据频率和 15 Hz 部署节拍不能混为一谈。转换必须使用已核对的作者变换与相机/状态同步规则，不能仅凭维度相同直接接入。

## 3. 已有事实，以及不能从中得出的结论

### 3.1 已有正面证据

作者配置重验证记录报告了真实权重严格恢复，以及 50/50 个有限 `[32,16]` 输出。它证明模型加载和一定范围的离线语义一致性，**不证明已经抓取或堆叠成功**。

较旧 `REAL_LGG100_ADAPTIVE_WORKFLOW.md` 中“真实神经调用为 0”等历史状态不再代表这些更新后的证据；不能直接复制旧计划的完成状态。不同报告中的 `g1_contract_verified` 也必须按其证据范围解释，不能用语义审计通过覆盖执行门槛失败。

### 3.2 当前保存的阻塞证据

| 保存记录 | 事实 | 本阶段如何使用 |
|---|---|---|
| Adaptive-OFF realtime | `completed_cycles=0`、`task_success=false`、`execution_performed=false` | 不把这次记录称为 VLA 闭环成功 |
| 外层拒绝 | `swept_path_preflight_failed` | 这是包装层原因，不能据此直接认定发生了碰撞 |
| 下层拒绝 | `reason=unreachable`，目标索引 0，`collision_reasons=[]` | 优先诊断首目标的 IK 收敛和调用链 |
| 保存的 IK trace | 30 次迭代未通过，250 次能够收敛，最后未贴住关节限位 | 支持“求解预算/初始化可能造成拒绝”的待验证假设，不证明全路径或任务可行 |
| Realtime 提交观测年龄 | 119.93975099176168 ms，项目门槛为 100 ms | 即使解决 IK，仍需独立解决调度/新鲜度预算 |
| 诊断提交范围 | `committed_prefix_only`，检查 2 个目标 | 不把前缀检查说成整个 32 点 chunk 的检查 |
| 保存的时间参数 | `prefix_duration_s=1/30 s`，正式部署约定为 15 Hz | 审核前缀时间参数与时间网格的定义；暂不将其直接判为根因 |

Realtime 与 paused 报告的首动作数值不同，不作为同一个 chunk 在不同时间模式下的严格配对证据。后续比较必须使用相同 raw/canonical action hash、相同初始物理状态。

本轮接受上述失败记录为已有事实，不重新运行旧失败来“确认它确实失败”。应先形成代码路径解释，再运行修正方案的最小验证。

### 3.3 来源指纹

核对时得到：

| 文件 | SHA-256 |
|---|---|
| `g1_policy_contract.yaml` | `839e2a3576b3fbb7b09aa457838b4187e53b100f91fa5f48dc74458b21163374` |
| `results/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json` | `f0452531398a05b05120b7e39b5a4d6fe52de6f23e3ac28bcb3ad2109315c99f` |
| `results/lgg100_author32_closed_loop_t0_offset008_baseline_paused.json` | `8a79dc26be3f21a189244bc519929829de5afa70c5747acc20db334e0df2fe1d` |
| `results/lgg100_author32_online_ik_convergence_trace.json` | `7880e37eb86da86cc39b38c6ae9c952d5125b141698214c04e78cede44ca1126` |

## 4. 按依赖执行的工作包

以下定义 VLA 各阶段及准入条件；实际离线范围见第 10 节，第 11—12 节另记独立人工技能和 planner 证据。未通过的神经执行门槛仍保持关闭。

### E0：建立独立、可复现的 G1 实验入口

**操作：**

1. 在新项目的独立 G1 实现分支接入已有 G1 场景、contract、IK、夹爪映射和拒绝检查；Panda 继续留在参考分支。
2. 固定源提交、第三方资产版本、checkpoint、作者 transform 与部署参考版本。
3. 记录实际 Python、MuJoCo、JAX/jaxlib、CUDA、驱动、GPU、模型加载参数与依赖清单。历史 L40S 结果不能替代本轮环境检查。
4. 从已有授权存储取得配套 checkpoint/norm stats、同步三相机样本、raw/canonical chunk 和 manifest；逐项核对来源及 hash。
5. 核对场景、初始站姿、桌面、方块、相机、EEF 与 pelvis/world 变换。不能只因图像“看着相似”就写标定通过。
6. 确认原任务成功判据的对象身份、红蓝黄堆叠关系、稳定窗口、超时和失败分类，并在执行前写入本轮协议。仅 EEF 到位不能算堆叠完成。

**产物：** `g1_baseline_manifest.json`、`g1_baseline_protocol.json`、依赖/资产清单。

**通过门槛：** 所有实际使用的输入和判据均可追溯；仍缺的必要样本或配置被明确列为阻塞，不猜默认值。真机命令发布能力保持关闭。

**当前可用性边界：** 本机已核实 100 条缓存 Parquet；没有在已搜索位置找到配套 checkpoint/norm stats、author32 同步样本及 raw/canonical NPZ 包。本轮通过已有 `shihua-vla` SSH 配置尝试只读访问历史 GPU 工作站，但 banner exchange 超时，未能验证远端文件或 GPU；不据此假定远端资源不存在，也不重复训练。

### E1：解释并修复首目标的拒绝

**操作：**

1. 沿 `lgg100_quarantined_closed_loop.py` 的实际路径，核对当前调用的 IK/preflight 实现、初始关节解、迭代预算、位姿定义、返回码和拒绝分类。
2. 使用保存的同一个初始状态与目标，检查 pelvis/world、EEF site、左右手顺序、xyzw、单位和夹爪映射；不改目标坐标来消除错误。
3. 优先审查 30 次预算、warm start、冗余分支及已有 fast sequential IK 的适用性。250 次收敛只作为离线参考，不能不计延迟地直接加大线上预算。
4. 对原失败目标及已有正确性 corpus 验证候选修正，记录位置/姿态残差、限位、碰撞结果和实测耗时；不可达或碰撞样本仍必须拒绝。

**产物：** `first_target_diagnosis.json`、最小修正、相同输入的修正验证记录。

**通过门槛：** 有可复现的拒绝原因和修正依据；沿用已有 `5 mm / 3°` 追踪门槛，未放宽碰撞与限位检查；危险目标不能因为修正而被放行。离线可达不自动解锁正式闭环。

### E2：解决正式链路的新鲜度预算

**操作：**

1. 分别记录相机采集、接收、推理开始/结束、IK/preflight 开始/结束、动作提交/生效时间。
2. 标注时钟来源；不能直接相减两个未同步主机的 monotonic 时间。观测年龄按实际被消费的采集时间计算，不用 HTTP 接收时间冒充采集时间。
3. 在相同输入下测量推理、渲染/传输、IK、检查与提交各段耗时，区分冷启动编译和稳态。
4. 复用并核对作者的预取、时间对齐与 joint-space blend 语义，不改变观察/动作契约或把暂停物理的运行当实时运行。
5. 修复造成超时的实际计算/调度问题；场景变化后不得复用失效的碰撞判定来节省时间。

**产物：** `g1_timing_trace.jsonl`、分段延迟统计、新鲜度与断线拒绝记录。

**通过门槛：** 每次实际提交满足冻结的观测年龄 ≤100 ms，以及 contract 的 response-age 和 watchdog 条件；过期动作不得执行。若当前硬件不能满足，报告不可行和实测瓶颈，不能默默增大阈值。

### E3：检查 chunk、物理跟踪与阶段交接

**操作：**

- 审核 32 点 action、15 Hz 时间网格、首点时间含义、预取、时间对齐与过渡。
- 明确预检覆盖范围；每个实际提交的目标及其连续运动路径必须经过检查，不用只检查两个点证明整个 chunk。
- 验证相邻 chunk 的 IK 连续性、双臂/手指/躯干碰撞、关节限制、跟踪误差、夹爪状态与物体真实接触。
- 注入失效观测、非法 quaternion、IK 拒绝和断线等既有故障路径，确认暂停/hold 及拒绝语义不被绕过。

**产物：** `g1_chunk_gate_report.json`、物理轨迹与拒绝案例。

**通过门槛：** 实际执行路径全部有检查证据；未知、超时或异常情况停下并保留原因。确定性 fixture 仍只证明下游控制链，不冒充神经策略成功。

### E4：完成第一个 Adaptive-OFF 真正 VLA 任务

只有 E0–E3 通过，才在隔离 MuJoCo 进程运行一条完整任务：

> 新鲜三相机/状态 → 真实冻结 VLA → 检查 → 物理执行 → 新观测 → 下一 chunk → 原任务判据。

- 不加载手机任务；不调用新 planner 为模型补动作；不回退到手写轨迹后仍记为 VLA 成功。
- 原任务超时、掉落、意外碰撞、观测过期、IK 或运行时异常均按协议保存为失败。
- 保存状态、动作、时钟、接触与任务判据；允许检查对应录像，但成功不能由播放完成或末端到位代替。

**产物：** 一条完整原始运行记录、对应录像和 `adaptive_off_baseline.json`。

**通过门槛：** 原始任务确实完成、模型持续闭环、无非法或未检查动作；实际动态和物理稳定性满足冻结协议。一个成功案例只是工程里程碑，不是泛化结论。

### E5：冻结后的基线确认

1. 开发结束后记录源代码、模型、参数和 evaluator 指纹。
2. 在同一已训练任务分布中预登记 10 个确认初态/随机种子，记录完整初始物理状态；确认集不用于调参。
3. 本计划的首轮工程目标为 10 个预登记案例全部完成，并且均满足动作、接触、新鲜度、超时和物理稳定条件。达不到就如实报告失败，不删样本或改分母。
4. 若确认后修改了算法或参数，使用新版本号和新确认批次，原记录保留。
5. 分开报告任务成功、仿真完成时间、wall time、推理/检查耗时、掉落、意外碰撞、拒绝/超时、跟踪误差与 settling。

**产物：** `g1_baseline_confirmation.json`、逐案例数据与基线报告。

10 例是本计划预先选择的小规模工程门槛，不是统计上的高置信成功率证明，也不代表手机、新场景或真机已经通过。本阶段不做 Adaptive-ON 优越性结论。

## 5. 当前就能开始的第一项

**从 E0 开始，在独立 G1 实现分支整理已存在的文件与输入清单；随后对保存的首目标做 E1 的代码路径诊断。**

优先处理的两个问题是：

1. 为什么现有 30 次 IK 的线上路径把一个离线可收敛目标归为 `unreachable`？
2. 在不放宽检查的前提下，怎样使完整观察→推理→检查→提交满足 100 ms 观测年龄预算？

先得到解释和最小修正，再运行对应的窄验证；不先重跑所有失败，不先下载整个 EgoDex，不先训练新 π0.5，也不先增加手机任务。

## 6. 右侧 G1 场景观察器

`work/g1-baseline` 的 `g1_scene_preview.py` 默认使用仓库固定子模块，经启动完整性检查后导入 `stack_scene`。它使用真实 MuJoCo model/data 与 Renderer，不加载 VLA、不导入硬件命令发布入口。

```sh
git submodule update --init third_party/g1-vla-control
git -C third_party/g1-vla-control submodule update --init third_party/mujoco_menagerie third_party/dex1_1_service
uv run --locked python g1_scene_preview.py --port 8778
terminal-browser open http://127.0.0.1:8778/ --split right --size 0.48
```

源子模块及上述两个资产必须已初始化。启动检查固定提交、已跟踪文件字节、contract SHA、资产版本及同进程模块来源；不悄悄退回主目录旧工程、其他资产、Panda 或录像。显式 `--g1-source` 也必须通过相同验证。

默认暂停。当前页面的“开始抬臂技能”执行第 11 节的人工技能；空闲物理预览也使用同一受检反馈保持控制器，不再直接维持固定关节指令。单步为一个 0.002 s 物理步；四个原相机保留。技能执行／停止期间禁止暂停、复位或手动单步，不能靠冻结画面冒充停稳；页面刷新频率不是 VLA 或真机控制频率。

旧观察器检查见第 10 节；当前技能页面实测见第 11 节。人工技能与 UI 检查不解锁任何神经任务或真机 Gate。

## 7. 后续阶段的准入条件

| 后续工作 | 必须先具备的证据 |
|---|---|
| 手机拍照仿真 | G1 原任务基线确认；再单独建立手机抓取、相机朝向与照片验收，缺少手机技能时明确采集/适配新数据 |
| 人的示范视频与 planner | 对应可调用技能已经完成；视频提取目标/步骤，不能把未知动作伪装成已有能力 |
| 速度方法比较 | 原任务或手机任务分别建立基线；固定 checkpoint、动作语义、初态和验收，仅按协议比较调度 |
| 真机只读 Shadow | 完整标定、官方 limits、通信新鲜度、watchdog、故障注入等相应 Gate 通过；仍不发布动作 |
| 真机动作 | 相应全部硬件 Gate 通过、操作者就位、支撑/E-stop 与独立运动授权；不能凭仿真通过自动执行 |

## 8. 明确禁止的捷径

- 手改历史 JSON 的通过标志，或删除失败、超时、拒绝记录。
- 用旧 horizon-50/30 Hz/crop 配置结果代替 author-32/15 Hz/pad 结果。
- 改 timestep、执行器强度、质量、摩擦、碰撞体或物体绑定来制造通过/加速。
- 为减少时间删掉握持、支撑、松手、接触或稳定条件。
- 把暂停诊断的时间表、录像时长或固定 fixture 当正式任务性能。
- 在真实 G1 上直接运行作者 `main_eef.py`、`main.py`、`replay.py` 或任何未审批 publisher。
- 将缺失许可的第三方运动控制代码直接复制进新仓库；已有部署审计指出 `g1-client` 当时未发现根 LICENSE，本阶段只引用固定来源并复现已核对的行为。

## 9. 固定来源

- [G1 源工程与冻结提交](https://github.com/youseihuayu-wonderful/g1-vla-control/tree/a5e40a25c5c080ae114a925dae94215c7cb6d793)
- [Policy contract](https://github.com/youseihuayu-wonderful/g1-vla-control/blob/a5e40a25c5c080ae114a925dae94215c7cb6d793/g1_policy_contract.yaml)
- [作者配置重验证报告](https://github.com/youseihuayu-wonderful/g1-vla-control/blob/a5e40a25c5c080ae114a925dae94215c7cb6d793/LGG100_AUTHOR32_REVALIDATION_CN.md)
- [Realtime 失败记录](https://github.com/youseihuayu-wonderful/g1-vla-control/blob/a5e40a25c5c080ae114a925dae94215c7cb6d793/results/lgg100_author32_closed_loop_t0_offset008_baseline_realtime.json)
- [Paused 失败记录](https://github.com/youseihuayu-wonderful/g1-vla-control/blob/a5e40a25c5c080ae114a925dae94215c7cb6d793/results/lgg100_author32_closed_loop_t0_offset008_baseline_paused.json)
- [首目标 IK 收敛诊断](https://github.com/youseihuayu-wonderful/g1-vla-control/blob/a5e40a25c5c080ae114a925dae94215c7cb6d793/results/lgg100_author32_online_ik_convergence_trace.json)
- [部署参考与 Publisher 风险审计](https://github.com/youseihuayu-wonderful/g1-vla-control/blob/a5e40a25c5c080ae114a925dae94215c7cb6d793/YUHAO_G1_CLIENT_DEPLOYMENT_AUDIT_CN.md)

## 10. 2026-11-12 执行记录：本地 E0 清单与 E1 首目标子门

### 已落地的入口和边界

- 独立实现分支：`work/g1-baseline`，起点为计划提交 `c57c99e8827a3847aeb7526b11f165b7b1f21ec4`。
- `g1_source.py` 激活仓库内固定源；`g1_baseline_manifest.py` 保存来源、应用文件 SHA、依赖、资产与输入可用性。只初始化 Menagerie 与 Dex1 资产，Unitree SDK 未初始化；原 G1 工程未修改。
- [本轮清单](results/g1_baseline_manifest.json)记录实际 Darwin/arm64、Python、MuJoCo、NumPy 与已安装依赖，以及本机原缓存的 100 个 Parquet 的 SHA。原数据是关节空间数据，不能当 checkpoint 的 EEF 输入或归一化统计。
- 清单与诊断拒绝覆盖既有文件、悬空链接或写入冻结源目录。该清单不是 checkpoint 严格恢复、完整 E0 输入齐备或 E4 任务通过。

### 首目标原因和候选修正

实际旧调用链为 `lgg100_quarantined_closed_loop.py` → `G1SweptPathPreflight(ik_iterations=args.online_ik_iterations)`；保存运行传入 30 次。保存 trace 中，此时右臂姿态仍超注册门槛，后续离线迭代能收敛且没有关节顶限，因此不能把这次 `unreachable` 等同于几何不可达。本轮读取旧证据，不为确认旧错误而重跑旧失败。

新项目的 `g1_preflight.py` 复用源工程已有 `G1FastSweptPathPreflight` 与有界、按残差提前停止的求解器。没有直接改成 250 次；位置/姿态内部停止条件更严格，关节限位、碰撞分类和扫掠插值门槛未放宽。`iteration_step_s` 是数值 IK 更新参数，不是 MuJoCo 物理步长或动作执行速度。

[首目标诊断](results/g1_first_target.json)核对重建初态的双臂 FK 残差、world 目标及方块位置，均与保存证据匹配。两个记录的目标不同，不当作实时/暂停配对实验。

| 保存目标 | 有界 IK 次数 | 最大位置误差 mm | 最大姿态误差 ° | 首目标扫掠检查 ms |
|---|---:|---:|---:|---:|
| realtime | 26 | 0.223 | 2.350 | 6.774 |
| paused | 25 | 0.323 | 2.355 | 7.040 |

两者在 `free_space` 下通过 IK、限位和扫掠碰撞检查，输入 MjData 不变、物理步数为零。表中是本机一次离线检查耗时，不含完整渲染、推理、传输与提交，不能证明 100 ms 新鲜度或任何任务提速。

**重放边界：** 本地没有完整 raw32 NPZ，也没有完整保存的 MjData。为满足 chunk schema，将同一个保存首目标保持两个采样点，明确标为 synthetic hold；不是恢复出的第二个动作、完整神经 chunk 或位级全物理状态回放。候选门目前只用于本项目诊断，原子模块的神经闭环仍使用旧入口。

### 本轮实际验证

```sh
uv run --locked python g1_baseline_manifest.py --output results/g1_baseline_manifest.json
uv run --locked python g1_first_target.py --output results/g1_first_target.json
uv run --locked python -m unittest discover -s tests -p 'test_g1_first_target.py' -v
```

这些路径已保存本轮证据；复跑须使用新的输出路径。

- 5 项定向回归通过：保存目标可达、危险远目标拒绝、真实方块/手指碰撞拒绝、拒绝覆盖旧证据与悬空链接、拒绝写入冻结源目录。全部使用实际 MuJoCo 模型；无假造判定。
- [原有开发正确性 corpus](results/g1_project_correctness_corpus.json)：用一次性构造器适配，将 fast 分支接到真正的 `g1_preflight.create_preflight`，核对插值参数不变，其余物理与判据不 mock。7 个判定样本、3 个异常输入检查通过，无危险放行。按源依赖约束，用 `uv run --locked --with pyarrow==19.0.1` 提供该 corpus 的额外导入依赖；未改变主项目锁文件。[实际命令和适配代码](results/g1_project_correctness_corpus_invocation.json)随证据保存。
- [右侧观察器实测](results/g1_observer_smoke.json)：默认仓库源启动，实际浏览器完成播放、暂停、复位与单物理步；四个相机均得到 640×480 图像、HTTP 200，无页面运行错误或横向溢出。最终暂停在 0.002 s，VLA 未连接、真机禁用。用户右侧画面保留。

### 仍未通过及下一步

1. **E0 完整模型输入仍阻塞：** 本地项目环境未安装 JAX/OpenPI，缺冻结 checkpoint 及配套统计、同步三相机样本和原始输出包；GPU 工作站 SSH 只读探测超时。先恢复授权计算环境与原包访问，核对 hash/schema/成套来源，不重新训练、不拿原关节统计替代。
2. **正式任务协议尚未冻结：** 已读旧 evaluator，它按红蓝黄几何关系、线速度和连续 3 次成功循环判定，循环次数不等于固定秒数稳定窗口；未验证支撑接触、红块居中与完整任务超时/失败分类，不能冒充新的物理任务验收。
3. **E2 未通过：** 旧记录的提交观测年龄已超预算；本轮没有实时神经调用，不能用离线检查时间补算成新鲜观测。
4. 取得配套原包后，继续完整 chunk/初态/相机/物理标定及动态跟踪检查，随后才接候选门到新的正式闭环。E0–E3 全部通过后再运行 E4。此处是 VLA 路线；第 11 节人工技能与第 12 节受限 planner 独立验证，不能替代 VLA 基线或提前开展速度、真机结论。

## 11. 人工技能支线：真实物理抬臂—保持—回参考位姿

### 范围与控制器

本支线让现有 G1 仿真先具备可调用的人工编码能力，不要求先恢复 GPU／VLA，但也不修饰原模型的失败。本节验收时尚未连接 planner；后续接入见第 12 节。VLA 模型权重未加载，真机接口禁用。

[起始控制探针](results/g1_scripted_skill_initial_control_probe.json)发现：只保持原关节 ctrl，物理沉降与自由基座漂移会造成夹爪／方块接触。因此不能直接把旧“播放”作为安全技能，也不能删除碰撞检查或移动方块来通过。

`g1_arm_skill.py` 的 `ArmRaiseSkill` 改用世界末端参考位姿反馈、有界重力位置补偿与速度阻尼，再经原 EEF／关节速度、加速度和 jerk 滤波约束写入 `data.ctrl`。运行中的 `qpos/qvel` 不被控制器赋值；真实状态仅由 `mj_step` 演化。独立 scratch MjData 的 FK／扫掠预检允许设置关节位置，但绝不当执行数据或演示画面。

模型保留自由基座，腿部／躯干保持原站姿伺服；没有焊住机器人、修改增益、质量、摩擦或碰撞体，没有直接注入力绕过执行器。右臂目标由参考姿态肩俯仰的受限弧线定义，左末端维持原世界参考位姿。**返回指原世界末端位置／姿态，不是漂移后的新 home，也不是强行恢复逐位相同的全身关节数组。**

### 事前验收条件与状态含义

- 抬升前：原末端误差不超过 5 mm／3°，且机器人平移自由度不超过 0.05 m/s、转动自由度不超过 0.05 rad/s，末端线速度不超过 0.01 m/s、角速度不超过 0.05 rad/s，连续满足 0.5 s；最少初始保持 2 s、最长初始窗口 10 s，超时失败而非强行开始。
- 抬升 3 s，实际高度至少 40 mm 并保持 1 s，再用 3 s 返回；返回后的相同位姿／低速条件连续满足 0.5 s。最终稳定等待有界，不删除保持窗口。
- 始终执行 `free_space` 接触分类、真实关节范围、有限性、位置／姿态跟踪及基座稳定检查；硬位置跟踪上界 50 mm、硬姿态跟踪上界 3°、骨盆高度至少 0.70 m。其余声明值完整保存在结果 `acceptance/config`。
- 用户停止：取消名义轨迹，继续真实物理与有界反馈，连续低速 0.5 s 后才报告 `stopped`；不要求完成回位，也不报告 `succeeded`。
- 安全故障进入停止／失败路径；`failed/rejected` 锁存且不再积分，必须显式复位。非有限状态不交给 MuJoCo 自动重置。**故障冻结不算已经证明物理停稳。** 成功／已停后的保持仍检查新故障，不能因技能终止而忽略新接触。

这些是本机仿真工程门槛，不是 Unitree 官方力矩／电流／硬件控制周期认证，也没有证明真机平衡或实时性。

### 实际运行结果

| 场景 | 终态 | 仿真时长 s | 实测验收 |
|---|---|---:|---|
| 完整抬臂／保持／回位 | `succeeded` | 16.266 | 初始稳定在 8.766 s 完成；最大抬升 64.139 mm；返回误差左 0.102 mm、右 0.185 mm；实际保持 1 s，返位低速持续 0.5 s |
| 运动中请求停止 | `stopped` | 11.498 | 10.500 s 请求停止；继续积分 0.998 s 后停稳，其中连续低速 0.5 s；取消回位，未冒充完成任务 |

[完整动作](results/g1_scripted_arm_raise.json)与[停止记录](results/g1_scripted_arm_stop.json)均无禁止接触；包含阶段历史、实测轨迹、滤波预算及停止结论。CLI 使用原子新建、拒绝覆盖／悬空链接、拒绝输出到冻结源目录。

```sh
uv run --locked python g1_arm_skill.py --output results/g1_scripted_arm_raise.json
uv run --locked python g1_arm_skill.py --output results/g1_scripted_arm_stop.json --stop-after-s 10.5
uv run --locked python -m unittest discover -s tests -p 'test_g1_arm_skill.py' -v
uv run --locked python g1_baseline_manifest.py --output results/g1_scripted_skill_manifest.json
```

以上是本轮已执行命令；这些证据路径已存在，复跑应选择新路径。9 项回归通过，覆盖真实积分／无瞬移、完整动作和原世界参考回位、原命令预算／模型不变、运动中停稳、启动碰撞／限位／运动状态拒绝、运行时接触与姿态故障、非有限故障锁存、证据输出保护。

### 右侧实际页面与录像

- 同一个 `ArmRaiseSkill` 驱动 `g1_scene_preview.py`；`POST /skill/start`、`POST /skill/stop`、`GET /skill/result` 提供真实状态和结果。禁止由模型随意传入关节目标或绕过检查。
- [浏览器实测](results/g1_scripted_arm_browser_smoke.json)完成完整动作、实际抬升中的停止、空闲播放／暂停／单步／复位和相机切换；活动时复位、暂停、单步、手动播放与重复启动均返回 409，停止期间也拒绝冻结。四路图像为 640×480、HTTP 200，无横向溢出。
- 浏览器日志保留了后端重启期间的连接拒绝，不删除失败记录；[按最终导航时间分类](results/g1_scripted_arm_browser_error_context.json)后，最终验收期间的错误数为零。[一次性异常恢复探针](results/g1_scripted_arm_fault_recovery.json)另验证控制器异常不会把页面卡死在不可复位状态；它是注入异常检查，不是物理停止证据。
- [实际页面录像](media/g1-scripted-arm-baseline.mp4)已检查为 H264 MP4，约 22.9 s 墙钟录屏，保留了原始页面与仿真时钟；不是逐帧设置姿态生成的动画。录屏时长、仿真时长、物理步长不能混作部署频率或速度优化证据。GitHub 展示录像，本地右侧页面才是可交互仿真。
- [本轮源码／环境清单](results/g1_scripted_skill_manifest.json)记录控制器、测试、界面和依赖文件 SHA；清单进程自身不执行机器人，其 execution 字段不能替代上面的动作运行记录。原神经 contract 与其执行禁令保持不变。

本节验收结束时，右侧复位、暂停，可重新运行独立人工技能。随后仅允许 planner 调用这项已验收能力并读取真实 `status/reason/metrics`，实际交付见第 12 节。未知抓取、手机拍照、人类示范操作仍不支持；此支线不替代原 VLA 故障定位，也不解锁真机动作。

## 12. 真实 planner 接入：人工技能与实测反馈

### 已实现的边界

`g1_planning.py` 复用现有 `planner.py` 的真实 `request_structured` 订阅传输，不调用 Panda 规划逻辑、不构造固定模型答案、不回退 API 密钥。实际使用已登录 ChatGPT 的官方 Codex CLI；本轮请求模型标识为 `gpt-5.6-terra`，不据此推断服务端权重版本。

唯一技能为 `g1_right_arm_raise_hold_return`，参数必须是空对象。它是第 11 节原控制器的一次固定宏动作，内部初始稳定、抬升、保持、返回、稳定阶段均由人工编码；没有声称模型自行拆解了这些阶段。控制器、冻结 G1 子模块和物理判据未改动。

- 初次真实模型回复只能是完整支持的一次技能提案，或明确 `unsupported`。手机、拍照、保持五秒、左臂、重复动作等均不在能力目录内；不能只做请求中支持的部分。
- 提案展示绑定的原始任务、真实 JSON 与整数编号。批准时重新核对编号、场景 revision 和真实状态指纹；布尔／旧编号、漂移和冲突操作不得启动执行。一次会话最多调用一次技能。
- 规划、待审批和结果评估期间**有意暂停仿真时钟**。这是仿真演示边界，不是支持真机等待，也不证明 100 ms 神经观测新鲜度达标。
- 获批后只调用原 `ArmRaiseSkill.start()`，再以原执行器与 `mj_step` 推进真实物理。不向模型开放电机、关节、速度或力参数。
- 终态后把真实 `status/reason/metrics`、停止结论和执行异常反馈给模型。只有本地 `succeeded` 且实际模型反馈一致，整体 `success` 才为真。模型不能把 `failed/stopped/rejected` 升格为成功；模型不可用或反馈不一致也不会伪装成通过。
- 执行中取消会继续原物理停止验证，再请求真实结果反馈；不是冻结画面。低速停稳不证明碰撞已解除；即便模型原文说“安全停止”，也不是硬件安全认证。
- 原技能报告中的 `planner_connected=false` 描述不直接调用模型的底层 primitive；规划会话／页面顶层仅在真实线程证据出现后标记 planner 已连接。VLA、policy 和真机始终禁用，输入仅为结构化状态，不含相机或人的示范视频。

### 独立真实运行

下表均实际调用订阅模型；物理执行使用真实 G1 MuJoCo，不使用测试替身。每行是单次验证，不拼成成功率或速度对照。

| 请求／检查 | 本地终态 | 实际模型反馈 | 技能调用次数 | 证据 |
|---|---|---|---:|---|
| 右臂抬起、保持一秒、返回 | `succeeded` | `succeeded` | 1 | [完整记录](results/g1_planner_success.json) |
| 抬臂后拿手机拍照再返回 | `unsupported` | 未执行，无执行评估 | 0 | [完整拒绝](results/g1_planner_unsupported.json) |
| 10.500 仿真秒主动停止 | `stopped` | `stopped` | 1 | [真实停止](results/g1_planner_stopped.json) |
| 人工注入红块／左指真实碰撞 | `failed` | `failed` | 1 | [失败反馈](results/g1_planner_collision_failure.json)／[实际一次性驱动代码](results/g1_planner_collision_invocation.json) |

正向执行为 8,133 个真实物理步、16.266 仿真秒，最大右末端抬升 64.139 mm，最终原世界坐标返回误差左 0.102 mm、右 0.185 mm。停止试验在 11.498 仿真秒完成实测停稳，整体成功为假。手机复合请求的物理步数为零，前后状态指纹相同。碰撞负例只在检查夹具中移动红块以制造真实接触，不是正常执行办法；原碰撞拒绝保持有效，局部失败未被模型反馈掩盖。

```sh
uv run --locked python -m unittest discover -s tests -p 'test_g1_planning.py' -v
uv run --locked python run_g1_planning.py --execute --output results/g1_planner_success.json
uv run --locked python run_g1_planning.py --task '请抬起右臂，然后拿起手机拍照，再回到原位。' --output results/g1_planner_unsupported.json
uv run --locked python run_g1_planning.py --execute --stop-after-s 10.5 --output results/g1_planner_stopped.json
```

以上命令本轮已执行，输出路径已存在，复现必须选新路径。runner 默认只生成真实提案，只有显式 `--execute` 才批准本次实际决策；证据原子新建，拒绝覆盖、悬空链接与冻结源目录。停止负例的 `scenario_passed` 只表示预期停止路径验证通过，不等于任务 `success`。

27 项定向契约测试通过，覆盖旧／布尔／重复审批、完整拒绝、非法参数、状态变化、真实结果不可被模型升格、执行异常、停止等待、取消迟到回复和清理。协议测试中的受控 provider／executor 替身不冒充真实神经或物理证据；实际观察器检查使用真实 MuJoCo。第 11 节的 9 项物理回归是此前单独记录，不重复计入本轮测试数。

### 右侧页面实测与录像

- `POST /planner/start` 请求真实提案，`POST /planner/execute` 只接受当前整数 `decision_id`，`POST /planner/cancel` 取消规划或进行物理停止；`GET /planner/report` 导出完整实际调用与执行证据。独立手动技能入口保留，但规划锁定时拒绝使用，绝不自动回退。
- [实际页面完整动作](results/g1_planner_ui_approval.json)：规划与审批前仿真时间保持 0、没有技能调用；批准后只有一次物理执行。引擎已经成功但模型反馈尚未返回时，整体成功仍为假；真实反馈一致后才显示整体成功。
- 旧／布尔／重复审批返回 400，冲突手动启动、复位、暂停返回 409。执行中非法 JSON 返回 400、超大请求返回 413，真实物理仍继续，未被无效请求冻结或写入运行错误。
- [完整拒绝五秒保持](results/g1_planner_ui_guards.json)：模型明确指出固定技能只保持一秒，物理时间为 0、调用次数为 0、批准按钮禁用。该记录也保留了自动化 `waitFor` 默认两秒快速失败的原始异常，当时仍未审批、没有运动。
- [随后完成的同一停止会话](results/g1_planner_ui_stop.json)：继续同一个未审批提案，没有重新调用规划；在实际抬升中按页面取消，先观察到 `stopping` 且物理仍活动，待实测低速稳定后才是 `stopped`。真实模型也报告停止而非成功，原始反馈已在页面显示。前述两秒未审批片段仅留元数据，不当作停止录像。
- 538×656 的实际右侧窗口无横向溢出；批准执行时自动折叠输入／提案并回到页首，机器人画面完整可见。当前页面保留本地实测终态与模型原文的明确区分。
- [审批及物理执行 MP4](media/g1-planner-approved-skill.mp4)约 33.5 墙钟秒，完整模型反馈另见对应 JSON；[同一提案的物理停止与反馈 MP4](media/g1-planner-physical-stop.mp4)约 24.9 墙钟秒，从未审批提案开始。均为实际浏览器 H264 录屏，保留页面和仿真时钟，不拼接为任务耗时，也不是运行时录像回放。
- [当前源码／环境清单](results/g1_planner_manifest.json)另记录 planner 传输、会话、runner、物理技能、页面与测试的 hash。清单进程自身不调用模型、不推进机器人，实际执行以本节独立证据为准。

当前交付是**自然语言选择一个已验证人工宏动作、人工审批、物理执行、真实结果反馈**。尚无通用多技能规划、视频理解、手机拍照、VLA 动作执行或真机验证；第 10 节缺失输入与原 VLA 拒绝／实时预算问题仍独立保留。

最终界面部署后另生成了[真实待审批交付提案](results/g1_planner_handoff.json)：绑定原始任务的文字、实际模型 JSON 和提案 ID 均已在右侧显示；记录时仿真时间为 0、技能调用为 0，只有一次真实规划请求，尚未授权运动。用户核对后可点击“批准当前提案并执行”；执行时界面自动折叠提案展示机器人。
