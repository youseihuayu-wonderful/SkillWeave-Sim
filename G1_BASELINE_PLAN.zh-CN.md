# SkillWeave Sim：G1 VLA 基线第一阶段执行计划

## 0. 本阶段只解决一个问题

**让冻结的 G1 VLA 在真实 MuJoCo 物理与完整拒绝检查下，完成它已经训练过的叠方块任务，先建立 Adaptive-OFF 基线。**

这里的“真实 MuJoCo”指本地物理仿真计算，不是真机。本轮交付为：已有证据核对、可操作的 G1 场景观察器、本文计划；本轮没有重新执行 VLA 闭环，没有修复控制逻辑，没有向真机发送动作。

本阶段不增加手机、不训练新模型、不接人类视频、不进行速度优化或真机运动。每个后续阶段必须以上一阶段的证据为输入，而不是以网页能显示机器人为通过条件。

## 1. 项目与分支边界

- 新项目：`youseihuayu-wonderful/SkillWeave-Sim`。
- `main` 和 `work/skillweave-sim` 保留已验证的 **Panda/MuJoCo** 参考项目，不将其结果改称 G1 结果。
- `plan/g1-baseline` 保存本计划和 G1 场景观察工具。它不是已经迁移完成的 G1 VLA 运行版本。
- 后续实现使用独立的 `work/g1-baseline` 分支，接入经过来源核对的 G1 模块；不覆盖原 G1 仓库，不混用 Panda 与 G1 的机器人接口。
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

以下产物是接下来执行时应生成的文件，不表示这些实验已经完成。

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

**当前可用性边界：** 本机已找到 100 条缓存 Parquet；未在本地找到前述 author-32 的多模态 NPZ/输出包。旧报告指向 `/home/user1/workspace/shihua/results`，这是历史路径，本轮没有连接或确认该主机。下次执行须先确认授权环境与文件可用性，不重复大规模训练。

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

本分支提供 `g1_scene_preview.py`。它导入指定 G1 源工程的 `stack_scene`，使用真实 MuJoCo model/data 与 Renderer，不加载 VLA、不导入硬件命令发布入口。

```sh
uv run --locked python g1_scene_preview.py \
  --g1-source "$HOME/g1-vla-control" --port 8778
terminal-browser open http://127.0.0.1:8778/ --split right --size 0.48
```

源工程和其 MuJoCo/Dex1 资产必须已经存在；本分支没有通过这个观察器完成全部 G1 运行时迁移。观察器缺少源文件或发生物理/渲染异常时明确报错，不换成 Panda 或录像。

默认暂停。播放只推进原场景的参考站姿控制；单步为一个 0.002 s 物理步；可复位，并切换外部、头部、左腕、右腕视角。观察器页面刷新频率不是正式 VLA 部署频率。

本轮已在实际浏览器验证画面、播放/暂停、单步、复位与四个视角；四路图像请求均返回 200，观测状态中 `policy_connected=false`、`hardware_execution_enabled=false`，检查时页面无运行错误。这个 UI 检查不解锁任何任务或真机 Gate。

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
