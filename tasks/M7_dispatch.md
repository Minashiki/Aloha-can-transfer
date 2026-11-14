# M7 派发：DSRL（扩散策略噪声空间 SAC）

**先读这份交接单：`tasks/handover_M7_1.md`。**

项目根目录：`/home/minashiki/AIControl/FinalHomework`。背景见 `docs/PLAN.md` 第 8、9 节和第 10 节中的 M7（只读）。M6 接口与结论见 `tasks/M6_delivery_r2.md`、`tasks/M6_review_r3.md`。

**BC 基线（已有，只读）**：正式 checkpoint `outputs/m6/train/ckpt_last.pt`；种子 0–49 正式评测 31/50 = 0.62（Wilson 95% [0.48, 0.74]），失败全部在抓取阶段（`not_grasped` 15、`not_lifted` 4），结果在 `outputs/m6/train/eval/eval.json`。该评测的初始噪声未按种子固定（详见交接单第 3 节）。

## 硬性约束（违反任何一条即判为不合格）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`），禁止系统 Python、base 环境、新建环境。**不得安装任何包**；如果认为某个包不可缺，停下来在交付记录中写明理由，作为存疑项上报。命令写出环境前缀。
2. 不得修改：`docs/`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、M2 计算模块、M1 / M3 / M4 / M5 / M6 的配置文件、`src/aloha_pp/data/retarget.py`、`outputs/m5/dataset/` 下的数据集、M6 的扩散核心（`policy/diffusion.py`、`unet1d.py`、`vit.py`、`modules.py`）。可以给 `DiffusionPolicy` **增加**只读的方法（例如提取观测特征），但不得改变 `sample` 的行为；M1–M6 已有行为保持不变（见第 6 节回归）。
3. 不得修改增益、力矩上限、速度上限、成功判定阈值、`random_can`、夹爪开合量。
4. 不得把 DSRL / DPPO 官方仓库作为依赖安装或在运行时 import。源码可以 clone 到 `third_party/` 阅读（参考超参数与实现细节），本仓库中的实现自己写；若逐段复制了代码，文件头注明来源、commit、许可证，并把许可证全文加入 `THIRD_PARTY_LICENSES`。
5. 不得写 `outputs/m3/`–`outputs/m6/`；M7 输出写 `outputs/m7/`，回归写 `outputs/m7/regression/`。`m6_eval.py` 不带 `--out` 时默认写到 checkpoint 同目录的 `eval/`，对正式 checkpoint 运行会覆盖 BC 正式评测结果，所以凡调用 `m6_eval.py` 一律显式 `--out outputs/m7/...`。
6. 不得执行 `git commit` / `push` / `config` / `add`，不得添加 co-author，不得启动其他 agent。
7. **只做冒烟训练**（几分钟内）。正式训练不由你运行：给出命令、配置、实测吞吐和总时长估计，由用户启动。

## 要做的事

### 1. DSRL 核心（`src/aloha_pp/rl/`）

按 PLAN 第 8 节与 DSRL（Wagenmaker et al., 2025, “Steering Your Diffusion Policy with Latent Space Reinforcement Learning”）实现主变体：**冻结 M6 的扩散策略，SAC 在初始噪声空间学习**。

- **潜动作**：actor 输出初始噪声 w，形状为 `DiffusionPolicy.noise_shape = (16, 8)`（展平 128 维），经 tanh 再乘以幅度上限 `noise_scale`（配置项，取值依据写明）。环境动作 = `policy.sample(obs, noise=w)` 的动作块，执行前 8 步（第 8 步预览第 9 步，沿用 M6 的 `ChunkExecutor` 规则）。**一个动作块是 RL 的一步。**
- **actor 的输入（不得含物体真值）**：冻结的 M6 观测编码（ViT + SpatialEmb 的输出，eval 模式、无增强）与归一化后的本体状态（2 帧 `tcp_pose` + `gripper_opening`，与 M6 相同的 `pose_state` 规则）。给 `DiffusionPolicy` 增加一个只读方法返回这个特征；测试证明它不改变 `sample` 的输出。
- **非对称 critic**（PLAN 已确认）：critic 输入 = 特权状态（Can 位姿与线速度、TCP 位姿、夹爪开度、关节角 / 角速度，具体字段写明）+ 潜动作 w。可以额外拼接 actor 的观测特征，写明。双 Q、目标网络软更新、熵系数自动调节；UTD、网络尺寸、学习率、γ 等超参数参考 DSRL 官方实现写进 `configs/m7_dsrl.yaml`，写明出处。
- **初始化**：可选“先用 w ~ N(0, I) 采集若干步填充回放池”（即 BC 本身的行为），步数写进配置。
- 可选变体 DSRL-NA（噪声混叠）**不做**；在交付中说明若主变体学不动可以考虑它。

### 2. 环境包装（`src/aloha_pp/rl/`）

- 每个 RL 步执行 8 个控制拍；奖励默认**稀疏**：成功判定通过的那一步奖励 1 并终止（terminal）；超时（默认 25 s，与 M6 评测相同）为截断（truncated，不作为终止状态做 bootstrap）。
- **分阶段奖励**作为配置开关（默认关闭；PLAN：“DSRL 学不动时加分阶段奖励”）：用特权信息定义（例如双指接触、Can 提起、到达目标格上方、松手入格），每段奖励与条件写明。BC 的失败全部在抓取阶段，分阶段奖励至少要能区分“未抓住”“抓住未提起”“已提起”；每个阶段奖励每个 episode 最多给一次，不能靠反复进出某阶段刷奖励（测试覆盖）。
- 环境组装用 `src/aloha_pp/sim/c3.py` 的共享函数，评测与 M6 完全相同（含沉降后的 `controller.reset`）。
- **并行环境**：支持 N 个并行环境（子进程，各自 EGL 渲染），策略对 N 个观测批量推理；N 为配置项。实测单环境与 N = 4、8 时每秒 RL 步数，给出推荐值。
- **训练 / 验证 / 评测种子分离**：训练 episode 的初始条件种子从 10000 起；训练中验证用另一组固定种子（默认 20000 起，配置项）；最终评测固定用 `random` 模式种子 0–49（与 M4 E2、M6 评测相同）。三组两两不重叠。M6 的训练中验证用了种子 0–3（与评测重叠），M7 不沿用。

### 3. 训练脚本（`scripts/m7_train.py`、`configs/m7_dsrl.yaml`）

- 输入：M6 BC checkpoint 路径（默认 `outputs/m6/train/ckpt_last.pt`，只读加载，不写其所在目录）。
- 回放池、checkpoint（actor / critic / 目标网络 / 熵系数 / 优化器 / 步数 / BC checkpoint 路径与哈希 / git commit 与 dirty）、断点续训（回放池可选择是否保存，写明）、日志（回报、成功率、critic / actor 损失、熵系数、每秒步数）。
- **训练中验证**：每隔若干环境步在**验证种子**上跑 N 个 episode（actor 取确定性均值噪声；N 默认 16，用并行环境跑，配置项），记录成功率与 Wilson 区间，并**至少录一条 episode 视频**（≤ 2 MB），日志中给出路径（PLAN 第 9 节）。M6 每次只验证 4 条，后期 3/4–4/4 而正式只有 0.62，所以 N 不要再取 4 这种量级；N 与验证间隔对总训练时间的影响写进时长估计。
- 不得用评测种子 0–49 挑选 checkpoint。最终参与对比的 checkpoint 只能是 `ckpt_last` 或按验证种子成功率选出的那个，选法写进配置和日志。
- `--live-viewer` 开关，默认关闭。

### 4. 评测与对比（`scripts/m7_eval.py`）

- 在种子 0–49 上评测 DSRL actor（确定性），输出格式沿用 M6 的 `aloha_pp.eval.v1`，`method = bc_dsrl`。
- `m7_eval.py` 同时支持两种 BC 噪声对照，同样用冻结的 M6 checkpoint、同样的环境与执行规则，只换初始噪声：
  - `method = bc_zero`：每个动作块 w = 0（DSRL actor 的 tanh 输出在 0 附近时与之对应，是“未学习”的确定性起点）；
  - `method = bc_seeded`：w ~ N(0, I)，但用按评测种子构造的独立随机数发生器生成，同一种子重跑逐位相同（不改 M6 `sample(noise=None)` 的行为，在 M7 侧生成噪声后传入）。
- 对比脚本：读取 M6 BC 正式评测（`outputs/m6/train/eval/eval.json`，只读）、`bc_zero`、`bc_seeded` 与 DSRL 的评测结果，输出成功率（Wilson 95% 区间）、主失败原因分布、IK 拍数、放置误差、episode 时长的对比表；另给出逐种子的成功 / 失败转移（例如 BC 失败而 DSRL 成功的种子列表）。同样格式也能放入 M4 规划器 E2 结果（规划器作为参照）。
- 至少 1 条成功和 1 条失败视频（若存在）。

### 5. 冒烟测试与时间估计

- 冒烟：用正式 BC checkpoint `outputs/m6/train/ckpt_last.pt`（只读）跑几分钟（并行环境 + 少量更新），证明采集、回放、SAC 更新、训练中验证、checkpoint / 续训、`m7_eval` 与对比脚本都能跑通。用正式 checkpoint 是为了让成功（奖励 1 + terminal）路径在冒烟中真的出现；冒烟中 DSRL 的成功率本身不作为验收项。若冒烟时间内一次成功都没出现，如实写明。
- `m7_eval` 的三种方法（`bc_dsrl` 用冒烟 actor、`bc_zero`、`bc_seeded`）在冒烟中只跑种子 0–4；种子 0–49 的正式评测命令写进交付，由用户启动。
- 用实测吞吐给出正式训练的推荐环境步数、并行数和预计总时长（含训练中验证的开销）；说明依据（DSRL 论文 / 官方代码在 robomimic 图像任务上的样本量）。截止约 2026-11-25，M8 需留 2 天，正式训练的墙钟时间按不超过约 24 h 设计；做不到时给出可在该时间内完成的缩减配置并说明代价。

### 6. 测试与回归

- 潜动作：tanh 幅度上限；同一 obs + 同一 w 经 `sample` 得到的动作块逐位相同（复用 M6 的确定性）。
- actor 输入不含特权字段（按维度和字段名检查）；critic 输入含特权字段。
- 环境包装：一个 RL 步 = 8 个控制拍、第 8 步预览第 9 步；成功时 terminal、超时时 truncated；分阶段奖励关闭时奖励只有 0 / 1。
- 并行环境与单环境在同一种子、同一潜动作序列下的低维轨迹一致（图像按 M6 的像素差 ≤ 1 口径）。
- SAC：在构造的小问题上 critic 损失下降；目标网络软更新；熵系数更新方向正确。
- 回放池存取、checkpoint 往返。
- 训练 / 验证 / 评测种子集合两两不相交。
- `bc_seeded`：同一评测种子重跑，噪声序列与结果逐位相同；`bc_zero` 每个块传入的 w 全为 0。
- 分阶段奖励：每阶段每 episode 至多一次；关闭时奖励只有 0 / 1。
- **回归**：`pytest tests -q` 全部通过（M0–M6 原有测试不得删改）；`check_env.py` 退出码 0；`m1_grasp_smoke.py` 输出 `GRASP_OK`；M4 E1 用 `m4_run.py --e1 --out outputs/m7/regression/m4_e1`，与 `outputs/m4/summary.csv` E1 行共有指标逐项一致；M6 冒烟 checkpoint 在种子 0–2 上用 `m6_eval.py --out outputs/m7/regression/m6_eval` 评测，结果与 `outputs/m6/smoke_r2/eval/` 一致（成功标志、主原因、IK 拍数）。

## 交付物

1. 上述代码、配置、脚本、测试；`outputs/m7/` 下的冒烟 checkpoint、日志、评测与视频（outputs 不进 git，交付记录中列出文件名和大小）。
2. `tasks/M7_delivery_r1.md`，包含：
   - 改动文件清单；运行过的命令及输出摘要；
   - DSRL 设计：潜动作与幅度上限、actor / critic 输入字段、网络结构、SAC 超参数及出处、初始化、奖励与终止 / 截断；
   - 分阶段奖励的定义（默认关闭）；
   - 并行环境的实现与实测吞吐（1、4、8 个环境）；
   - 训练 / 验证 / 评测种子分离；训练中验证的 N、间隔与 checkpoint 选法；
   - 冒烟结果与正式训练的推荐配置、启动命令、预计时长；`bc_zero`、`bc_seeded`、`bc_dsrl` 正式评测（种子 0–49）的启动命令；
   - 对比脚本的输出示例（M6 BC 正式评测 + 冒烟中种子 0–4 的 `bc_zero` / `bc_seeded` / `bc_dsrl`）；
   - 未完成或存疑的点。

## 完成标准

- 冒烟训练中采集、SAC 更新、训练中验证（含视频）、checkpoint 续训、`m7_eval` 与对比脚本全部跑通。
- actor 不使用物体真值，critic 使用特权状态；测试通过。
- 测试与回归全部通过。
- 做不到的项不得绕过或降低标准；如实写明，并作为失败上报。
