# M6 派发：Diffusion Policy（图像，BC）

**先读这份交接单：`tasks/handover_M6_1.md`。**

项目根目录：`/home/minashiki/AIControl/FinalHomework`。背景见 `docs/PLAN.md` 第 8、9 节和第 10 节中的 M6（只读）。数据集说明见 `src/aloha_pp/data/README.md`，M5 结论见 `tasks/M5_review_r4.md`。

## 硬性约束（违反任何一条即判为不合格）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`），禁止系统 Python、base 环境、新建环境。**不得安装任何包**（环境里没有 einops、torchvision、diffusers、hydra，代码中不得依赖它们；需要的张量变形用 torch 原生写）。如果你认为某个包确实不可缺，停下来在交付记录中写明理由，作为存疑项上报，不要自行安装。命令写出环境前缀。
2. 不得修改：`docs/`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、M2 计算模块、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`、`configs/m4_task.yaml`、`configs/m5_*.yaml`、`src/aloha_pp/data/retarget.py`、`outputs/m5/dataset/` 下的数据集文件。M1–M5 已有行为保持不变（见第 6 节回归）。
3. 不得修改增益、力矩上限、速度上限、成功判定阈值、`random_can`、夹爪开合量。
4. **不得把 DPPO / DSRL 官方仓库作为依赖包安装或在运行时 import。** 源码只能 `git clone` 到 `third_party/`（已被 `.gitignore` 排除）阅读；需要的部分**复制**进 `src/aloha_pp/policy/`，每个文件头注明来源仓库、commit、原路径和许可证，并把许可证全文放进 `src/aloha_pp/policy/THIRD_PARTY_LICENSES`（或同等文件）。
5. 不得写 `outputs/m3/`、`outputs/m4/`、`outputs/m5/`；M6 输出写 `outputs/m6/`，回归写 `outputs/m6/regression/`。
6. 不得执行 `git commit` / `push` / `config` / `add`，不得添加 co-author，不得启动其他 agent。
7. **只做冒烟训练**（几分钟内）。正式训练不由你运行：给出命令、配置和实测的单步 / 单 epoch 耗时与总时长估计，由用户启动。

## 要做的事

### 1. 引入扩散模型代码（`src/aloha_pp/policy/`）

- 从 DPPO 官方仓库（`irom-princeton/dppo`，先确认地址与许可证）取出：扩散模型（训练损失、噪声调度）、条件 UNet1D、DDIM 采样，以及图像编码器（优先用 DPPO 自带的 ViT 编码器与随机平移增强，避免依赖 torchvision）。只取需要的部分，去掉对 hydra / einops / wandb 的依赖。
- 记录 clone 的 commit；交付记录列出每个复制文件的原路径与改动摘要。

### 2. 策略（PLAN 第 8 节）

- **观测**：两路图像 `agentview`、`wrist_cam_left`（96×96）+ 本体状态 `tcp_pose`（7）+ `gripper_opening`（1）。**不含物体真值，也不含关节角**。观测历史帧数（1 或 2）自定并写明。
- **动作**：数据集中的 8 维动作（base 系绝对 TCP 位姿 xyz + 四元数 wxyz + 夹爪开度）。归一化方法写明；四元数在送入环境前归一化，并检查数据中四元数的符号一致性（q 与 −q），写明处理。
- **动作块**：预测 16 步、执行 8 步（20 Hz，约 0.4 s）。
- **采样**：DDIM 5–10 步（配置项）。**DDIM 的 η = 0，给定初始噪声时采样必须是确定性的**——M7 的 DSRL 会在初始噪声空间做 SAC，所以必须提供接口 `sample(obs, noise=None)`：`noise` 为 None 时内部采样标准正态，传入时直接作为初始噪声；并公开噪声的形状。测试证明同一 `obs` + 同一 `noise` 两次输出逐位相同。
- **执行与 `next_action`**（M4 / M5 约定）：执行动作块的第 k 步时，`next_action` 取块内第 k+1 步；执行段最后一步（第 8 步）取块内第 9 步。环境用 C3 + preview 参考，和 M5 生成数据时完全相同（motor 模型、C3 增益、IK 设置）。这个环境 / 控制器组装请放到 `src/` 下一个共享函数里（如果需要从 `scripts/m4_run.py` 挪出，`m4_run.py` 改为调用它，行为不变）。

### 3. 训练（`scripts/m6_train.py`、`configs/m6_dp.yaml`）

- 数据：`outputs/m5/dataset/demos.hdf5`（120 条）。**BC 只用其中一部分**（PLAN：20–50 条，给 RL 留提升空间）：`n_train_demos` 写成配置项，默认 **30**，用固定种子从 120 条中抽取，抽到的 `source_id` 列表写进 checkpoint 和日志；其余示范可用于验证损失。
- 序列窗口在 episode 首尾的填充方式（重复首帧观测 / 末帧动作）写明。
- checkpoint（含模型、EMA 若使用、归一化统计、配置、训练示范列表、git commit 与 dirty 标记）、断点续训、日志（loss 曲线、学习率）。
- **训练中验证**（PLAN 第 9 节）：每隔若干 epoch 跑 N 个评测 episode，记录成功率，并**至少录一条 episode 视频**（.mp4，≤ 2 MB），日志中给出路径。
- `--live-viewer` 开关，**默认关闭**；打开时在验证 episode 中弹出 MuJoCo viewer 实时显示（无显示环境下不要求测试）。

### 4. 评测（`scripts/m6_eval.py`）

- 读 checkpoint，在 `random` 模式、**种子 0–49**（与 M4 E2 规划器的 50 个初始条件相同，便于直接对比）上评测；每个 episode 时限写成配置项（M5 示范最长 15.5 s，默认建议 25 s，写明）。
- 输出：成功率与 Wilson 95% 区间、失败原因分布（沿用 M4 / M5 的分类，按物理步判定）、放置误差、episode 时长、每个动作块的推理耗时（中位 / 最大）；至少 1 条成功和 1 条失败视频（若存在）。
- 评测日志的格式要能被 M7 复用（M7 将对比 BC 与 BC+DSRL）。

### 5. 冒烟测试与时间估计

- 冒烟训练：几分钟内，loss 明显下降（给出前后数值）；用冒烟 checkpoint 跑通训练中验证和 `m6_eval.py`（少量 episode 即可），产出成功率和视频。冒烟模型成功率可以为 0，不是验收标准。
- 用实测的训练吞吐给出正式训练的推荐配置（epoch 数或步数、batch size、学习率、DDIM 步数）和预计总时长；说明依据（例如 DPPO / Diffusion Policy 在 robomimic Can 图像任务上的设置）。

### 6. 测试与回归

- 噪声调度与 DDIM：η = 0 时给定噪声确定性；DDIM 步数改变时输出形状不变；训练损失在构造的小数据上可以下降（过拟合测试）。
- 网络：UNet1D、编码器输入输出形状；不同 batch 大小。
- 数据：窗口切片与首尾填充；归一化往返；训练子集抽取可复现。
- 执行器：动作块执行与 `next_action` 规则（第 8 步用第 9 步）；四元数归一化。
- checkpoint 保存 / 读取往返后输出逐位一致。
- **回归**：`pytest tests -q` 全部通过（M0–M5 原有测试不得删改）；`check_env.py` 退出码 0；`m1_grasp_smoke.py` 输出 `GRASP_OK`；M4 E1 用 `m4_run.py --e1 --out outputs/m6/regression/m4_e1`，与 `outputs/m4/summary.csv` E1 行共有指标逐项一致；用共享的环境组装函数重放 M5 数据集中 2 条示范，必须再次成功且与存储观测一致（证明评测环境与数据生成环境相同）。

## 交付物

1. 上述代码、配置、脚本、测试、第三方许可证文件；`outputs/m6/` 下的冒烟 checkpoint、日志、评测结果与视频（outputs 不进 git，交付记录中列出文件名和大小）。
2. `tasks/M6_delivery_r1.md`，包含：
   - 改动文件清单；运行过的命令及输出摘要（冒烟训练、训练中验证、评测、测试与回归）；
   - DPPO 来源（URL、commit、许可证）与每个复制文件的原路径和改动；
   - 策略设计：观测、历史帧、动作表示与归一化、四元数处理、网络结构与参数量、噪声调度、DDIM 步数、数据增强、窗口填充；
   - 噪声接口说明（形状、确定性测试结果），供 M7 使用；
   - 训练子集（种子、条数、`source_id` 列表）；
   - 冒烟训练 loss 前后数值、评测输出；推理耗时；
   - 正式训练的推荐配置、启动命令、预计时长；
   - 未完成或存疑的点。

## 完成标准

- 冒烟训练 loss 明显下降；训练中验证与 `m6_eval.py` 能产出成功率、失败分布和视频。
- `sample(obs, noise)` 在 η = 0 下确定性，测试通过。
- 测试与回归全部通过；评测环境与数据生成环境一致（重放 2 条示范再次成功）。
- 做不到的项不得绕过或降低标准；如实写明，并作为失败上报。
