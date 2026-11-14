# M7 派发 r2：按审查结论修改

**先读这份交接单：`tasks/handover_M7_3.md`。**

项目根目录：`/home/minashiki/AIControl/FinalHomework`。修改依据：`tasks/M7_review_r2.md` 第 3 节（已审批）。背景：`tasks/M7_dispatch.md`、`tasks/M7_delivery_r1.md`、`tasks/M7_review_r1.md`、`tasks/M7_recheck_r1.md`。

## 硬性约束（违反任何一条即判为不合格）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`），禁止系统 Python、base 环境、新建环境。**不得安装任何包**。命令写出环境前缀。
2. `tasks/M7_dispatch.md` 的约束 2–6 全部继续有效：不改 `docs/`、`.gitignore`、`environment.yml`、`aloha/`、`assets/scene/`、M1–M6 配置、M6 扩散核心，不改变 `DiffusionPolicy.sample` 的行为；不写 `outputs/m3/`–`outputs/m6/`；不做 git 写操作，不加 co-author，不启动其他 agent；不把 `third_party/` 作为依赖。
3. 只做冒烟、回归和短程计时；**正式训练和种子 0–49 的正式评测不由你运行**，只给出命令。
4. 只改下面清单列出的内容；第一版的其余设计保持不变。

## 修改清单

### 必做

1. **评测录像隔离**：`scripts/m7_eval.py` 增加 `--videos {none,auto,all}`，默认 `none`。`rl/evaluate.py` 的 `evaluate_latent` 在 `none` 时不调用任何录像渲染（现在 `index == 0` 永远录像）。`eval.json` 记录请求的录像模式和实际录像的种子列表。测试：`none` 模式下录像渲染的调用次数为 0。
2. **跨进程确定性回归**（代替第一版 spec 中 M6 冒烟评测的逐位对比）：两个独立进程，各用 `m7_eval.py --videos none` 在正式 BC checkpoint（`outputs/m6/train/ckpt_last.pt`，只读）上跑 `bc_zero`、`bc_seeded`，以及冒烟 actor（`outputs/m7/smoke/train/ckpt_last.pt`）的 `bc_dsrl`，种子 0–2，输出到 `outputs/m7/regression/det_a/<method>` 和 `det_b/<method>`。两次的 `episodes.csv` 中成功标志、主原因、IK 拍数、`place_xy`、时长逐项相同。写一个比对脚本或给出命令，在交付中列出比对结果，以及 torch / CUDA / 驱动版本和 CPU 线程数。
3. **验证录像隔离**：训练中验证的 16 条 episode 一律不录像；每次验证另跑 1 条录像 episode（种子 `val_seed_start`，单独的环境），只出视频，不计入成功率，不参与选模。日志中给出视频路径。
4. **best / last 都评测**：交付中的正式评测命令包含两条 `bc_dsrl`，`--actor-checkpoint` 分别为 `ckpt_best.pt` 和 `ckpt_last.pt`，输出目录分开。`m7_compare.py` 接受多个 DSRL 结果并分别标注（例如 `bc_dsrl_best`、`bc_dsrl_last`）；`selected.json` 指向的那个在表中标为“主结果（事先声明）”。
5. **对比口径**：
   - 表中写明每种方法的噪声来源（`bc`：M6 正式评测，随机且未固定；`bc_zero`：w = 0；`bc_seeded`：按种子固定的 N(0, I)；`bc_dsrl_*`：确定性 actor），以及是否含录像 episode。M6 正式评测一行标为“随机噪声（未固定）、种子 0–3 录像，仅作参考”。
   - 配对比较（`bc_seeded` vs 各 `bc_dsrl_*`，`bc_zero` vs 各 `bc_dsrl_*`）：给出逐种子的提升 / 退化计数和精确 McNemar（二项）p 值。Wilson 区间只作描述，不作为“有无提升”的判据。不要把 M6 的 0.62 当作配对基线。
6. **actor 方向测试**：单步 bandit，特征固定，奖励只取决于 w 某一分量的符号，全部为终止转移；固定种子和 CPU 线程数，小网络，CPU 上几秒内完成；断言确定性 w 的该分量从约 0 变为明显大于 0。
7. **真实吞吐复测**：用正式配置（8 环境、2048 宽、UTD 20），在 `outputs/m7/benchmark_real/` 下跑一次**真实训练循环**的短程计时：可临时把热身步数调小，热身后至少跑数百个 vec step，包含特征编码、写回放、SAC 更新、一次完整验证（16 条 + 1 条录像）和一次 checkpoint 保存。分项给出采集、编码、更新、验证、I/O 的耗时，重新估算 2e6 步的总时长。超过约 24 h 时，给出能在 24 h 内完成的 `total_env_steps`，写进配置并说明代价。不要改 UTD 和网络宽度。
8. **数值健康日志**：训练日志（按 `log_every`）增加 mean log π、Q 与 target Q 的均值和最大值、α、actor 确定性输出的均值绝对值。

### 低优先级（同轮完成）

9. `encode_actor_obs` 收紧输入：只接受两路图像、`tcp_pose`、`gripper_opening`；出现 `state` 键（或缺少 `tcp_pose`）直接报错。M6 的 `_cond` 不改。加测试。
10. 续训时比对不可变配置（`noise_scale`、SAC 超参数、网络尺寸、种子设置），与 checkpoint 中的不一致就报错退出。交付写明 `save_replay: false` 续训时回放池为空、不重新热身、凑够一个 batch 后立即恢复 UTD 20 的行为。

### 只写进交付，不改代码

- SAC 与 SB3 的细节差异：critic 损失没有 0.5 因子；更新顺序为 critic → actor → α（SB3 先更新 α）；critic 直接吃 w 而不是 w / 1.5；初始化不同（`tasks/M7_recheck_r1.md`“SAC 专项”）。
- 每个转移平均 2.5 次更新（官方 5 次）。
- 目标熵 0 在 128 维下，初期 log π 约 −86。

## 回归

- `pytest tests -q -p no:cacheprovider` 全部通过（原有测试不得删改）。
- 第 2 项的确定性回归。
- `scripts/check_env.py` 退出码 0；`scripts/m1_grasp_smoke.py` 输出 `GRASP_OK`。
- 冒烟训练（`--smoke --out outputs/m7/smoke_r2/train`）和续训能跑通，日志中有第 8 项的新字段，验证视频单独产出。

## 交付物

`tasks/M7_delivery_r2.md`，包含：改动文件清单；运行过的命令及输出摘要；逐项对应修改清单 1–10 的完成情况和证据；确定性回归的比对结果；真实吞吐分项和新的时长估计；更新后的正式训练、续训、四条正式评测（`bc_zero`、`bc_seeded`、`bc_dsrl` best、`bc_dsrl` last，均为 `--videos none`）、单独录像、对比的命令；未完成或存疑的点。

## 完成标准

清单 1–10 全部完成，回归全部通过。做不到的项不得绕过或降低标准；如实写明，并作为失败上报。
