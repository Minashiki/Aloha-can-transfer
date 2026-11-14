# M4 交付记录（r1）

环境：`conda run -n AIControl`。增益、力矩上限、速度上限、成功阈值、`random_can` 范围和夹爪开合量都没有改。未执行 git commit / push / add。HEAD 仍是 `10d094b`。

规划器读的是 `get_privileged()` 在 reset 之后给出的 Can 位姿，再对着目标格中心规划。它是使用物体真值的对照组。

## 改动文件

- `src/aloha_pp/control/quintic.py`：`JointStream.push(q_target, q_future=None)`。`q_future is None` 时仍是 M3 的等待模式。传入未来结点时本拍播放当前目标，滞后 50 ms。
- `src/aloha_pp/control/tracking.py`：`set_target(..., next_action=None)`。有下一目标时做第二次 IK，初值是当前解。夹爪指令改为 `action[7]`，并夹到夹爪执行器的 `ctrlrange`。
- `src/aloha_pp/control/gains.py`：预留按关节 ω。默认等于 M3 的 ω=20 时，返回的数组就是 `configs/m3_track.yaml` 里的 `kp` / `kd` / `ki`。
- `src/aloha_pp/sim/env.py`：`step(..., next_action=None)`。`next_action is None` 时不把这个参数传给 `set_target`，M3 的三参数 mock 仍然可用。左右手指接触分开记录，`fingers_released` 仍是“两指都没接触”。
- `src/aloha_pp/planning/schedule.py`、`planner.py`、`metrics.py`、`__init__.py`：状态机、路点、失败分类和任务指标。
- `configs/m4_task.yaml`：等待、斜坡、停留、放置间隙、按关节 ω 的占位。不覆盖 M1 / M3 的增益和阈值。
- `scripts/m4_run.py`、`scripts/m4_analyze.py`：实验只写日志；分析只读日志。
- `tests/test_m4_reference.py`、`tests/test_m4_task.py`。

没有改 `docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、`kin/fk.py`、`kin/spatial.py`、`dyn/rnea.py`、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`。`.gitignore` 在本任务开始前已有未提交改动，这次没有再动它。

## 1. 未来目标接口

`TrackingController.set_target(model, data, action, next_action=None)`，`AlohaCanEnv.step(action, ..., next_action=None)`。

`next_action is None` 时走 M3 的 lookahead 缓存：本拍保持，时刻 `kT` 的指令在 `(k+2)T` 到达，滞后 100 ms。有 `next_action` 时，本拍播放的段以当前目标的 IK 解为终点，在 `(k+1)T` 到达，滞后 50 ms。终点速度和加速度仍是中心差分

`(q_{k+1} - q_{k-1}) / (2T)`，`(q_{k+1} - 2 q_k + q_{k-1}) / T²`。

`q_{k+1}` 是下一目标的 IK 解，求解前把种子设成当前解，解完再把种子设回去。一个 tick 最多两次 IK，两次耗时分别记在 `ik_solve_s` 和 `ik_next_solve_s`。下一目标的 IK 失败时，这一拍退回“未来结点等于当前关节角”，并清掉 IK 成功标记。

速度超限的处理没改：超限的关节走 M3 的制动 / 限幅，并留下速度超限标记。

`q_future` 只参与差分，不入队。同一条流里两种模式切换时：

- 缓存是空的，又给了未来结点：立刻播放当前目标，用未来结点做中心差分，播完不入队。滞后回到 50 ms。
- 缓存里还有等待模式欠下的结点，而这一拍的目标就是它：把这个结点播放一次，差分用新给的未来结点，然后清空缓存。同一个结点只播一次。
- 缓存里的结点和下发的目标不同（策略改了主意）：先把欠下的结点播完，差分用这次真正下发的目标，再把新目标放进缓存。欠下的结点不丢。
- 纯等待模式、缓存又是空的：和 M3 一样，只缓存、本拍保持。

每一段的起点都是当前的 `q`、`q̇`，所以 `q_d` 和 `q̇_d` 在切换处保持 C¹。规划器的最后一个动作把自身当作 `next_action`，这样最后一个结点会在预览模式里被播出来，而不是留在缓存里。

夹爪不再写死 `gripper_open`。指令取 `action[7]`，再夹到执行器 `ctrlrange`（0.002–0.037）。`reset` 后的 50 步稳定段仍把夹爪保持在 `gripper_open`。M3 的动作里夹爪一直是 `gripper_open`，回归表和这条改动一致。

按关节 ω 写在 `configs/m4_task.yaml` 的 `omega_joints`。这次是 `null`，六个关节都是 ω=20、ζ=1。`resolve_gains` 在这个情况下直接返回 M3 配置里的数组，测试核对过逐元素相同。M4 没有启用按关节 ω。正式实验的控制器是 C3，`gravity_compensation` 为 false。

### E4：滞后和纹波

路径与 `m3_run.py` 相同，Can 移到 `[1.5, 1.5, 0.2]`，不抓取。C3，fixed_1 / fixed_2，fast / medium / slow。fast 用各自参考模式在速度上限下重新搜索，medium = fast × 2，slow = fast × 4。两种模式搜到的时长相同，也和 M3 相同：fixed_1 为 2.35 / 4.70 / 9.40 s，fixed_2 为 3.65 / 7.30 / 14.60 s。限制来自段的内部，少掉的那 50 ms 没有缩短最短时长。

肩关节 `q̈_d` 的 RMS，以及 25 个物理步滑动平均之后的 RMS：

| 组 | 模式 | 关节 RMSE | TCP 对原始参考 | TCP 对 FK(q_d) | 饱和占比 | 肩饱和 | 力矩峰值 | q̈ RMS | q̈ 滑动平均 RMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 fast | 等待 | 2.99 mrad | 36.4 mm | 2.36 mm | 0.938% | 4.95% | 26.6 Nm | 8.372 | 6.942 |
| fixed_1 fast | 未来目标 | 3.01 mrad | 19.3 mm | 2.38 mm | 0.954% | 5.03% | 26.6 Nm | 8.444 | 7.003 |
| fixed_1 medium | 等待 | 3.21 mrad | 21.0 mm | 2.78 mm | 0 | 0 | 12.3 Nm | 2.068 | 1.919 |
| fixed_1 medium | 未来目标 | 3.23 mrad | 11.4 mm | 2.80 mm | 0 | 0 | 12.3 Nm | 2.078 | 1.928 |
| fixed_1 slow | 等待 | 3.42 mrad | 12.1 mm | 3.10 mm | 0 | 0 | 9.09 Nm | 0.512 | 0.500 |
| fixed_1 slow | 未来目标 | 3.43 mrad | 7.24 mm | 3.11 mm | 0 | 0 | 9.09 Nm | 0.514 | 0.501 |
| fixed_2 fast | 等待 | 2.92 mrad | 32.5 mm | 2.37 mm | 0.188% | 0.894% | 26.3 Nm | 5.431 | 4.618 |
| fixed_2 fast | 未来目标 | 2.94 mrad | 16.9 mm | 2.38 mm | 0.190% | 0.905% | 26.3 Nm | 5.463 | 4.645 |
| fixed_2 medium | 等待 | 3.22 mrad | 18.1 mm | 2.70 mm | 0 | 0 | 11.6 Nm | 1.332 | 1.247 |
| fixed_2 medium | 未来目标 | 3.23 mrad | 9.83 mm | 2.70 mm | 0 | 0 | 11.6 Nm | 1.337 | 1.251 |
| fixed_2 slow | 等待 | 3.40 mrad | 10.3 mm | 2.89 mm | 0 | 0 | 8.38 Nm | 0.330 | 0.322 |
| fixed_2 slow | 未来目标 | 3.40 mrad | 6.20 mm | 2.89 mm | 0 | 0 | 8.38 Nm | 0.331 | 0.322 |

未来目标相对等待模式，肩关节 `q̈_d` RMS 的比值是 1.002–1.009，滑动平均后也是这个范围，低于 1.2。medium 和 slow 两种模式的饱和占比都是 0。fast 在这条 M3 路径上仍然饱和，肩关节大约 5.0%（fixed_1）和 0.90%（fixed_2），力矩峰值顶到约 26.6 Nm 和 26.3 Nm。这和等待模式是同一件事：速度上限挡住了参考速度，挡不住这段参考里的加速度，所以命令力矩还会碰到上限。

相对原始参考的 TCP RMSE 降到等待模式的大约一半到六成。关节 RMSE、相对 `FK(q_d)` 的 TCP 误差和力矩峰值几乎不变。E1 和 E2 使用未来目标模式。

## 2. 规划器

状态顺序：approach → pregrasp → descend → wait_close → close → dwell_closed → lift → carry → descend_place → wait_open → open → dwell_open → retreat → return → wait_success。编号 0–14。

八个路点：home、Can 正上方（加 `pregrasp_clearance` 0.08 m）、抓取点（Can 中心）、抬起（加 `lift_height` 0.10 m）、目标格正上方、放置点、撤离（与目标格正上方重合）、回到 home。姿态从上往下。段与段之间用 M3 的五次时间律，按 20 Hz 采样成 `[xyz, quat(wxyz), gripper]`，经 `env.step(action, next_action=...)` 下发。`next_action` 是下一条，最后一条重复自身。

运动段的最后一拍命令路点本身；参考晚一个控制周期到达，开合前的等待把这段滞后算进去。

**yaw。** Can 的碰撞体是圆柱，绕竖直轴转动不改变罐和手指的接触，改变的是手指往哪边伸。TCP 的 +Y 是手指张开方向。`yaw = 0` 让张开方向沿基座 +Y，`yaw = π/2` 让它沿基座 −X。`choose_yaw` 比较 Can 到左 bin 两面内壁的剩余距离，张开方向朝剩余更大的那一侧。放置沿用同一个 yaw，避免小增益的腕关节在持罐时再扭转。fixed_1、fixed_2 都是 yaw = 0。E2 的 50 个种子里 39 个是 0，11 个是 π/2。

**放置高度。** 闭合段的最后一拍读 `can_center - tcp`。偏移的模长大于 5 cm 时丢掉，当作这次没抓住。TCP 的放置位置是格子中心减去这个偏移，竖直方向是 `cell_z + place_gap - offset_z`，并且不低于 `cell_z + place_min_above`。`place_gap` 是 6 mm，`place_min_above` 是 3 mm。瞄准的是成功判定用的格子中心。`place_pos` 在 fixed_2 上比格子中心偏了大约 14 mm，两种误差都记。时长搜索阶段还没有实测偏移，用 0 偏移、TCP 高度 `cell_z + gap`。闭合后按同样的 tick 数重建设放置之后的段；抬起之前的前缀不变，所以停留段里已经发出的 `next_action` 仍是停留姿态。实测偏移大约是 x −1.7 mm、y +0.6 mm、z −2.8 mm，罐心在 TCP 下方约 3 mm。松手时罐心被放在格子中心上方 6 mm，罐底靠近底板，留出这段下落。

**开合时机。** `settle_s = 0.25`，即 5/ω，大约五个闭环时间常数。等待拍数是 1 个滞后拍加 `round(0.25/0.05) = 5` 拍，共 6 拍、0.30 s，闭合前和张开前一样。`pregrasp_hold_s = 0.10`（2 拍）。斜坡 `grip_ramp_ticks = 12`，和 M1 冒烟测试相同，采样不含起点，最后一拍到达完全张开或完全闭合。闭合后 `post_close_dwell_s = 0.40`（8 拍，对应 M1 抬起前的停留）再抬起。张开后 `post_open_dwell_s = 0.20`。成功等待最长 `success_wait_s = 2.0`，成功计数满 0.5 s 就结束。开合时刻 TCP 相对该路点的位置误差写在元数据 `tcp_err_close`、`tcp_err_open`。

三档只乘运动段的 tick。fast 是未来目标模式下、速度上限内的最短总运动 tick，按段长比例分配后再用完整日程做 IK 和 `simulate_stream(scheme="preview")`。medium、slow 是同一组 tick 乘 2 和乘 4。开合、等待、停留三档相同。重复姿态沿用上一次 IK。

| 位置 | 档 | 运动 tick | 运动时长 | 实际结束 | 峰值 \|q̇\|（waist … wrist_rotate, rad/s） |
| --- | --- | --- | --- | --- | --- |
| fixed_1 | fast | 9, 5, 7, 13, 6, 7, 7 | 2.70 s | 4.75 s | 1.880, 1.002, 1.327, 0.014, 2.203, 1.882 |
| fixed_1 | medium | ×2 | 5.40 s | 6.75 s | 0.932, 0.491, 0.643, 0.015, 1.027, 0.932 |
| fixed_1 | slow | ×4 | 10.80 s | 10.75 s | 0.465, 0.241, 0.322, 0.016, 0.511, 0.465 |
| fixed_2 | fast | 13, 6, 8, 24, 8, 8, 13 | 4.00 s | 5.70 s | 1.935, 0.866, 1.024, 0.017, 1.743, 1.936 |
| fixed_2 | medium | ×2 | 8.00 s | 8.65 s | 0.966, 0.415, 0.492, 0.018, 0.865, 0.967 |
| fixed_2 | slow | ×4 | 16.00 s | 14.55 s | 0.483, 0.205, 0.247, 0.018, 0.430, 0.483 |

速度上限是 `[2, 2, 2, 3, 3, 3]` rad/s。搜索给出的峰值与上限之比，fixed_1 fast 是 0.940，fixed_2 fast 是 0.970。slow 的实际结束早于排满的运动时长，是因为罐子已经放下并静止，成功计数在回 home 的途中走完，循环提前停止。

**失败分类**，每个 episode 一个原因，顺序如下。成功则原因是空。

| 原因 | 条件 |
| --- | --- |
| `ik_fail` | 任一物理步的 IK 成功标记为假 |
| `not_grasped` | `dwell_closed` 期间两指没有同时接触 Can |
| `dropped` | 已经抓住，并且搬运中罐的 z 低于抓取高度减去 `drop_z`（0.03 m），或两指都松开时罐仍高于格子中心 2 cm、连续不少于 10 个物理步（20 ms） |
| `not_released` | 最后一步任一手指仍接触 Can |
| `outside` | 结束时罐心不在目标格内 |
| `not_still` | 结束时线速度不低于 `vel_thresh`（1 cm/s） |
| `timeout` | 日程走完仍未成功 |
| `other` | 上面都不是 |

持罐段是 `dwell_closed` 到 `wait_open`。搬运判定用 lift、carry、descend_place。

## 3. 日志

沿用 M3 的 npz + json，逐物理步。新增 `gripper_cmd`、`gripper_q`、`can_linvel`、`finger_contact`（N×2）、`planner_state`、`success_count`、`success`、`ik_solve_s`、`ik_next_solve_s`、`ik_success`。`phase` 在任务里是规划器状态，在 E4 里是 M3 轨迹相位。元数据有模式、种子、Can 初始位姿、目标格、`place_pos`、失败原因、`reference_mode`、yaw、抓取偏移、开合时刻的 TCP 误差。

## 4. 实验

`conda run -n AIControl python -u scripts/m4_run.py`

脚本内 `total_s 121.70`。其中 E4 闭环 6.61 s，E1 7.65 s，E2 95.60 s，视频 7.20 s，时长搜索和启动约 4.64 s。conda 启动算在外面，整条命令大约 123 s。低于 15 分钟。

`conda run -n AIControl python -u scripts/m4_analyze.py` 第一次约 2.3 s。补上 E2 汇总行后又跑了一次。

视频是 agentview，480×360，10 fps。ffmpeg 把高度补到 16 的倍数 368。实验本身不渲染相机，视频是事后按同一规划重放的。单条都小于 2 MB：

| 文件 | 字节 |
| --- | --- |
| `E1_C3_fixed_1_fast.mp4` | 64860 |
| `E1_C3_fixed_1_medium.mp4` | 80237 |
| `E1_C3_fixed_1_slow.mp4` | 106101 |
| `E1_C3_fixed_2_fast.mp4` | 77499 |
| `E1_C3_fixed_2_medium.mp4` | 102020 |
| `E1_C3_fixed_2_slow.mp4` | 138211 |
| `E2_C3_random_s00.mp4` | 76717 |

E2 没有失败，所以只留了一条成功视频。`--viewer` 接在 MuJoCo 被动 viewer 上，默认关闭，这次没有显示器，没有测。

输出（不进 git）：

- `outputs/m4/runs/`：E1 六组、E2 五十组、E4 十二组，各一份 npz 和 json。
- `outputs/m4/summary.csv`、`summary.md`、`summary_main.md`、`summary_wrist.md`。csv 在五十行之后有 `E2_summary`。
- `outputs/m4/figures/e1_fixed1_medium.png`、`e1_paths.png`、`e1_speed.png`、`e2_scatter.png`、`e4_compare.png`。
- `outputs/m4/videos/` 上表七个文件。
- `outputs/m4/m3_regression/`：M3 重跑的 `summary.csv`、`summary.md`、`runs/`、`videos/`、`figures/`，以及原先就在 `outputs/m3/` 里的 `r1_baseline.json`。

### E1

六组都成功。关节 rad，长度 m，力矩 Nm。`place_xy` 是罐心相对格子中心的水平误差，`place_pos` 那一列是相对配置里 `place_pos` 的三维误差。

| 组 | 成功时间 | 运动时长 | 结束 | 关节 RMSE | TCP 对参考 | TCP 对 FK(q_d) | 饱和 | 力矩峰值 | place xy | 对 place_pos | 倾斜 | 滑移 | 闭合 TCP 误差 | 张开 TCP 误差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_1 fast | 4.722 | 2.70 | 4.75 | 0.002838 | 0.002984 | 0.002970 | 0 | 16.82 | 0.003810 | 0.003810 | 0.000250 | 0.002265 | 0.003616 | 0.003788 |
| fixed_1 medium | 6.714 | 5.40 | 6.75 | 0.003178 | 0.003306 | 0.003288 | 0 | 10.02 | 0.004633 | 0.004634 | 0.000234 | 0.001354 | 0.004162 | 0.004154 |
| fixed_1 slow | 10.716 | 10.80 | 10.75 | 0.003359 | 0.003378 | 0.003358 | 0 | 8.351 | 0.005417 | 0.005417 | 0.000235 | 0.001112 | 0.004176 | 0.004147 |
| fixed_2 fast | 5.672 | 4.00 | 5.70 | 0.003056 | 0.002957 | 0.002932 | 0 | 14.42 | 0.004043 | 0.01622 | 0.000259 | 0.002794 | 0.003837 | 0.003891 |
| fixed_2 medium | 8.622 | 8.00 | 8.65 | 0.003283 | 0.003097 | 0.003086 | 0 | 9.090 | 0.004744 | 0.01684 | 0.000226 | 0.001717 | 0.004129 | 0.004029 |
| fixed_2 slow | 14.516 | 16.00 | 14.55 | 0.003420 | 0.003113 | 0.003094 | 0 | 7.732 | 0.005834 | 0.01785 | 0.000212 | 0.001511 | 0.004107 | 0.003950 |

速度超限标记六组都是 0。fixed_2 相对 `place_pos` 大约 16–18 mm，是因为 `place_pos` 离格子中心大约 14 mm，罐子落在格子中心附近。

持罐段腕部。力矩上限 wrist_angle 10 Nm，wrist_rotate 4 Nm。

| 组 | angle 误差 | rotate 误差 | angle 力矩 | rotate 力矩 | angle 占比 | rotate 占比 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_1 fast | 4.12 mrad | 3.39 mrad | 1.846 | 0.576 | 18.5% | 14.4% |
| fixed_1 medium | 3.31 mrad | 1.76 mrad | 0.993 | 0.281 | 9.9% | 7.0% |
| fixed_1 slow | 7.17 mrad | 1.12 mrad | 0.580 | 0.139 | 5.8% | 3.5% |
| fixed_2 fast | 3.52 mrad | 3.46 mrad | 1.463 | 0.586 | 14.6% | 14.7% |
| fixed_2 medium | 2.87 mrad | 1.80 mrad | 0.796 | 0.291 | 8.0% | 7.3% |
| fixed_2 slow | 6.96 mrad | 1.27 mrad | 0.473 | 0.145 | 4.7% | 3.6% |

全程峰值和持罐段差不多，fast 的全程 wrist_angle 力矩是 1.889 Nm（18.9%）。

### E2

C3，medium，`random`，种子 0–49。50 / 50 成功。成功率 1.000，Wilson 95% 区间 [0.929, 1.000]。失败计数是 `success:50`，没有 `ik_fail`、`not_grasped`、`dropped`、`outside`、`not_still`、`not_released` 或 `timeout`。

放置 xy 误差中位数 4.50 mm，最大 4.77 mm。三维误差中位数 4.50 mm，最大 4.77 mm。关节 RMSE 中位数 3.19 mrad。力矩峰值最大 11.48 Nm。滑移最大 1.52 mm。持罐段 wrist_angle 误差最大 3.70 mrad，wrist_rotate 最大 1.96 mrad。逐条表在 `outputs/m4/summary.md` 和 `summary.csv` 的 `E2_summary` 行。

这 50 个初始位置都放进了 fixed_1 那一格。`random_can` 只按罐半径往内收，更贴壁的几何这组种子没有做成失败样本。

## 5. 结论

1. C3 在 fixed_1 和 fixed_2 的三档速度下都把 Can 放进格子并满足成功判定。关节 RMSE 停在 2.8–3.4 mrad，三档几乎一样。峰值力矩随放慢下降：fixed_1 是 16.8、10.0、8.4 Nm，fixed_2 是 14.4、9.1、7.7 Nm。任务轨迹上六组饱和占比都是 0。放置 xy 误差是 3.8–5.8 mm，fast 略小，放慢并没有单调地把罐子放得更准；这几毫米主要来自松手后大约 6 mm 的下落。E4 使用的 M3 跟踪路径上，fast 肩关节仍然饱和大约 5%，力矩峰值 26.6 Nm。速度上限能挡住参考速度，挡不住这段参考的加速度，所以只限速度不足以避免力矩饱和。任务自己的 fast 峰值停在 16.8 Nm，没有碰到 20 Nm 的肩上限。没有为了消掉 E4 的饱和去改增益或加加速度限制。

2. 随机初始位置、medium、种子 0–49：50 次全部成功。Wilson 95% 区间是 [0.929, 1.000]。这组样本里没有可分类的失败。放置误差集中在 4.5 mm 附近，最大 4.8 mm。

3. 未来目标把滞后从 100 ms 降到 50 ms。相对原始参考的 TCP RMSE 降到等待模式的大约一半到六成。肩关节 `q̈_d` RMS 只升到等待模式的 1.002–1.009 倍，medium 不饱和。纹波没有回来。E1、E2 因此用未来目标模式。

4. 持罐时腕部够用。wrist_angle 误差峰值最大约 7.2 mrad，力矩峰值 1.85 Nm，占 10 Nm 上限的 19%。wrist_rotate 误差峰值约 3.5 mrad，力矩峰值 0.59 Nm，占 4 Nm 上限的 15%。滑移最大 2.8 mm，和 M1 冒烟测试的 2.8 mm 同一量级。这是 11.5 g 的罐子、抓取偏移大约几毫米的结果。

## 6. 测试与回归

```text
conda run -n AIControl python -m pytest tests -q --tb=line
```

`56 passed in 3.52s`。原有 M0–M3 测试没有删改。新增测试覆盖：默认参数与 M3 等待模式逐步一致；有 `next_action` 时结点在 `(k+1)T` 到达且终点速度、加速度等于中心差分；两种模式来回切换以及“预告的下一目标和下一次实际下发不一致”时 C¹、不丢结点、不重复播放；夹爪跟随 `action[7]` 并夹到 0.002–0.037；默认 ω 和显式的全 20 向量生成的增益与 `configs/m3_track.yaml` 逐元素相同；状态顺序；开合前等待不少于滞后加稳定时间；只缩放运动段；构造日志上的失败分类；放置误差、滑移、腕部峰值和 Wilson 区间的数值。

```text
conda run -n AIControl python scripts/check_env.py
```

`ALL CHECKS PASSED`，退出码 0。

```text
conda run -n AIControl python scripts/m1_grasp_smoke.py
```

`GRASP_OK`（`hold_lift 0.0851`，`max_slip_m 0.002780`，`ik_failures 0`），退出码 0。

```text
conda run -n AIControl python -u scripts/m3_run.py
conda run -n AIControl python -u scripts/m3_analyze.py
```

`m3_run.py` 的 `total_s` 是 27.15（M3 交付记录里是 27.02，这是墙钟）。重跑的 `summary.md` 与 `tasks/M3_delivery_r2.md` 里的汇总表 40 行、每一格文本相同，`cell_diffs 0`。仓库里没有当时的 `summary.csv`，交付记录保存的是 `.6g` 文本；按这份文本，数值差就是 0，比 1e-9 更紧。等待模式的 E4 与这张表里的 C3 六行也一致，例如 fixed_1 fast 的关节 RMSE 0.002989、TCP 参考误差 0.0364231、饱和占比 0.00937853。

`m3_run.py` 把输出写死在 `outputs/m3`。重跑前这个目录已经存在，里面有 `r1_baseline.json`。同名日志和两条视频被这次重跑盖掉了。汇总与归档表逐格相同。整棵目录复制在 `outputs/m4/m3_regression/`，`outputs/m3/` 也按重跑后的内容放了回去。

## 7. 未完成或存疑

- 上面这条：M3 重跑覆盖了已有的 `outputs/m3` 同名文件。指标与归档表一致，原始未舍入 csv 本来就不在仓库里。
- E4 日志的 `success` 是 false。罐子被移开，不参与抓放判定。这列不表示跟踪失败。
- `--viewer` 没有在有显示的环境里开过。
- E2 的 50 个种子全部成功。`random_can` 只按罐半径内缩，贴壁时手指碰壁的可能仍然在，这组种子没有抽到那样的失败。
- 任务 fast 没有力矩饱和；饱和出现在 E4 的 M3 路径上，比例和 M3 交付时一样。没有改 ω、力矩上限或速度上限。
