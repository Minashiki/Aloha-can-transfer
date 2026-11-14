# M5 交付 r2

按已审批的 `tasks/M5_review_r2.md` 第 1–8 项改完并重新生成。环境全部是 `conda run -n AIControl`。没有安装包，没有 git 写操作，没有启动其他 agent，没有写 `outputs/m3/`、`outputs/m4/`。动作从第一次闭环起就按 float32 执行并按同一数值存储（二选一里选了这一支，没有另存 float64 动作）。

## 1. 改动文件

相对基线 `fe540b4`，本轮在 r1 已有文件上继续改：

- `src/aloha_pp/sim/env.py`：`capture_state` / `_reset_from_state` 增加 `time`、`qacc_warmstart`、`qacc`，以及模型有执行器状态时的 `act`。默认 `reset` 不走这条路径。
- `src/aloha_pp/data/retarget.py`：home 五次接近（首样本是 home）、段标签、事件索引随 prepend / 重采样 / 提前一拍映射、preview 参考流峰值判据。
- `src/aloha_pp/data/dataset.py`、`src/aloha_pp/data/README.md`：`phase`、`next_actions`、`time0`、`qacc_warmstart0`、`git_dirty`、`file_sha256`。
- `configs/m5_retarget.yaml`：`lift_thresh_m`、`grip_closed_tol`、`lost_physics_steps`；home 接近与速度判据的说明。
- `scripts/m5_generate.py`、`scripts/m5_replay.py`、`scripts/m5_analyze.py`、`scripts/m5_validate.py`。
- `tests/test_m5_retarget.py`、`tests/test_m5_env.py`、`tests/test_m5_dataset.py`。

未改增益、力矩上限、速度上限、成功阈值、`random_can`、`gripper_open` / `gripper_close`，也未改受保护文件。

## 2. 命令与输出摘要

生成因单条探测约 8.7 s、200 条会超过 20 分钟，从一开始就写入日志。实际总时长 **1216.9 s（20 分 17 秒）**。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_generate.py
```

日志 `outputs/m5/generate.log`，`outputs/m5/timing.json`：

| 阶段 | 秒 |
|---|---|
| 重定向（含同一次执行的渲染） | 1186.2 |
| 规划器噪声 | 30.5 |
| 合计 | 1216.9 |

重定向成功并写入 115 条。规划器噪声写入备用 8 条。生成日志里两条逐位核对：

```text
bit-exact retarget max_qpos 0.000e+00
bit-exact planner_noise max_qpos 0.000e+00
```

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_validate.py outputs/m5/dataset/demos.hdf5
```

`OK`。长度 min 318、median 848、mean 848.43、max 1408（115 条）。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_validate.py outputs/m5/dataset/planner_noise.hdf5
```

`OK`。长度 min 120、median 135.5、mean 140.25、max 167（8 条）。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_analyze.py
```

重定向 115/200 = 0.575。原因：success 115，not_lifted 43，ik_unreachable 32，ref_speed 6，dropped_after_lift 4。人类位置或搬运残差拍数比例 min/median/max = 0.462 / 0.561 / 0.704。各段秒数（min / median / max，成功示范）：

| 段 | min | median | max | 合计 |
|---|---|---|---|---|
| synth_approach | 0.90 | 2.15 | 3.25 | 241.8 |
| human_pregrasp | 3.55 | 12.15 | 20.65 | 1425.0 |
| grip_wait | 1.10 | 2.75 | 4.15 | 305.1 |
| grip_ramp | 2.35 | 5.30 | 7.60 | 587.4 |
| grip_dwell | 1.00 | 1.85 | 2.75 | 208.7 |
| carry | 3.80 | 11.40 | 20.65 | 1331.3 |
| place_slide | 0.55 | 1.90 | 4.40 | 226.6 |
| place_down | 1.05 | 4.90 | 8.45 | 552.8 |
| retreat | 0 | 0 | 0 | 0 |

撤离段在计划里有标签，但成功判定在开爪停留期间就已经成立，存盘在那一拍截断，所以最终序列里没有撤离拍。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_replay.py
```

5 条全部再次成功（1 条规划器噪声 seed 1001，4 条重定向 source id 0、1、3、5）。相对存储的 float32 关节观测，float32(回放 q) 的最大差是 **0**。float64 回放 q 与这份 float32 观测的最大差是 1.19e-7，这是量化，不是两次仿真的差。视频单条最大 362 KB。

```text
conda run -n AIControl --no-capture-output python -m pytest tests -q --tb=line
```

**79 passed in 5.86s**。

```text
conda run -n AIControl --no-capture-output python scripts/check_env.py
```

`ALL CHECKS PASSED`（CUDA 可用）。

```text
conda run -n AIControl --no-capture-output python scripts/m1_grasp_smoke.py
```

`GRASP_OK`。

```text
conda run -n AIControl --no-capture-output python scripts/m4_run.py --e1 --out outputs/m5/regression/m4_e1
conda run -n AIControl --no-capture-output python scripts/m4_analyze.py --runs outputs/m5/regression/m4_e1/runs --out outputs/m5/regression/m4_e1
```

6 组 E1 全部成功。与 `outputs/m4/summary.csv` 的 E1 行比较 75 个共有字段、450 项，最大绝对差 **0**。

## 3. 第 1–8 项

1. **F1。** 接近段首样本是 home，位置和姿态都用五次时间律。峰值判据从 home 关节角出发，对最终 float32 动作（含提前一拍和末拍重复）做与控制器一致的 IK（home 位姿返回 home 关节角，失败则保持上一解），再送入 `simulate_stream(..., scheme="preview")`。写入数据集的 115+8 条，参考速度峰值比最大 **0.4996**（重定向）和 **0.496**（规划器噪声），严格 ≤ 0.5，超限标记为 0（限幅峰值与未限幅峰值一致，说明没有触发制动）。未限幅峰值与限幅峰值在成功示范上相同：min/median/max = 0.4720 / 0.4903 / 0.4996。闭环速度是逐物理步的 `|qvel|/v_max`，不是 20 Hz 采样；成功示范 min/median/max = 0.4723 / 0.4906 / **0.5003**。没有任何示范的闭环峰值比 > 0.6，所以诊断名单为空。闭环峰值紧贴参考峰值，差大约 0.001。

2. **F2。** 恢复 `qacc_warmstart` 和 `time`（以及 `qacc` / `act`）。从第一次 `env.step` 起执行 float32 量化后的动作，`next_actions[t]` 记下该拍真正传入的预告，回放原样使用。成功与写盘是同一次带图像的执行。测试 `test_stored_action_replay_matches_bit_exact` 和生成日志里的整段 `qpos` 最大差都是 **0**。r1 的 9/122 翻转是当时恢复路径丢掉 `qacc_warmstart`，并且首轮用 float64、写盘后回放用 float32，不是 1e-12 的敏感比例。本轮没有做敏感比例统计。

3. **F3。** 接触、高度和闭合命令按物理步记录。`lost` 为闭合期间连续 ≥ 10 个物理步（500 Hz 下 20 ms）双指都不接触。`not_lifted`：闭合命令第一次到达 `gripper_close`（容差 `grip_closed_tol` 1e-4）的那个物理步的 Can 高度为参考，此后这段连续闭合里最大抬升 < `lift_thresh_m` 0.02 m。`dropped_after_lift` 排在 `outside` 之前：先抬过 2 cm，再失去双指接触或相对该参考高度跌过 M4 的 `drop_z` 0.03 m。43 条 `not_lifted` 的抬升 min/median/max = 0 / 3.7 mm / **16.2 mm**，都低于 2 cm。4 条 `dropped_after_lift`（source id 27、73、160、176）抬升 25.7 mm 到 305 mm，最终仍不在格内。本轮没有再分出 `outside`。抓取几何（本项目 TCP 的指闭合轴是 site +Y，垂直于它的水平轴是 site +Z，高度是世界 z）在成功与 `not_lifted` 上重叠：`not_lifted` 落在成功组 p10–p90 内的比例，闭合方向 0.58、垂直方向 0.88、高度 0.79。源示范的 eef–Can 偏移和接近轴世界 z 分量同样重叠（成功组接近轴 z 的中位 0.959）。不设可分性门槛。`ik_unreachable` 32 条：31 条最差样本在 `carry`（例如 source 11，位置 (0.403, −0.126, 0.352) m，位置误差 10.4 mm），1 条在 `place_slide`（source 136，(0.241, −0.492, 0.207) m，误差 10.6 mm）。

4. **F4。** 每条示范有与存储动作等长的 `phase`。重定向的取值见 README：合成接近、人类抓取前位置、夹爪等待 / 斜坡 / 停留、搬运（合成基线 + 人类残差）、合成滑移、合成下降、合成撤离。规划器噪声用 M4 的 schedule 状态，`phase_scheme=planner`。`human_fraction` 只数人类抓取前位置和搬运，不含重复的夹爪等待。姿态改成 top-down，夹爪时序改成 M4 的等待和斜坡。数据集与 README 称为「部分重定向 + 合成放置」。

5. **F5。** 根属性有 `git_commit`（HEAD `fe540b4`）、`git_dirty=true`，以及 `file_sha256`（`src/aloha_pp/data/`、`planning/noise.py`、`sim/env.py`、`scripts/m5_generate.py`、`scripts/m4_run.py`、M1 / M3 / M4 / M5 配置）。

6. **F6。** 上表是最终离散序列的时长。等待和夹爪斜坡已经随 slowdown 拉长（成功示范 slowdown min/median/max = 1.92 / 4.59 / 6.89）。中位长度 848 拍 = 42.4 s。

7. **N1。** `grasp_index` / `release_index` 是最终动作序列上的闭爪斜坡和开爪斜坡起点。`source_grasp` / `source_release` 才是变换前的源索引。测试 `test_event_index_follows_the_close_ramp` 检查闭爪斜坡标签紧跟等待标签。

8. **重新生成与回归。** 主文件 115 条全部是重定向；备用 8 条规划器噪声。校验、分析、5 条重放、`pytest`、`check_env`、`m1_grasp_smoke`、M4 E1 都已通过，见第 2 节。

## 4. 数据集

| 文件 | 条数 | 大小 |
|---|---|---|
| `outputs/m5/dataset/demos.hdf5` | 115，全部 `retarget` | 1.3 GB |
| `outputs/m5/dataset/planner_noise.hdf5` | 8，`planner_noise` | 17 MB |

另外：`episodes.csv`、`grasp_geometry.csv`、`ik_unreachable.csv`、`summary.csv`、`timing.json`、`generate.log`、`figures/` 下的新图、`videos/` 下 5 条 mp4 和 5 张双相机拼图。`exec_peak_over_0.6.csv` 为空，因为没有闭环峰值比超过 0.6 的示范。

## 5. 未完成或存疑

- 6 条重定向没有执行，原因 `ref_speed`，报告的峰值比是 0.4999、0.5001、0.5001、0.5009、0.5009、0.5012（source id 44、85、163、51、117、120）。搜索在一次放大之后若不再做 `simulate_stream` 复核，就拒绝该尺度，所以 0.4999 那条也被排除。写入的示范全部 ≤ 0.4996 且标记为 0。这是偏保守，没有把 0.5 的标准放宽。
- 合成撤离在计划中存在，最终存储长度为 0，原因是成功截断发生在开爪停留，见第 2 节。
- 抓取几何不能把 `not_lifted` 和成功分开。没有为此改抓取点或提高成功率。
- 工作树仍是脏的（`git_dirty=true`）。数据集用内容哈希指向生成时的文件，不依赖一次提交。
