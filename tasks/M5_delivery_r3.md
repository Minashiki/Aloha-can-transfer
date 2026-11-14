# M5 交付 r3

在已验收的 r2 上处理 G1、G2，然后重新生成。环境全部是 `conda run -n AIControl`。没有安装包，没有 git 写操作，没有启动其他 agent，没有写 `outputs/m3/`、`outputs/m4/`。r2 已验收的性质保持不变：home 边界五次接近、float32 动作从第一次执行起、同一次执行写图像、生成与回放 float64 状态逐位一致、逐拍 phase 与 `human_fraction`、F3 分类、`git_dirty` 与 `file_sha256`、事件索引落在最终序列的斜坡起点。

## 1. 改动

- `src/aloha_pp/data/retarget.py`：运动段（合成接近、人类抓取前、搬运、合成滑移、合成下降、合成撤离）各用本段预览峰值决定放慢系数。段边界用五次插值过渡，窗口最多占相邻段的一半，也最多 `scale_blend_ticks` 个源样本。夹爪等待、斜坡、停留在这次缩放之后插入，时长固定为 M4 的数值。`simulate_stream(..., scheme="preview")` 的复核不再有 4 次上限，只在达到 `max_scale` 仍不满足时记失败。
- `configs/m5_retarget.yaml`：`scale_blend_ticks: 5`。夹爪时长的注释说明它们在缩放之后插入。
- `scripts/m5_generate.py`、`scripts/m5_analyze.py`：`episodes.csv` 增加六列段放慢系数，分析打印其分布。
- `src/aloha_pp/data/README.md`：`human_fraction` 的说明改为分段放慢，并写明夹爪块保持 M4 时长。
- `tests/test_m5_retarget.py`：边界五次过渡、后段不被前段的放慢系数拉长、夹爪拍数不随放慢变化、复核在第 6 次仍继续。

未改增益、力矩上限、速度上限、成功阈值、`random_can`、`gripper_open` / `gripper_close`，也未改受保护文件。

## 2. 命令与耗时

生成 532.6 s（8 分 53 秒），低于 20 分钟。日志 `outputs/m5/generate.log`。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_generate.py
```

| 阶段 | 秒 |
|---|---|
| 重定向（含同一次执行的渲染） | 502.5 |
| 规划器噪声 | 29.9 |
| 合计 | 532.6 |

重定向写入 120 条，规划器噪声写入 8 条。生成日志：

```text
bit-exact retarget max_qpos 0.000e+00
bit-exact planner_noise max_qpos 0.000e+00
```

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_validate.py outputs/m5/dataset/demos.hdf5
conda run -n AIControl --no-capture-output python -u scripts/m5_validate.py outputs/m5/dataset/planner_noise.hdf5
```

两份都是 `OK`。主集长度 min 193、median 242.5、mean 243.81、max 309（120 条）。备用集 min 120、median 135.5、mean 140.25、max 167（8 条）。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_analyze.py
```

重定向 120/200 = 0.600。原因：success 120，not_lifted 41，ik_unreachable 32，dropped_after_lift 7。没有 `ref_speed`。人类位置或搬运残差拍数比例 min/median/max = 0.413 / 0.486 / 0.575。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_replay.py
```

5 条全部再次成功（规划器噪声 seed 1001，重定向 source id 0、3、6、7）。`float32(回放 q)` 与存储观测的最大差是 0。float64 回放 q 与这份 float32 观测的最大差是 1.17e-7，这是量化。

```text
conda run -n AIControl --no-capture-output python -m pytest tests -q --tb=line
```

**83 passed in 6.38s**。

```text
conda run -n AIControl --no-capture-output python scripts/check_env.py
```

`ALL CHECKS PASSED`。

```text
conda run -n AIControl --no-capture-output python scripts/m1_grasp_smoke.py
```

`GRASP_OK`。

```text
conda run -n AIControl --no-capture-output python scripts/m4_run.py --e1 --out outputs/m5/regression/m4_e1
conda run -n AIControl --no-capture-output python scripts/m4_analyze.py --runs outputs/m5/regression/m4_e1/runs --out outputs/m5/regression/m4_e1
```

6 组 E1 全部成功。与 `outputs/m4/summary.csv` 的 E1 行比较 75 个共有字段、450 项，最大绝对差 **0**。

## 3. G1 分段放慢与固定夹爪

成功示范的参考速度峰值比（执行时记录的指令序列，再送入 preview 流）min/median/max = 0.4795 / 0.4955 / 0.5000。未限幅峰值与限幅峰值相同，超限标记为 0。0.5000 是四位小数；这 120 条都通过了 `≤ 0.5·v_max` 的门限。闭环逐物理步峰值比 min/median/max = 0.4794 / 0.4954 / 0.5060。超过 0.5 的只有两条，都低于 0.6 的诊断线：source 20 为 0.5005（elbow，参考 0.4997），source 119 为 0.5060（shoulder，参考 0.4999）。

各段放慢系数（成功示范，min / median / max）：

| 段 | min | median | max |
|---|---|---|---|
| synth_approach | 1.916 | 4.648 | 6.885 |
| human_pregrasp | 1.000 | 1.000 | 1.599 |
| carry | 1.000 | 1.106 | 1.861 |
| place_slide | 1.000 | 1.000 | 1.846 |
| place_down | 1.000 | 1.000 | 1.126 |
| retreat | 1.000 | 1.000 | 1.000 |

接近段承担了 home 边界的放慢（中位 4.65）。人类抓取前和搬运的中位系数是 1.00 和 1.11，不再被整条轨迹里最快的那一瞬一起拉长。

最终序列的段时长（成功示范，秒，min / median / max）。夹爪三行在 120 条上是常数：

| 段 | min | median | max | r2 中位 |
|---|---|---|---|---|
| synth_approach | 0.85 | 2.02 | 3.05 | 2.15 |
| human_pregrasp | 1.90 | 2.88 | 4.80 | 12.15 |
| grip_wait | 0.60 | 0.60 | 0.60 | 2.75 |
| grip_ramp | 1.20 | 1.20 | 1.20 | 5.30 |
| grip_dwell | 0.60 | 0.60 | 0.60 | 1.85 |
| carry | 1.95 | 2.90 | 5.00 | 11.40 |
| place_slide | 0.25 | 0.45 | 0.95 | 1.90 |
| place_down | 0.65 | 1.15 | 1.55 | 4.90 |
| retreat | 0.15 | 0.25 | 0.55 | 0 |

`grip_wait` 0.60 s 是两段「1 个滞后拍 + settle 0.25 s」（各 0.30 s）。`grip_ramp` 1.20 s 是两段 12 拍（各 0.60 s）。`grip_dwell` 0.60 s 是闭合后 0.40 s 加张开后 0.20 s。这些拍数不随放慢系数变化。

示范长度：中位 242.5 拍 = **12.12 s**（r2 是 848 拍、42.4 s）。最短 9.65 s，最长 15.45 s（r2 最长 70.4 s）。`human_fraction` 的中位从 0.561 降到 0.486，因为人类段不再被全局系数拉到十几秒，夹爪和接近段在更短的总长里占的比例更高；人类段本身仍在，抓取前中位 2.88 s，搬运中位 2.90 s。

## 4. G2 复核循环

重定向 200 条里 `ref_speed` 为 **0**。复核一直迭代到满足或 `max_scale`，没有 4 次上限。达到 `max_scale` 64 的有 8 条，它们的 IK 位置误差仍超过 `ik_skip_pos_err`，原因记为 `ik_unreachable`，不是速度搜索半途停下。

规划器噪声仍有 7 次尝试因参考峰值超过 0.5 未写入（15 次尝试里成功 8 条，写入的参考峰值比最大 0.496）。规划器本身没有改；这是写入前的同一条 0.5 过滤，不是重定向的 `slow_until`。

## 5. 数据集

| 文件 | 条数 | 大小 |
|---|---|---|
| `outputs/m5/dataset/demos.hdf5` | 120，全部 `retarget` | 512 MB |
| `outputs/m5/dataset/planner_noise.hdf5` | 8，`planner_noise` | 17 MB |

另外：`episodes.csv`（含六列段系数）、`grasp_geometry.csv`、`ik_unreachable.csv`、`summary.csv`、`timing.json`、`generate.log`、`figures/`、`videos/` 下 5 条 mp4。`exec_peak_over_0.6.csv` 为空。

## 6. 存疑

- 闭环峰值有两条略高于 0.5（0.5005、0.5060），参考峰值仍 ≤ 0.5，没有一条超过 0.6。
- 存储下来的撤离段中位只有 0.25 s。计划里的上升约 0.5 s，成功判定在撤离途中成立后截断。夹爪等待、斜坡和停留是完整的。
- 成功率从 r2 的 115/200 变为 120/200，是分段放慢改变了到达抓取点的时间，不是单独去抬成功率。`not_lifted` 41 条、`dropped_after_lift` 7 条，抓取几何仍然重叠，没有为此改抓取点。
- 工作树仍是脏的。数据集用生成时的内容哈希，不依赖一次提交。
