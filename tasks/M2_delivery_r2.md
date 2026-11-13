# M2 交付记录（r2）

环境：`conda run -n AIControl`（Python 3.12.14，mujoco 3.17.0，numpy 2.5.4）。没有新装包，没有改已安装包的版本。没有改 `docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`，没有改 M1 的 `sim/`、`kin/ik.py`、`robot.py`，也没有改 `kin/fk.py`、`kin/spatial.py`、`dyn/rnea.py` 的计算逻辑。没有 git commit / push / add。注入脚本只放在 `/tmp`，没有留在仓库里。

## 改动文件

- `scripts/m2_validate.py`：`Tracker.add` / `add_scalar` 在累计之前检查有限性；非有限样本把该项标成 `FAIL nonfinite`，`ok` 为假，退出码非 0。腕部条件数仍是 `sigma_max / max(sigma_min, 1e-16)`，CSV 列名、纵轴和图注改为「分母下限 1e-16 的条件数估计」，`wrist_angle = 0` 标注「理论上为无穷大」。
- `scripts/m2_coupling.py`：保留 shoulder / elbow 扫描和 1 s、0.5 rad 轨迹。新增 wrist_angle 全范围扫描、elbow × wrist_angle 平面上 shoulder–wrist_angle 的热力图（含 armature）、峰值速度 2 rad/s 的快速档，以及不含重力的动态耦合对比。被驱动关节力矩为 `M_ii q̈ + (C q̇)_i + g + B q̇ + frictionloss·sign(q̇)`。
- `tests/test_m2_validate.py`：新增。覆盖四种非有限样本、有限但超阈值、正常通过，以及条件数文案常量。

## 命令和输出摘要

### `conda run -n AIControl python -B scripts/m2_validate.py`

退出码 0。15 项全部 PASS，误差与 r1 相同（M 的相对误差仍是 `1.033e-10`）。`validation_errors.csv` 增加 `nonfinite` 列，正常运行全是 0。

```
fk_pos_world                                  4.441e-16    9.236e-13 PASS
jacobian                                      1.110e-15    3.249e-11 PASS
M                                             6.661e-16    1.033e-10 PASS
g                                             5.329e-15    1.130e-12 PASS
christoffel_Cqd                               6.261e-13    3.899e-11 PASS
Mdot_minus_2C_skew                            1.110e-16    0.000e+00 PASS
ALL WITHIN TOLERANCE
```

腕部：`wrist_angle = 0` 的最小奇异值仍是 `3.860333e-19`（roll = 0）和 `1.248109e-17`（roll = 1）。按分母下限 1e-16 算出的估计是 `1.873698e16` 和 `1.734085e16`，图和日志都标明这里理论上为无穷大。`wrist_angle = 0.4` 的估计是 25.79 和 23.55，公式没改，非奇异点数值不变。

### NaN 注入

`runpy.run_path` 返回的是 globals 的副本，改那份副本不会影响 `main`。注入打在 `main.__globals__` 上，并把 `OUT` 指到 `/tmp/m2-r2-nan-inject`，避免覆盖 `outputs/m2`。仓库里没有这份脚本。

```python
ns = runpy.run_path("scripts/m2_validate.py", run_name="m2_validate_inject")
globs = ns["main"].__globals__
globs["OUT"] = Path("/tmp/m2-r2-nan-inject")
globs["gravity_torque"] = lambda chain, q, state=None: np.full(6, np.nan)
sys.exit(ns["main"]())
```

命令：`PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B /tmp/m2_r2_nan_inject.py`

`g` 一行是 `0.000e+00  0.000e+00  1.0e-08  1.0e-08 FAIL nonfinite`，最后打印 `FAILED g`。其余 14 项仍 PASS。退出码 1。

### `conda run -n AIControl python -B scripts/m2_coupling.py`

退出码 0，打印 `COUPLING_OK`。1 s 档与 r1 一致：峰值速度 0.9375 rad/s，峰值加速度 2.8867 rad/s²；shoulder 运动时 elbow 保持力矩 4.3237 N·m（上限的 0.2162）。快速档是同一段 0.5 rad 五次多项式，时长 `0.5 × (15/8) / 2 = 0.46875` s，峰值速度正好 2.0000 rad/s，峰值加速度 13.1375 rad/s²。端点速度和加速度为 0。

### `PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -m pytest tests -q -p no:cacheprovider`

`29 passed in 2.82s`，退出码 0。含原来的 M0、M1、M2 测试，外加 3 个新测试。

### `conda run -n AIControl python -B scripts/check_env.py`

退出码 0。`ALL CHECKS PASSED`。torch 2.15.0+cu132，CUDA 13.2，sm_120，GEMM 最大误差 `2.289e-04`。

## E1

沿 shoulder 全范围，三对系数的最大偏差是 `2.220e-16`（含 armature）和 `4.996e-16`（不含）。沿 elbow 全范围，elbow–wrist_angle 的跨度是 `3.608e-16` / `7.216e-16`。沿 wrist_angle 全范围（其余关节在 home）：

| 关节对 | 含 armature | 不含 armature |
|---|---|---|
| shoulder–elbow | [+0.1047, +0.2633] | [+0.3805, +0.6684] |
| shoulder–wrist_angle | [−0.2259, +0.4122] | [−0.4793, +0.7492] |
| elbow–wrist_angle | [−0.0135, +0.3931] | [−0.0374, +0.8647] |

elbow × wrist_angle 平面上，shoulder–wrist_angle 的归一化系数（含 armature）是 [−0.2555, +0.4824]。

可放进报告的两句：沿 shoulder 扫描时这三对系数是常数，因为它们都在 shoulder 及其下游，绕与 elbow 平行的 shoulder 轴转动只是把该子树刚体旋转，相对几何不变。elbow–wrist_angle 不随 elbow 变化，因为这两个关节都在 elbow 下游，绕 elbow 转动不改变这一对的相对几何和惯性子块，含与不含 armature 都一样。

## E2

不含重力的动态耦合 `M_ij q̈_j + 科氏/离心`。快速档相对 1 s 档的时间缩放是 `1/0.46875`，动态项按这个比的平方放大，实测倍数是 4.55。峰值和与力矩上限的比：

| 主动关节 | 被动关节 | 1 s 峰值 (N·m) | 比值 | 快速档峰值 (N·m) | 比值 |
|---|---|---:|---:|---:|---:|
| shoulder | elbow | 0.3076 | 0.0154 | 1.3998 | 0.0700 |
| elbow | shoulder | 0.2506 | 0.0125 | 1.1407 | 0.0570 |
| elbow | wrist_angle | 0.0679 | 0.0068 | 0.3089 | 0.0309 |
| shoulder | wrist_angle | 0.0387 | 0.0039 | 0.1761 | 0.0176 |

其余被动关节都更小。没有任何一个被动关节接近力矩上限。

被驱动关节本身的力矩是 `M_ii q̈_i + (C q̇)_i + g_i + B_i q̇_i + frictionloss_i·sign(q̇_i)`。只动一个关节时，其他关节加速度为 0，所以没有额外的惯性耦合灌进主动关节；`(C q̇)_i` 的量级是 `1e-16`，单轴转动的离心力不对该转轴产生力矩。符号与 r1 的摩擦核对一致：电机要抵消 `qfrc_passive = −B q̇` 和 `−frictionloss·sign(q̇)`。

| 档 | 关节 | 峰值 (N·m) | 比值 | 饱和 | 峰值时刻的分解 |
|---|---|---:|---:|---|---|
| 1 s | shoulder | 5.9197 | 0.2960 | 否 | 只有重力 −5.9197（t = 1 s） |
| 1 s | elbow | 4.3237 | 0.2162 | 否 | 只有重力 −4.3237（t = 0） |
| 快速 | shoulder | 10.8371 | 0.5419 | 否 | t = 0.3891 s：M_ii q̈ −8.2473，C 0，g −5.8639，B q̇ +1.2742，摩擦 +2.0000 |
| 快速 | elbow | 7.4925 | 0.3746 | 否 | t = 0.3891 s：M_ii q̈ −6.7475，C 0，g −3.1692，B q̇ +1.2742，摩擦 +1.1500 |

分项峰值不能相加。shoulder 快速档分开看是 |M_ii q̈| 8.52、|g| 5.92、|B q̇| 4.00、frictionloss 2.00，加起来会超过 20 N·m；减速段里惯性和重力同号，阻尼和摩擦仍顺着正速度，反而把净力矩从这两项的和里减回去。实际峰值是 10.84 N·m。

## 结论

1. 非有限值不再能混过验收。正常样本的 15 项误差没有变化。
2. `wrist_angle = 0` 的条件数理论上为无穷大。图和 CSV 里的约 `1e16` 是分母下限 1e-16 的估计，不是未经截断的精确值。`wrist_angle = 0.4` 仍约为 26 和 24。
3. 沿 shoulder 的水平线和沿 elbow 不变的 elbow–wrist_angle 是这条串联臂的不变性。系数真正随构型变化的是 wrist_angle：含 armature 时 shoulder–wrist_angle 扫过 [−0.226, +0.412]，平面图扫过 [−0.256, +0.482]。
4. 峰值速度提到关节上限 2 rad/s 后，不含重力的耦合力矩大约变成 1 s 档的 4.55 倍，最大 1.40 N·m，是上限的 0.070。主动关节自己的峰值是上限的 0.54 和 0.37，两档都不饱和。

## 新的输出文件

- `outputs/m2/coupling_vs_wrist_angle.csv`
- `outputs/m2/coupling_vs_wrist_angle.png`
- `outputs/m2/coupling_shoulder_wrist_map.csv`
- `outputs/m2/coupling_shoulder_wrist_map.png`
- `outputs/m2/single_joint_shoulder_fast.csv`
- `outputs/m2/single_joint_shoulder_fast.png`
- `outputs/m2/single_joint_elbow_fast.csv`
- `outputs/m2/single_joint_elbow_fast.png`
- `outputs/m2/dynamic_coupling_no_gravity.csv`
- `outputs/m2/dynamic_coupling_no_gravity.png`
- `outputs/m2/dynamic_coupling_peaks.csv`
- `outputs/m2/driven_joint_torque.csv`
- `outputs/m2/driven_joint_torque_summary.csv`

同时重写了原有的 `wrist_singularity.csv` / `.png`（列名和图注）、`validation_errors.csv`（多一列 `nonfinite`）和 `coupling_data.npz`（多了 wrist 扫描和热力图数组）。1 s 档的 `single_joint_shoulder` / `single_joint_elbow` 仍在，数值与 r1 一致。

## 未完成或存疑

没有未完成项。单关节轨迹里主动关节自己的科氏项是数值零，不是漏项。条件数曲线在奇异点上画的仍是截断后的有限估计，理论值由标注说明为无穷大。
