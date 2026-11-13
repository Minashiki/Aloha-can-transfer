# M2 交付记录（r1）

环境：`conda run -n AIControl`（Python 3.12.14，mujoco 3.17.0，numpy 2.5.4）。没有新装包，没有改已安装包的版本。没有改 `docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`，也没有改 M1 的 `sim/`、`kin/ik.py`、`robot.py`。没有 git commit / push / add。

## 参考系和方程

自写模型的世界系就是 `aloha/aloha.xml` 的 world。基座系是 `left/base_link` 的 body 系。这个 body 焊在世界上，四元数是单位四元数，所以两个系的姿态相同，原点相差 `body_pos = (-0.041018, -0.00014, 0)` m。正运动学同时给出各臂 link 和 `left/tcp` 在世界系、基座系中的位姿。几何雅可比是 `left/tcp` 的 6×6，平移在上、旋转在下，世界系，与 `mj_jacSite` 的列顺序一致。

关节参数从 `MjModel` 读出：`body_pos`、`body_quat` 在父 link 系；关节轴和锚点在子 body 系、关节运动之前。本模型每个关节锚点都在 body 原点，构造时若不是这样会直接报错。

动力学在世界系做牛顿–欧拉，再投影到关节轴上，符号与 MuJoCo 一致（正力矩使 q 增大）。方程是

`M(q) q̈ + C(q, q̇) q̇ + g(q) + B q̇ + τ_fric = τ`

M 含 armature，B = diag(damping)。`τ_fric` 不进 RNEA。手指锁在给定开度，速度和加速度都是 0，手指以及 `left/gripper_base` 作为刚性体留在树上。

MuJoCo 3.17 的 `mj_rne(flg_acc=1)` 在 `mj_fwdVelocity` 更新过空间速度之后，包含科氏 / 离心和重力，**不含 armature**。因此：

- 不含 armature 的 RNEA 直接对 `mj_rne`；
- 含 armature 的 RNEA 对 `mj_rne + armature·q̈`，也对 `mj_fullM` 的臂块乘 q̈ 再加 `qfrc_bias`。

只调用 `mj_kinematics` 而不更新空间速度时，`mj_rne` 会用过期的速度，那一次对照没有意义。

## 改动文件

新增：

- `src/aloha_pp/kin/spatial.py`：四元数、轴角、旋转误差，只用 numpy。
- `src/aloha_pp/kin/fk.py`：从 `MjModel` 拷贝参数，正运动学、`left/tcp` 雅可比、腕轴共线判定、运动学参数表。
- `src/aloha_pp/dyn/rnea.py`：RNEA、单位加速度列的 M、动能形式的第二份 M、g、C(q,q̇)q̇、Christoffel 的 C、`Ṁ−2C`、合并惯量参数表。
- `scripts/m2_validate.py`、`scripts/m2_coupling.py`。
- `tests/test_m2_kinematics.py`、`tests/test_m2_dynamics.py`。
- `tasks/M2_delivery_r1.md`。

修改：

- `src/aloha_pp/kin/__init__.py`、`src/aloha_pp/dyn/__init__.py`：导出新符号。原有 IK 导出保留。

没有删文件。

`outputs/m2/` 被 gitignore 排除，里面有：

- `kinematic_parameters.csv`、`kinematic_parameters.md`
- `inertia_parameters.csv`、`inertia_parameters.md`
- `validation_errors.csv`
- `wrist_singularity.csv`、`wrist_singularity.png`
- `coupling_vs_elbow.csv`、`coupling_vs_shoulder.csv`、`coupling_curves.png`
- `coupling_heatmap.csv`、`coupling_heatmap.png`
- `coupling_shoulder_elbow_map.png`、`coupling_data.npz`
- `single_joint_shoulder.csv`、`single_joint_shoulder.png`
- `single_joint_elbow.csv`、`single_joint_elbow.png`

## 容差

绝对容差和相对容差都是 `1e-8`，写在 `scripts/m2_validate.py` 里。双精度大约 `2e-16`。臂长不到 1 m，抽样里的关节角是 O(1)，正确的旋转连乘应该落在 `1e-15` m / rad 附近。质量矩阵元素大约 `1e-3` 到 `1` kg·m²，抽样力矩大约 `1` 到 `10` N·m，同样的舍入远小于 `1e-12` N·m。`1e-8` 比这个舍入高四到七个数量级，又比报告里会当成物理量的毫米、毫弧度、毫牛·米小。相对误差只在参考值绝对值大于 `1e-8` 的分量上取，避免和真零比出一个很大的比值。任一项超出容差时脚本退出码非 0。

Christoffel 符号用四阶中心差分，步长 `h = 1e-3` rad：

`(M(q−2h) − 8 M(q−h) + 8 M(q+h) − M(q+2h)) / (12 h)`

二阶差分、步长 `1e-6` 时，较小的科氏分量上相对残差大约 `7e-8`，过不了 `1e-8`。四阶、`1e-3` 把绝对残差压到 `6e-13`、相对残差压到 `4e-11`。armature 不随 q 变，不进入这些偏导数。

## 运行命令和输出摘要

全部带 `conda run -n AIControl`。

### `python -B scripts/m2_validate.py`

200 组 `(q, q̇, q̈)`，种子 `20261113`。q 在关节范围内均匀抽取，q̇ 在 `[-2, 2]` rad/s，q̈ 在 `[-10, 10]` rad/s²。同一批臂的样本分别在手指开度 `0.037` 和 `0.015` 上各算一遍，手指速度和加速度为 0。退出码 0。

```
item                                       max_abs      max_rel      abs_tol    rel_tol    status
fk_pos_world                                  4.441e-16    9.236e-13    1.0e-08    1.0e-08 PASS
fk_rot_world                                  1.055e-15    1.055e-15    1.0e-08    1.0e-08 PASS
fk_pos_base                                   4.441e-16    4.837e-14    1.0e-08    1.0e-08 PASS
fk_rot_base                                   8.572e-16    8.572e-16    1.0e-08    1.0e-08 PASS
jacobian                                      1.110e-15    3.249e-11    1.0e-08    1.0e-08 PASS
M                                             6.661e-16    1.033e-10    1.0e-08    1.0e-08 PASS
M_energy                                      4.441e-16    3.747e-12    1.0e-08    1.0e-08 PASS
g                                             5.329e-15    1.130e-12    1.0e-08    1.0e-08 PASS
Cqd                                           1.554e-15    2.368e-12    1.0e-08    1.0e-08 PASS
rnea_body_vs_mj_rne                           1.066e-14    9.453e-14    1.0e-08    1.0e-08 PASS
rnea_full_vs_mj_rne_plus_armature             1.066e-14    4.271e-12    1.0e-08    1.0e-08 PASS
rnea_full_vs_Mqdd_plus_bias                   1.066e-14    1.177e-11    1.0e-08    1.0e-08 PASS
passive_vs_minus_B_qd                         0.000e+00    0.000e+00    1.0e-08    1.0e-08 PASS
christoffel_Cqd                               6.261e-13    3.899e-11    1.0e-08    1.0e-08 PASS
Mdot_minus_2C_skew                            1.110e-16    0.000e+00    1.0e-08    1.0e-08 PASS
```

`M` 是单位加速度列的 RNEA（含 armature）对 `mj_fullM` 的 6×6 臂块。`M_energy` 是用各 body 质心雅可比和动能写出的第二份 M，对第一份。`g` 对 q̇ = 0 时的 `qfrc_bias`。`Cqd` 对关掉重力后的 `qfrc_bias`。`passive_vs_minus_B_qd` 是 `qfrc_passive = −B q̇`，200×2 组上误差是 0。Christoffel 和反对称性用另外 8 个构型，手指开度 0.037。

手指自由度（交接单第 4 节）：

- 两种开度下，自写的“手指刚性固定”模型和 `mj_fullM` 臂块一致，最大绝对误差 `6.661e-16`。推测成立：手指速度、加速度为 0 时，臂关节块就是把手指锁在该开度上的 6 自由度模型。
- 开度从 0.037 改到 0.015，臂块本身会变，元素最大差 `2.497e-4`，平均绝对差 `9.953e-5`。这个差别高于 `1e-8`，所以开度要进模型；它又远小于 shoulder / elbow 的 armature（0.395 / 0.383）。
- 臂–手指耦合块 `|M_af|` 最大 `6.347e-2`。手指若有加速度，会在臂上产生力矩；加速度为 0 时臂方程只用 `M_aa`。
- 两指同时以 0.05 m/s 滑动时，臂上的 `qfrc_bias` 最多变化 `4.474e-4` N·m。速度不为 0 时，偏置力不再等于锁死模型。

摩擦演示用 M1 的 home（`configs/m1_pick_place.yaml` 的 `home_q`），shoulder 速度 0.4 rad/s，elbow 速度 −0.25 rad/s，接触数 0：

```
qfrc_passive arm [ 0.  -0.8  0.5  0.   0.   0. ]
-B qd          [-0.  -0.8  0.5 -0.  -0.  -0. ]
qfrc_constraint arm [ 0.   -2.    1.15  0.    0.    0.  ]
frictionloss arm    [0.   2.   1.15 0.   0.   0.  ]
finger equality qfrc_constraint [ 0.001201 -0.001201]
```

阻尼只在 `qfrc_passive`，等于 `−B q̇`。shoulder / elbow 的 frictionloss 在 `qfrc_constraint` 里，大小是 frictionloss，方向与速度相反（`−frictionloss·sign(q̇)`）。手指上那一对等大反向的约束力来自 1:1 的 equality，不是 frictionloss。

腕部：`forearm_roll` 和 `wrist_rotate` 都是局部 +X 铰链，中间只有 `wrist_angle` 绕局部 Y。两轴共线当且仅当 `wrist_angle` 是 π 的整数倍；关节范围内只有 0。`forearm_roll` 取 0 和 1 rad 时，交叉积的最小值都在 `wrist_angle = 0`，线距约 `1e-17` m。该处雅可比最小奇异值约 `4e-19` / `1e-17`，条件数约 `1.87e16` / `1.73e16`。`wrist_angle = 0.4` 时最小奇异值约 `0.072` / `0.074`，条件数约 26 / 24。自写雅可比的最小奇异值与 `mj_jacSite` 相差最大 `3.886e-16`。曲线在 `wrist_singularity.png`。

### `python -B scripts/m2_coupling.py`

典型构型是 M1 home：`[0, -0.246326, 0.356357, 0, 1.460443, 0]`。退出码 0，约 10 s。

```
M diag with armature    [0.11938 0.64864 0.53068 0.15254 0.02104 0.00260]
M diag without armature [0.11938 0.25364 0.14768 0.01254 0.01304 0.00260]
armature                [0     0.395  0.383  0.14   0.008  0    ]
```

归一化系数 `M_ij / sqrt(M_ii M_jj)`，典型构型下绝对值最大的四对：

| 关节对 | 含 armature | 不含 armature |
|---|---:|---:|
| waist–wrist_rotate | −0.3945 | −0.3945 |
| waist–forearm_roll | +0.2106 | +0.7345 |
| elbow–wrist_angle | +0.2012 | +0.4846 |
| shoulder–elbow | +0.1641 | +0.4974 |

沿 elbow 全范围（其余关节停在 home）：shoulder–elbow 含 armature 为 `[+0.0114, +0.3908]`，不含为 `[+0.0548, +0.9419]`；shoulder–wrist_angle 含 armature 为 `[−0.0839, +0.2732]`，不含为 `[−0.1855, +0.4418]`。elbow–wrist_angle 在这条扫描上是常数（+0.2012 / +0.4846）。沿 shoulder 全范围，这三对都是常数。shoulder–elbow 在 shoulder×elbow 平面上：含 armature `[+0.0117, +0.3908]`，不含 `[+0.0563, +0.9419]`。

单关节五次多项式，1 s 内 +0.5 rad，峰值速度 0.9375 rad/s，峰值加速度 2.8867 rad/s²。其他关节的保持力矩峰值：

- shoulder 运动时，elbow 保持力矩最大 4.3237 N·m，是 20 N·m 上限的 0.2162；其中重力 4.3237，惯性耦合 0.2779，科氏 / 离心 0.1164。wrist_angle 最大 0.3402 N·m（上限 10）。
- elbow 运动时，shoulder 保持力矩最大 3.7938 N·m，是上限的 0.1897；重力 3.7938，惯性耦合 0.2663，科氏 / 离心 0.1022。wrist_angle 最大 0.3235 N·m。

0.1897 和 0.2162 与 M1 重力保持在 home 上的静力矩比一致：这条慢轨迹的峰值就是重力，惯性耦合和科氏项没有把峰值再抬高。没有关节碰到力矩上限。

### `python -B -m pytest tests -q -p no:cacheprovider`

```
..........................                                               [100%]
26 passed in 2.86s
```

含原来的 M0、M1 测试。退出码 0。M2 覆盖 20 个随机构型（两种开度各 10 个）上的 FK、雅可比、M、g、C q̇、RNEA；M 对称且特征值大于 `1e-8`；两份 M 一致；5 个构型上 Christoffel 与 RNEA 的速度积一致，并且 `Ṁ−2C` 反对称。

### `python -B scripts/check_env.py`

```
python: 3.12.14
torch: 2.15.0+cu132
torch.version.cuda: 13.2
mujoco: 3.17.0
numpy: 2.5.4
[PASS] cuda_available: True
[PASS] sm_120: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
[PASS] gpu_matmul: shape=2048x2048 max_abs_err=2.289e-04 rtol=0.01 atol=0.01
[PASS] mujoco_load_step: neq=1 steps=20
[PASS] wrist_cam: shape=(96, 96, 3) dtype=uint8 min=0 max=45 saved=/home/minashiki/AIControl/FinalHomework/outputs/m0_wrist_cam.png
ALL CHECKS PASSED
```

退出码 0。

## 耦合分析结论

1. 典型构型下，含 armature 时归一化耦合最大的是 waist–wrist_rotate（−0.395）和 waist–forearm_roll（+0.211）。shoulder、elbow 的对角线里 armature 占了大半（0.649 里的 0.395，0.531 里的 0.383），所以 shoulder–elbow 只有 +0.164。
2. 去掉 armature 后，shoulder–elbow 升到 +0.497，elbow–wrist_angle 升到 +0.485，waist–forearm_roll 升到 +0.735。elbow 扫过全范围时，不含 armature 的 shoulder–elbow 系数最高到 +0.942，含 armature 时最高 +0.391。armature 把重关节上的归一化耦合压到大约原来的三分之一到二分之一。
3. shoulder 角不改变这三对系数：shoulder 和 elbow 的轴平行，绕 shoulder 转不会改变它们和下游关节的相对几何。forearm_roll 固定为 0 时，elbow–wrist_angle 也不随 elbow 角变化。耦合随构型的变化主要出现在 elbow 扫描上。
4. 1 s、0.5 rad 的五次多项式（峰值 0.94 rad/s、2.89 rad/s²）下，其他关节的保持力矩由重力主导。elbow 被 shoulder 带动时峰值 4.32 N·m，是上限的 0.22；惯性耦合只有 0.28 N·m，科氏 / 离心 0.12 N·m。wrist 上的保持力矩不超过 0.34 N·m，上限是 4 到 10 N·m。这条速度下没有力矩饱和。
5. `left/gripper_link` 合并 `gripper_base` 和两指之后质量 0.8458 kg。开度从 0.037 收到 0.015，质量不变，最大主惯量从 `2.613e-3` 降到 `2.363e-3` kg·m²。其他 link 没有可动子 body，合并结果就是自身的 `body_mass` / `body_ipos` / `body_inertia`。

## 未完成或存疑

没有未完成项。三件实现时已经核对、但不该写成解析恒等式的事：

- RNEA 要求关节锚点在 body 原点。`aloha.xml` 编译后确实全是 0；模型若改成非零锚点，构造函数会拒绝，而不是悄悄算错。
- C 矩阵的偏导数是四阶数值差分，不是符号微分。它和 RNEA 速度积的差是 `6e-13`，`Ṁ−2C` 的对称残差是 `1e-16`。
- `τ_fric` 只在验证脚本里对照到 `qfrc_constraint`，没有放进自写的光滑逆动力学。这是任务要求的边界，不是漏项。
