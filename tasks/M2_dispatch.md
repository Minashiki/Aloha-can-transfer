# M2 派发：运动学 / 动力学建模与验证

**先读这份交接单：`tasks/handover_M2_1.md`。**

项目根目录：`/home/minashiki/AIControl/FinalHomework`。背景见 `docs/PLAN.md` 第 5 节和第 10 节 M2（只读）。

## 硬性约束（违反任一条即判为不合格）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`）。禁止使用系统 Python、base 环境，禁止新建环境，不得改变已安装包的版本。交付记录里的命令要写出环境前缀。
2. 不得修改：`docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/` 下的模型文件、`assets/scene/`。M1 的代码（`sim/`、`kin/ik.py`、`robot.py`）原则上不改；确实需要改时，在交付记录中说明理由，并且不得改变已有行为（M1 的测试必须仍然通过）。
3. 不得执行 `git commit` / `push` / `config` / `add`，不得添加 co-author。
4. 不得启动其他 agent。
5. **“自写”的含义**：自写的 FK、雅可比、RNEA、M、C、g 只能从 `MjModel` 中**读取参数**（`body_pos`、`body_quat`、`body_ipos`、`body_iquat`、`body_mass`、`body_inertia`、`jnt_axis`、`jnt_pos`、`dof_armature`、`dof_damping` 等），**不得调用** MuJoCo 的计算函数（`mj_kinematics`、`mj_forward`、`mj_rne`、`mj_fullM`、`mj_jac*` 等）来得到结果。MuJoCo 的计算函数只能在验证代码中用作对照。只用 numpy。

## 建模约定

- 对象：`aloha/aloha.xml`（机器人本体，不含 Can）。对 6 个臂关节建模；手指关节固定在给定的开度、速度为 0，手指及以下的所有 body 都视为刚性附着在所属 link 上。
- 坐标系：世界坐标系 = `aloha.xml` 的 world；`left/base_link` 固定在世界中。在交付记录中写明自写模型采用的是哪个参考系。
- 动力学方程：`M(q)q̈ + C(q,q̇)q̇ + g(q) + B·q̇ + τ_fric = τ`，其中 M 包含 armature，B = diag(damping)。`τ_fric` 只做说明，不进入验证（MuJoCo 中它是约束力）。

## 要做的事

### 1. 运动学（`src/aloha_pp/kin/`，例如 `fk.py`）

- 正运动学：任意 q 下，各臂 link 以及 `left/tcp` 在世界坐标系和 base 坐标系中的位姿。
- 几何雅可比：`left/tcp` 的 6×6 雅可比，平移在上、旋转在下，世界坐标系，与 `mj_jacSite` 的约定一致。
- 导出一张运动学参数表（各关节的轴、相对父 link 的位置和姿态），供报告使用，格式为 Markdown 或 CSV，放在 `outputs/m2/`。
- 腕部奇异：给出 `forearm_roll` 与 `wrist_rotate` 共轴时的条件（例如 `wrist_angle` 的取值），并在该条件附近统计雅可比最小奇异值 / 条件数，输出曲线。

### 2. 动力学（`src/aloha_pp/dyn/`）

- RNEA：`τ = RNEA(q, q̇, q̈, gravity)`。
- `M(q)`：用 CRBA 或对 RNEA 做单位加速度列求解均可，**包含 armature**。
- `g(q)` = RNEA(q, 0, 0, gravity)；`C(q,q̇)q̇` = RNEA(q, q̇, 0, 无重力)。
- C 矩阵本身（用于耦合分析和性质验证）：由 M 的偏导数按 Christoffel 符号构造（偏导数可用解析方法或中心差分，写明方法和差分步长），并验证 `C(q,q̇)q̇` 与 RNEA 的结果一致，以及 `Ṁ − 2C` 是反对称矩阵。
- 导出惯性参数表（各 link 质量、质心、惯量，包括附着在其上的手指等子 body 的合并结果），放在 `outputs/m2/`。

### 3. 交叉验证（`scripts/m2_validate.py`）

- 在关节范围内随机取 ≥ 200 组 (q, q̇, q̈)，固定随机种子；手指开度取 0.037 和 0.015 两种。
- 逐项与 MuJoCo 对比：FK（各 link 和 `left/tcp`）↔ `mj_kinematics`；雅可比 ↔ `mj_jacSite`；M ↔ `mj_fullM` 中 6×6 的臂关节块；g、C q̇ ↔ `qfrc_bias`（分别在 q̇ = 0 和无重力的条件下比较）；完整 RNEA ↔ `mj_rne`（flg_acc = 1）。
- 对照时手指关节的速度和加速度都取 0，并检查手指自由度对臂关节块的影响（这是交接单第 4 节中的推测）。
- 输出每一项的最大绝对误差和最大相对误差。**容差事先写在脚本里**，并在交付记录中给出依据（double 精度、链长、数值量级）；目标量级为 ≤ 1e-8，任何一项超出容差时退出码非 0。
- 同时报告 `qfrc_passive`（阻尼）与 `B·q̇` 的对比，说明 frictionloss 出现在 `qfrc_constraint` 中。

### 4. 关节耦合分析（`scripts/m2_coupling.py`）

- M(q) 的非对角项：归一化耦合系数 `M_ij / sqrt(M_ii·M_jj)` 随构型的变化。至少对 shoulder–elbow、shoulder–wrist_angle、elbow–wrist_angle 三对，沿 elbow、shoulder 的取值做扫描，输出曲线和热力图（某个典型构型下的 6×6 矩阵）；并说明 armature 对耦合系数的影响（分别给出含 armature 和不含 armature 的结果）。
- 单关节驱动：让一个关节（至少 shoulder 和 elbow 各做一次）按平滑轨迹运动（例如五次多项式，1 s 内转动 0.5 rad），其他关节保持静止。用自写的逆动力学计算其他关节需要的保持力矩，并把它分解为惯性耦合项（M_ij q̈_j）、科氏 / 离心项和重力项；与力矩上限对比，输出曲线。
- 图片保存到 `outputs/m2/`（png），数据保存为 csv / npz。脚本几分钟内运行完毕。

### 5. 测试（`tests/test_m2_*.py`）

- 少量随机构型（例如 20 个）上的 FK、雅可比、M、g、C q̇、RNEA 与 MuJoCo 的一致性；M 对称正定；`Ṁ − 2C` 反对称；单位加速度列构造的 M 与另一种实现的结果一致（如果做了两种实现）。
- 原有的 M0、M1 测试必须仍然通过。

## 交付物

1. 上面列出的代码、脚本、测试，以及 `outputs/m2/` 下的图表和参数表（`outputs/` 被 gitignore 排除，交付记录中要列出文件名）。
2. 交付记录 `tasks/M2_delivery_r1.md`，必须包含：
   - 改动文件清单；
   - 运行过的命令及输出摘要：`m2_validate.py` 的完整误差表、`m2_coupling.py` 的关键数值、`pytest tests -q`、`check_env.py`；
   - 容差的选择依据；对手指自由度推测的验证结论；
   - 耦合分析的主要结论（3–6 条，附数值，供报告使用）；
   - 未完成或存疑的点。

## 完成标准

- `m2_validate.py` 退出码为 0，所有项都在容差之内；`pytest tests -q` 全部通过；`check_env.py` 退出码为 0。
- 耦合分析的图表和数据已生成。
- 有做不到的项，不得绕过或降低标准；在交付记录中如实写明，并作为失败上报。
