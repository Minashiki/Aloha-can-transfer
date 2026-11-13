# M2 复核 r1（Codex）

日期：2026-11-13；基线 `c6b6910`。已阅读 dispatch、handover、delivery、review 及 PLAN 第 5 节和 M2。假设：按当前 ALOHA 六臂关节、手指锁定的明确建模范围验收，不要求通用机器人库；后续报告增强与当前实现缺陷分开判断。本次只写本报告，未安装包、改环境、启动 agent 或执行 git 写操作。

## 一、逐条回应

### 1. 合规（四项）

| 编号 | 原审查项 | 判断 | 独立依据 |
|---|---|---|---|
| C1 | 受保护文件未改 | 部分同意 | `git diff c6b6910 -- .gitignore environment.yml scripts/check_env.py aloha assets/scene` 无输出，支持受跟踪文件未改。PLAN、WORKFLOW 被忽略，本次没有派发前哈希，不能独立重证 Claude 所述全部哈希核对，也没有证据认定它们被改动。 |
| C2 | M1 代码未改 | 同意 | `git diff c6b6910 --stat` 只有 kin/dyn 的 __init__.py；对 sim、kin/ik.py、robot.py 的 diff 无输出。两个 init 的差异是保留既有 IK 导出并增加新模块导出。 |
| C3 | 模型代码不调用 MuJoCo 计算函数 | 同意 | 阅读 fk.py、spatial.py、rnea.py 全文，fk.py:102/105/143 仅用名称查询，30–31 用枚举，参数数组均复制；forward、Jacobian、RNEA、M、C、g 均由 NumPy 运算完成。dyn/rnea.py 不导入 mujoco。名称查询符合只读参数约束，不属于计算器偷用。 |
| C4 | 没有 commit、只在 AIControl 运行 | 部分同意 | `git log -1 --oneline` 为 `c6b6910 M1：ALOHA 模型修复、PickPlaceCan 场景、DLS IK 与抓取环境`。本次 Python 均用 AIControl；交付虽然总述声明环境前缀，但多个小节命令实际省略前缀，不能据此证明每条历史执行的解释器，更不能由 HEAD 证明所有 git 历史操作。未发现违规证据。 |

### 2. 复跑（三项）

**R1：同意正常数值结果。** 为避免改写 outputs，用 runpy 加载脚本，仅在 main 的 globals 将 OUT 改为 `/tmp/m2-recheck-codex/validate`，运行原 main。退出码 0，原 15 项全部 PASS：

| 项 | max_abs | max_rel |
|---|---:|---:|
| fk_pos_world | 4.441e-16 | 9.236e-13 |
| fk_rot_world | 1.055e-15 | 1.055e-15 |
| fk_pos_base | 4.441e-16 | 4.837e-14 |
| fk_rot_base | 8.572e-16 | 8.572e-16 |
| jacobian | 1.110e-15 | 3.249e-11 |
| M | 6.661e-16 | 1.033e-10 |
| M_energy | 4.441e-16 | 3.747e-12 |
| g | 5.329e-15 | 1.130e-12 |
| Cqd | 1.554e-15 | 2.368e-12 |
| rnea_body_vs_mj_rne | 1.066e-14 | 9.453e-14 |
| rnea_full_vs_mj_rne_plus_armature | 1.066e-14 | 4.271e-12 |
| rnea_full_vs_Mqdd_plus_bias | 1.066e-14 | 1.177e-11 |
| passive_vs_minus_B_qd | 0 | 0 |
| christoffel_Cqd | 6.261e-13 | 3.899e-11 |
| Mdot_minus_2C_skew | 1.110e-16 | 0 |

这支持当前有限样本的正确性，不能推出验证器能拒绝所有无效计算，见 N1。

**R2：同意。**

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -m pytest tests -q -p no:cacheprovider
```

`26 passed in 2.50s`。另以 runpy 仅将 check_env 的 IMAGE_PATH 指向 `/tmp/m2-recheck-codex/m0.png` 后运行原 main，退出码 0：torch 2.15.0+cu132、CUDA 13.2、sm_120；GEMM 最大误差 2.289e-04；neq=1、20 步；腕部图 (96,96,3)、min=0/max=45；`ALL CHECKS PASSED`。

**R3：同意图表结果。** 已实际查看 `outputs/m2/coupling_curves.png` 和 `single_joint_shoulder.png`，坐标轴、图例、量纲与力矩限值文字齐全；保持力矩图显示惯性/科氏项与重力分别绘制，并非只显示总力矩。原 coupling main 仅将 OUT 改到 `/tmp/m2-recheck-codex/coupling` 后也复跑，退出码 0：

```text
M diag with armature    [0.11938 0.64864 0.53068 0.15254 0.02104 0.00260]
M diag without armature [0.11938 0.25364 0.14768 0.01254 0.01304 0.00260]
shoulder-elbow typical +0.1641 / +0.4974
elbow scan shoulder-elbow [+0.0114,+0.3908] / [+0.0548,+0.9419]
elbow-wrist_angle typical +0.2012 / +0.4846
peak |qd| 0.9375 rad/s; peak |qdd| 2.8867 rad/s²
shoulder drive: elbow max|total| 4.3237; inertia 0.2779; C 0.1164 N·m
elbow drive: shoulder max|total| 3.7938; inertia 0.2663; C 0.1022 N·m
COUPLING_OK
```

上述分项最大值不一定发生在同一时刻，不应把分项峰值直接相加解释为总峰值。outputs/m2 的 18 个原图表/数据/参数表均存在；本次没有覆盖它们。

### 3. 技术核对（四项）

**T1，mj_rne 不含 armature：同意，限于当前实测 API/版本。** RNEA 的 body 力/力矩递推在 `dyn/rnea.py:59–115`，轴投影后于 121–122 单独添加 armature*qdd。验证脚本先更新位置/速度再设置 qacc，避免旧空间速度；同样比较含 armature 的 mj_fullM*qdd+bias，处理正确。

本次另用独立种子 9471、两开度共 8 个状态，手工按 model.joint(name) 找索引、设置 qpos/qvel/qacc，通过 mj_forward/mj_fwdVelocity/mj_rne 构造参照，得到：

```text
max |rnea_full - mj_rne_arm - model.dof_armature_arm*qdd| = 3.774758283725532e-15
```

该实验没有借用原验证脚本的 _prepare。结论可写入报告；不把它推广成未经版本验证的所有引擎接口规则。

**T2，耦合系数不变性：同意这些扫描结果。** 对当前串联臂，改变 shoulder 角使有关子树共同刚性旋转，内积形式的惯性项不变；shoulder 后的 elbow 相对角可改变 shoulder 的惯性与耦合。elbow–wrist_angle 位于 elbow 下游，整体绕 elbow 旋转不改变这两关节的相对几何和它们的惯性子块。含/不含 armature 两种系数的分母均满足相同不变性。这里应讨论相应子块和归一化分母，不泛化为“整个 M 都不依赖 shoulder”——涉及 waist 的项一般仍可变化。平行轴符合本模型，但不能只凭平行轴四字推出任意机构都满足这些结论。

**T3，手指锁定模型：部分同意，补充适用边界。** 两开度下 M_aa 误差 6.661e-16，开度变化引起最大差 2.497e-04；|M_af| 最大 6.347e-02；非零手指速度演示的 bias 差 4.474e-04，均复现。递推中手指固定位置、零速度/加速度，并把其质量与惯量向父 body 传递，正是任务要求。

但“只用 M_aa”必须伴随明确约束 qd_f=qdd_f=0。这不是实际自由滑动或仅 equality 两指联动时的有效自由动力学；若手指自由响应，应考虑耦合块，而非把 M_aa 自动当成消元后的质量矩阵。交付已写锁定边界，允许保留其任务范围内结论。

**T4，容差/Christoffel：同意，独立性需准确描述。** 四阶模板符号正确；`dyn/rnea.py:221–233` 的 deriv[k,i,j]=∂M_ij/∂q_k、Christoffel 三项及 qd 收缩索引正确。h=1e-3 是对平滑旋转惯性函数的合理数值选择，未通过放宽 1e-8 的验收阈值过关。200 组、两种开度、全关节范围、qd±2/qdd±10 足以满足 M2 冒烟与规格；不等于全输入数学证明。Christoffel 的额外 8 点和 pytest 的 5 点开度为 0.037，范围已披露。

独立性分层：

- RNEA 与 MuJoCo 的 mj_rne/bias/fullM 是外部计算对照；同读模型参数是本任务允许的输入，并非用同一结果比较自身。
- 单位加速度 M 与动能雅可比 M 共用 FK/惯性参数，但公式路径不同；再有 MuJoCo M 对照，不能称“只做同源比较”。
- Christoffel Cqd 与 RNEA 速度积走数值偏导与递推两条路径，提供实质交叉验证。
- 原 Mdot 与 C 来自同一份对称化 deriv，反对称残差很小主要证明代数索引自洽；它本身不能独立证明偏导数正确。原本已有 Cqd 对照，故这不是已证实的实现错误。

本次另外对 **MuJoCo 的 M** 做独立二阶中心差分（不调用自写 mass_matrix），构造 Christoffel C 和 Mdot：h=1e-4 时 C 差 3.164e-09、独立 skew 残差 1.266e-08；h 缩到 2e-5 时分别降为 **1.300e-10、5.200e-10**，符合二阶截断误差下降，支持自写实现。前一组残差超 1e-8 来自该探针的较粗二阶差分，不能误报为原四阶实现失败。

### 4. F1–F3

**F1【低】：反驳其“变量不好/曲线没信息”的缺陷定性。** `scripts/m2_coupling.py:265` 后按规格明确扫描 shoulder 与 elbow，完全符合 dispatch 第 4 节。水平线实证了理论不变性，恰好能用于解释哪些构型因素不影响所选惯性子块，不能称没有信息量或需要替换。加入 wrist_angle/forearm_roll 扫描有报告增强价值，但应作为额外分析，保留要求的 shoulder 扫描；不能因实现按 spec 工作就认定 spec 有缺陷。

**F2【低】：部分同意。** 0.94 rad/s 下动态项相对重力较小是事实，快速轨迹有助于展示其随时间缩放的平方增长；但任务明确把“0.5 rad、1 s 五次轨迹”列为可接受例子，实现已照做。图中动态项已经单独绘制，能看到其非零耦合，不能说当前只能证明“耦合很小”：无 armature 的惯性归一化系数最高 0.942，动态幅值还取决于加速度。新增快档与去重力曲线是可选增强，不是通过 M2 的必要修复；若加入，确应考虑主动关节本身的阻尼/摩擦和力矩限值，避免从其他关节保持力矩推断全部关节不饱和。

**F3【低】：部分同意。** `kin/fk.py:29` 和 YAML 均写 left/tcp 是事实，但模型明确针对固定名称的唯一 TCP，并非承诺跟随任意环境配置。当前没有 M1 那种两个名称实际矛盾的 bug。集中常量有维护价值；要求环境额外拒绝配置并修改 M1 robot.py 则是设计选择，而不是唯一必要修复，尤其当前 dispatch 原则上不改 M1 行为。避免为去除一个字符串重复引入跨层约束。

### 5. Cursor 状态：部分同意

未查看 Cursor GUI，不能独立重证 86.8%；更换对话与交接属于协调工作流，非动力学实现验收。本次未启动任何 agent。

### 6. 总结论：部分同意

同意主体数学实现正确、自写约束遵守、正常样本全部通过、参数表和图表已生成；不同意必修项“只有 F1–F3”，它们主要是增强建议，而 N1 是实质验收门禁缺陷。另“误差比容差小五个数量级以上”若指全部指标并不准确：Christoffel 绝对误差 6.261e-13 相对 1e-8 约四个数量级；M 最大相对误差 1.033e-10 仅约两个数量级。数值仍充分达标，无需夸大安全余量。

## 二、新增问题

### N1【中】非有限值可被误差 Tracker 静默记为零，完整验证会错误通过

- 位置：`scripts/m2_validate.py:71–87`。
- 现象：`max(已有有限值, nan)` 保留已有值，故 max_abs/max_rel 不记录 NaN；ref 为 NaN 时相对误差 mask 又排除该分量。add_scalar 也有同样问题。探针结果：

```text
Tracker.add([nan],[1])       -> max_abs=0, max_rel=0, ok=True
Tracker.add([1],[nan])       -> max_abs=0, max_rel=0, ok=True
Tracker.add([inf],[inf])     -> max_abs=0, max_rel=0, ok=True
Tracker.add_scalar(nan)      -> max_abs=0, ok=True
```

- 完整复现：仅在 runpy 的内存 globals 将 gravity_torque 改为返回六维 NaN，运行原 main（输出重定向 /tmp），结果仍为：

```text
g  0.000e+00  0.000e+00  1.0e-08  1.0e-08 PASS
ALL WITHIN TOLERANCE
exit 0
```

- 影响：数值实现今后因溢出、无效输入等返回非有限值时，核心验收命令可能把失效结果当成精确一致。当前正常数据没有发现 NaN，这不否定已复现的门禁缺口；pytest 中 assert_allclose 的拒绝能力也不能替代独立脚本的退出契约。
- 建议：在累计前明确检查 got/ref/标量的有限性，有异常就将该项标记 FAIL 或抛出错误；不能只在累计后的 max 上检查 isfinite，因为 NaN 已被 max 丢弃。
- 验收：上面四种情况必须失败，有限且超阈值情况仍失败；完整 NaN 注入必须非零，正常 15 项仍通过。测试应验证该失败路径。

### N2【低】腕部“condition number”实际采用分母截断值，未在图表/交付说明

- 位置：`scripts/m2_validate.py:130–132`、wrist_singularity.csv/图和交付的腕部结论。
- 现象：使用 `sigma_max / max(sigma_min, 1e-16)`，而不是通常的 sigma_max/sigma_min。roll=0、wrist_angle=0 时 sigma_min=3.860333e-19，记录 cond=1.873698e16；直接以该浮点 SVD 值作比约为 4.85e18。理论奇异点条件数应视为无穷，浮点极小值本身也不宜过度精确解释。
- 影响：曲线能正确显示病态趋势，但读者会将约 1e16 误认为未经截断的精确条件数，且 CSV 无法看出该量被限制。不会影响 FK/Jacobian 或 RNEA 的数值正确性。
- 建议：将列名/图例/报告明确标成采用 1e-16 分母下限的条件数估计，或保留原比值并用绘图显示上限单独处理；对理论奇异点明确标注。
- 验收：CSV、图注与实现公式一致，零点不宣称有限且准确的理论条件数，非奇异点统计不变。

## 三、总结

计数口径：4 条合规、3 条复跑、4 条技术核对、F1–F3、1 条 Cursor 状态、1 条结论，共 16 项：**同意 8、反驳 1、部分同意 7；新增 2（中 1、低 1）**。

- 同意：C2/C3，R1/R2/R3，T1/T2/T4。
- 反驳：F1 的缺陷定性与替换必需性，保留可选增强方向。
- 部分同意：C1/C4，T3，F2/F3，Cursor 状态、总结论。
- 新增：N1 非有限误差漏检；N2 条件数截断未说明。没有发现需要改写 RNEA 递推或 Christoffel 公式的证据。

验证命令：

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -m pytest tests -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B /tmp/m2-recheck-codex/run.py
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B /tmp/m2-recheck-codex/independent.py
```

run.py 运行仅重定向 OUT/IMAGE_PATH 的原验证、耦合、自检函数及 Tracker 单例探针；independent.py 运行独立 MuJoCo M 差分、armature 核对与完整 NaN 注入。两次差分仅在 /tmp 修改探针步长。本次所有 Python 均在 AIControl，matplotlib 缓存也设置在 /tmp；项目内唯一写入为本报告。
