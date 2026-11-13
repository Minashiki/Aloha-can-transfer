# M2 修改派发 r2（已获用户审批）

**先读这份交接单：`tasks/handover_M2_2.md`。** 然后读 `tasks/M2_review_r2.md` 的“最终结论”和 `tasks/M2_recheck_r1.md` 的 N1、N2。

项目根目录：`/home/minashiki/AIControl/FinalHomework`。

## 硬性约束（与 `tasks/M2_dispatch.md` 相同）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`）；禁止系统 Python、base 环境、新建环境，不得改变已安装包版本。交付记录中的命令写出环境前缀。
2. 不得修改：`docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、M1 代码（`sim/`、`kin/ik.py`、`robot.py`），以及 `kin/fk.py`、`kin/spatial.py`、`dyn/rnea.py` 的计算逻辑。
3. 不得 `git commit` / `push` / `config` / `add`；不得添加 co-author。
4. 不得启动其他 agent。
5. 只改下面列出的内容，可修改的文件：`scripts/m2_validate.py`、`scripts/m2_coupling.py`、`tests/test_m2_*.py`（可新增测试文件）。

## 要修改的内容

### 1. N1：验收脚本拒绝非有限值（必修）
- `scripts/m2_validate.py` 的误差累计（`Tracker` 及标量累计）在累计**之前**检查 got / ref / 标量是否有限；出现 NaN / Inf 时该项判为 FAIL（并记录出现非有限值），最终退出码非 0。
- 验收：
  - 新增 pytest：`[nan] vs [1]`、`[1] vs [nan]`、`[inf] vs [inf]`、标量 NaN 四种情况都判为 FAIL；有限但超阈值的情况仍判为 FAIL；正常情况 PASS。
  - 一次性完整注入（例如用 runpy 把某个计算函数替换为返回 NaN，**不要把注入代码留在仓库**）：退出码非 0；在交付记录中写明注入方法和输出。
  - 正常运行：15 项仍全部 PASS，退出码 0。

### 2. N2：条件数的说明（必修）
- 腕部条件数的 CSV 列名、图例或图注、交付说明都要写明“分母下限 1e-16 的条件数估计”；理论奇异点（wrist_angle = 0）标注“理论上为无穷大”。非奇异点的数值不变。

### 3. E1：耦合扫描增强（报告用）
- **保留**现有的 shoulder / elbow 扫描。
- **新增**沿 wrist_angle 全范围的扫描（其余关节在 home），至少包括 shoulder–wrist_angle、elbow–wrist_angle 两对，含 / 不含 armature 两种情况。
- **新增** elbow × wrist_angle 平面上 shoulder–wrist_angle 的归一化耦合热力图（含 armature）。
- 在交付记录中用 1–2 句话解释：为什么沿 shoulder 扫描系数为常数，为什么 elbow–wrist_angle 不随 elbow 变化（可直接用于报告）。

### 4. E2：快速单关节轨迹（报告用）
- 保留现有的 1 s / 0.5 rad 情况；**新增**一档快速情况：五次多项式，峰值速度约 2 rad/s（关节速度上限），shoulder 和 elbow 各做一次。
- 新增一张**不含重力**的动态耦合力矩图（惯性耦合 M_ij q̈_j + 科氏 / 离心），对比 1 s 档和快速档，给出峰值和与力矩上限的比值。
- 检查**被驱动关节本身**所需的力矩：`M_ii q̈_i + 耦合 + 重力 + 阻尼 B·q̇ + frictionloss·sign(q̇)`，报告峰值以及与力矩上限的比值，并说明是否饱和。
- 图片和数据保存到 `outputs/m2/`，文件名与现有的区分开。

## 交付物

- 上述修改，以及 `tasks/M2_delivery_r2.md`。交付记录必须包含：改动文件清单；运行过的命令及输出摘要（`m2_validate.py`、`m2_coupling.py`、`pytest tests -q`、`check_env.py`、NaN 注入）；E1 / E2 的关键数值和 2–4 条结论；新的输出文件名列表；未完成或存疑的点。

## 完成标准

- 每一项的验收都满足；`pytest` 全部通过（包括 M0、M1、M2 原有的测试）；`check_env.py` 退出码 0；`m2_validate.py` 正常运行退出码 0。
- 做不到的项不得绕过或降低标准；如实写明，并作为失败上报。
