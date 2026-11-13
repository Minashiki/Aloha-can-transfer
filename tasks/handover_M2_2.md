# 交接单 M2-2（新 Cursor 对话：按审批结论修改 M2）

> 原因：M2 实现对话（dispatch `ctx_faca1086f0ab`）上下文用量 86.8% > 65%，按规则换新对话。

## 1. 现在要做什么，最后要交什么

按用户已审批的 M2 结论修改：N1、N2（必修）+ E1、E2（报告增强，用户已批准）。任务说明见 `tasks/M2_dispatch_r2.md`。交付：修改后的脚本 / 测试 + `outputs/m2/` 新图表 + `tasks/M2_delivery_r2.md`。

## 2. 最新的要求，以及哪些旧要求已经作废

- 以 `tasks/M2_review_r2.md`“最终结论”和 `tasks/M2_dispatch_r2.md` 为准；`tasks/M2_dispatch.md` 的硬性约束与“自写”定义仍然有效。
- **作废**：`M2_review_r1.md` 的 F3（集中 TCP 常量）已撤销，**不要改** `kin/fk.py` 的 `TCP_SITE` 和 `robot.py`。
- **作废**：F1“把 shoulder 扫描替换掉”。现在是**保留** shoulder 扫描（作为不变性证据），**另外增加** wrist_angle 扫描（E1）。
- 仍然有效：现有 1 s / 0.5 rad 单关节轨迹保留，E2 是在其基础上增加一档。

## 3. 已确认的事项和依据

- 自写 FK / 雅可比 / RNEA / M / C / g 正确，15 项交叉验证全部 PASS（最小余量约 2 个数量级）。依据：`tasks/M2_delivery_r1.md`、`tasks/M2_recheck_r1.md` 技术核对。**不要改动 `kin/fk.py`、`kin/spatial.py`、`dyn/rnea.py` 的计算逻辑。**
- `Tracker`（`scripts/m2_validate.py:71–87`）用 max 累计误差，NaN / Inf 会被丢弃，注入 NaN 后仍 PASS、退出码 0。依据：`tasks/M2_recheck_r1.md` N1。
- 腕部条件数使用 `sigma_max / max(sigma_min, 1e-16)`（`scripts/m2_validate.py:130–132`）；理论奇异点在 wrist_angle = 0。依据：`tasks/M2_recheck_r1.md` N2。
- 沿 shoulder 扫描时耦合系数为常数，沿 elbow 扫描时 elbow–wrist_angle 为常数，这是理论不变性（M_ij 只依赖 i 之后的关节角，shoulder 与 elbow 轴平行）。依据：`tasks/M2_review_r1.md` 第 3 节、Codex 确认。
- 1 s / 0.5 rad 轨迹下，峰值速度 0.94 rad/s，保持力矩以重力为主（elbow 4.32 N·m，其中惯性耦合 0.28、科氏 0.12）。依据：`tasks/M2_delivery_r1.md`。

## 4. 还没验证的推测

- 推测：峰值速度接近 2 rad/s 的轨迹（例如 0.5 rad 在约 0.47 s 内完成），动态耦合力矩会是 1 s 档的约 4–5 倍（与时间缩放的平方成正比），但仍远低于力矩上限。需要实测。
- 推测：被驱动关节本身在快速档下，`M_ii q̈ + 阻尼 + frictionloss + 重力` 可能接近 shoulder / elbow 的 20 N·m 上限。需要实测并报告。

## 5. 试过不行的路和原因

- Christoffel 用二阶差分、步长 1e-6 时，相对残差约 7e-8，过不了容差；现在用的是四阶、步长 1e-3，不要改回去。依据：`tasks/M2_delivery_r1.md`。

## 冲突

- F1 的定性：Claude r1 认为是缺陷，Codex 认为实现符合 spec、不是缺陷。最终按 Codex 的意见处理（保留 shoulder 扫描，另增 wrist_angle 扫描），依据 `tasks/M2_review_r2.md`。不影响执行。

## 6. 下一步最小的一件事，以及做到什么样算合格

先修 N1：让 `Tracker` 在累计前检查有限性。合格标准：`[nan] vs [1]`、`[1] vs [nan]`、`[inf] vs [inf]`、标量 NaN 四种情况都判为 FAIL；新增的 pytest 通过；正常运行 `m2_validate.py` 仍是 15 项 PASS、退出码 0。
