# 交接单 M1-2（新 Cursor 对话：按审批结论修改 M1）

> 原因：M1 实现对话（dispatch `ctx_4543034dbdf4`）的上下文用量已到 90.8%，超过 65%，按规则换新对话。

## 1. 现在要做什么，最后要交什么

按照用户已审批的 M1 审查结论，修改 M1 的实现。任务说明见 `tasks/M1_dispatch_r2.md`。最后要交：代码 / 模型 / 配置 / 测试的修改，以及交付记录 `tasks/M1_delivery_r2.md`。

## 2. 最新的要求，以及哪些旧要求已经作废

- 最新要求以 `tasks/M1_review_r2.md` 的“最终结论”一节和 `tasks/M1_dispatch_r2.md` 为准。原始规格见 `tasks/M1_dispatch.md`，下面列出的作废项除外。
- **作废**：`M1_dispatch.md` 第 4 步“在 `ee_center_site` 和 `left/gripper` 中选一个作为 TCP”。改为新增专用 site `left/tcp`。
- **作废**：`M1_dispatch.md` 第 3 步“打开时每侧至少留约 1 cm 余量”。用户已接受直径 4.50 cm、每侧余量 0.62 cm。
- **作废**：原成功判定只看“格内 + 静止 + N 步”。现在还要加上“已松手（两指与 Can 均无接触）”。
- **作废**：`fixed_1` 和 `fixed_2` 共用同一个目标格。现在目标格按 mode 配置：`fixed_2` 放到右 bin 中与原格对角的 −X−Y 格；`random` 和 `fixed_1` 仍用 +X+Y 格。
- 仍然有效：`M1_dispatch.md` 的硬性约束、固定参数表、完成标准中的数值门槛（抬起 ≥ 5 cm、保持 ≥ 3 s、滑移 < 5 mm、漂移 < 0.01 rad）。

## 3. 已确认的事项和依据

- M1 r1 的实现主体合格：模型参数、场景、IK、环境、两个冒烟测试都已达到门槛。依据：`tasks/M1_review_r1.md` 第 2 节，以及 `tasks/M1_recheck_r1.md` 中的 R1、R3、R5。
- `mj_step` 之后，派生量（xpos、site_xpos、xmat 等）对应的是积分前的状态；对同一 qpos 重新 `mj_forward`，TCP 会变化约 0.16 mm。依据：`tasks/M1_recheck_r1.md` N1。
- `_mesh_aabb_center` 直接使用了 mesh 参考系下的顶点，与 body 系相差约 0.35 mm。依据：`tasks/M1_recheck_r1.md` N2。
- 姿态误差和 Jacobian 的坐标系实现是正确的（差商验证误差 8.22e-11）。依据：`tasks/M1_recheck_r1.md` 对 F5 的回应。**这部分不要改。**
- agentview 取景时，右 bin 的 x 坐标超出画面约 0.10（归一化坐标）。依据：`tasks/M1_recheck_r1.md` 对 F4 的回应。
- 顶层模型必须放在 `aloha/` 目录下，因为 meshdir 是相对于主 XML 解析的。依据：`tasks/M1_delivery_r1.md`。

## 4. 还没验证的推测

- 推测：`left/tcp` 可以挂在 `left/gripper_base` 下，放在张开时两侧指垫碰撞球的中点；夹爪开合时这个中点只沿开合方向对称变化，中点位置基本不动。需要实测张开和闭合两种状态。
- 推测：把 agentview 后移或升高、再加大 fovy，就能同时拍下两个 bin。需要用投影计算来验证。

## 5. 试过不行的路和原因

- Can 的碰撞几何用方案 A（缩放后的 mesh）：3 s 内滑移 8.86 mm，超过门槛。现在用的是圆柱（方案 B），**不要改回去**。依据：`tasks/M1_delivery_r1.md`。
- 夹爪 ctrl = 0.002 在动力学中到不了（指尖在 q ≈ 0.008 时就已接触）。闭合指令用 0.015。

## 冲突

- 关于 TCP 应该位于抓取中心这一要求的出处，Claude 在 r1 中写的是 PLAN 第 3 节，Codex 指出实际出自 dispatch 第 4 步。Claude 已接受这一更正，结论不变。依据：`tasks/M1_review_r2.md`。不影响执行。

## 6. 下一步最小的一件事，以及做到什么样算合格

先新增 `left/tcp` site，并把所有使用 TCP 的地方切换过去。合格标准：张开时 `left/tcp` 到两指垫中点的距离 < 1 mm；`pytest` 全部通过；`scripts/m1_grasp_smoke.py` 不再使用任何偏移量，仍然输出 `GRASP_OK`。
