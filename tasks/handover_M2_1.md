# 交接单 M2-1（新 Cursor 对话）

> 原因：M1 修改对话（dispatch `ctx_e4238745e8ef`）的上下文用量已到 65.6%，超过 65%，按规则换新对话。

## 1. 现在要做什么，最后要交什么

实现里程碑 M2“运动学 / 动力学建模与验证”，任务说明见 `tasks/M2_dispatch.md`。最后交付：自写的运动学和动力学模块、交叉验证与耦合分析脚本、测试，以及交付记录 `tasks/M2_delivery_r1.md`。

## 2. 最新的要求，以及哪些旧要求已经作废

- 以 `tasks/M2_dispatch.md` 为准。背景可读 `docs/PLAN.md` 第 5 节和第 10 节 M2（只读，不得修改 `docs/PLAN.md`、`docs/WORKFLOW.md`）。
- 已作废，**不要再引用**：TCP 是 `ee_center_site` 或 `left/gripper`、`tcp_to_grasp_center`、`can_center_local`、`robot.TCP_SITE`。现在唯一的 TCP 是 `left/tcp`。
- 已作废：Menagerie 的原始阻尼（20 / 18.49 等）。现在的阻尼以 `aloha/aloha.xml` 为准。
- M1 中 `.gitignore` 的修改授权已经作废，M2 不得修改 `.gitignore`。

## 3. 已确认的事项和依据

- 机器人本体模型是 `aloha/aloha.xml`，不含执行器和场景；顶层模型是 `aloha/aloha_motor.xml` / `aloha_position.xml`（含 Can 的 freejoint）。依据：提交 `c6b6910`，`src/aloha_pp/robot.py`（`robot_xml()` / `scene_xml()`）。
- 机器人有 6 个臂关节，另有 2 个手指滑动关节（`left/left_finger`、`left/right_finger`），由一条 equality 约束 1:1 耦合。依据：`aloha/aloha.xml`，`neq == 1`。
- 关节参数：armature 为 shoulder 0.395、elbow 0.383、forearm_roll 0.14、wrist_angle 0.008，waist 和 wrist_rotate 没有；frictionloss 为 shoulder 2.0、elbow 1.15；阻尼为 1.0 / 2.0 / 2.0 / 0.8 / 0.8 / 0.3。依据：`aloha/aloha.xml` 中的 default class，以及 `tests/test_m1_model.py`。
- 姿态误差和雅可比的坐标系约定：`mj_jacSite` 的旋转行在世界坐标系中；`kin/ik.py` 中的姿态误差实现已验证正确。依据：`tasks/M1_recheck_r1.md` 对 F5 的回应。
- `mj_step` 之后，派生量对应的是积分前的状态，读取前要先 `mj_forward` / `mj_kinematics`。依据：`tasks/M1_recheck_r1.md` 中的 N1。
- 静态重力力矩：shoulder 和 elbow 的重力力矩约为力矩上限的 0.19–0.24。依据：`tasks/M1_delivery_r2.md` 中 `m1_gravity_hold.py` 的输出。

## 4. 还没验证的推测

- 推测：只要把手指关节固定在某个 q、速度设为 0，那么 6×6 臂关节块的 `mj_fullM` 和 `qfrc_bias`，就等于“把手指当作刚性附着在 gripper 上”的 6 自由度模型的结果。需要实测。
- 推测：MuJoCo 的 `body_inertia` / `body_iquat` 是相对于 body 质心的主惯量；自写 RNEA 从 `MjModel` 读取这些参数，就能与 MuJoCo 的结果对齐到 1e-10 量级。

## 5. 试过不行的路和原因

- M2 是新模块，暂无。M1 的经验：直接把 mesh 顶点当作 body 局部坐标会产生约 0.35 mm 的偏差（`tasks/M1_recheck_r1.md` N2）。读取几何 / 惯性参数时，一定要注意它们所在的参考系。

## 冲突

无。

## 6. 下一步最小的一件事，以及做到什么样算合格

先实现自写的正运动学（FK），并与 MuJoCo 对比。合格标准：在 ≥ 100 个随机构型下，各 link 的位姿和 `left/tcp` 的位姿与 `mj_kinematics` 的结果相比，位置误差 < 1e-10 m，旋转误差 < 1e-10 rad。
