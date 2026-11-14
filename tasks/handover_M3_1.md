# 交接单 M3-1（新 Cursor 对话）

> 原因：M2 修改对话（`ctx_d3e266463be4`）用量 56.7%，按规则仍可继续，但 M3 是目前最大的里程碑，在该对话中继续做很可能中途超过 65%。为避免任务中途交接，直接开新对话。

## 1. 现在要做什么，最后要交什么

实现里程碑 M3“控制器与跟踪实验”，任务说明见 `tasks/M3_dispatch.md`。最后交付：插值器、控制器 B0 / B1 / C1–C4、批量实验与统一日志、绘图 / 统计脚本、测试，以及交付记录 `tasks/M3_delivery_r1.md`。

## 2. 最新的要求，以及哪些旧要求已经作废

- 以 `tasks/M3_dispatch.md` 为准；背景见 `docs/PLAN.md` 第 2、6 节和第 10 节 M3（只读，不得修改 PLAN / WORKFLOW）。
- 已作废：Menagerie 原始阻尼（现在以 `aloha/aloha.xml` 为准）；TCP `ee_center_site` / `left/gripper` / `tcp_to_grasp_center`（现在唯一的 TCP 是 `left/tcp`）；motor 执行器的 gear 150 / 200（现在 `gear=1`）。
- PLAN 第 6 节写的速度档“例如 2 / 4 / 8 s”只是示例，**按 `M3_dispatch.md` 的速度档定义执行**。
- O1（OSC）为可选项，M3 **不做**。

## 3. 已确认的事项和依据

- 模型：motor 版 `aloha/aloha_motor.xml`（`gear=1`，ctrl 即关节力矩，ctrlrange = 力矩上限）；position 版 `aloha/aloha_position.xml`（Menagerie kp + kv = 原阻尼值）。关节阻尼、`actuatorfrcrange`、armature、frictionloss 见 `aloha/aloha.xml`。依据：提交 `c6b6910`，`tests/test_m1_model.py`。
- B1 的重力补偿开关：`src/aloha_pp/sim/gravity.py` 的 `apply_gravity_compensation`（body `gravcomp` + joint `actuatorgravcomp`，补偿力计入 `actuatorfrcrange`）。依据：`tasks/M1_delivery_r1.md`“重力补偿开关”。
- 环境：`src/aloha_pp/sim/env.py` 的 `AlohaCanEnv`。`step(action, on_substep=None)` 返回 `(obs, success, info)`；每个动作执行 25 个物理步；物理步之后需要 `_sync_derived()` 才能读取派生量。当前 `LowLevelController` 接口只有 `reset` / `apply`（每个 20 Hz 动作调用一次）。依据：`src/aloha_pp/sim/controller.py`、`env.py:175–192`。
- 自写动力学 `aloha_pp.dyn.rnea`（M、g、C q̇）与 MuJoCo 一致（误差 ≤ 1e-14）。PLAN 规定：**控制循环中使用 MuJoCo 的 `qfrc_bias` / `mj_fullM`**，自写 RNEA 只用于验证和报告。依据：`tasks/M2_delivery_r1.md`、PLAN 第 5 节。
- `qfrc_passive = −B q̇`（阻尼）；frictionloss 在 `qfrc_constraint` 中。依据：`tasks/M2_delivery_r1.md` 摩擦核对。
- 速度达到 2 rad/s 的单关节五次轨迹：主动关节峰值力矩为上限的 0.54（shoulder）和 0.37（elbow），不饱和；动态耦合最大占上限的 0.070。依据：`tasks/M2_delivery_r2.md` E2。
- IK：`aloha_pp.kin.ik.DlsIk`，当前配置容差为 1 mm / 0.01 rad；两个 bin 的网格与随机样本全部收敛。依据：`tasks/M1_delivery_r2.md`。
- 两组固定位置：`fixed_1`（放置格 +X+Y）、`fixed_2`（放置格 −X−Y），见 `configs/m1_pick_place.yaml`。

## 4. 还没验证的推测

- 推测：B0（position 版，无重力补偿）在 shoulder / elbow 上存在明显的稳态下垂误差（M1 抓取保持时观察到手臂约 2 mm/s 的慢速“爬升”）。需要在 M3 中量化。
- 推测：20 Hz 目标 + 关节空间插值会引入约一个动作周期（50 ms）的跟踪滞后；相对“原始参考轨迹”的 TCP 误差在快速档会明显大于相对 q_d 的跟踪误差。需要实测并分开报告。
- 推测：把 IK 容差收紧到 1e-4 m / 1e-3 rad 后，单次求解耗时仍可接受（20 Hz 下远小于 50 ms）。需要实测。

## 5. 试过不行的路和原因

- 读取 `xpos` / `site_xpos` 等派生量时，如果不先同步就直接使用，会滞后一个物理步（M1 N1）。新的控制器在每个物理步计算力矩时，同样要注意这一点（`mj_step` 之后，下一次 `mj_step` 之前读取的 `qfrc_bias`、`qM` 对应的是哪个状态）。

## 冲突

无。

## 6. 下一步最小的一件事，以及做到什么样算合格

先扩展底层控制接口，使控制器能在每个物理步计算力矩，同时保持 M1 的 `PositionIkController` 行为不变。合格标准：M0–M2 的全部测试仍然通过；`scripts/m1_grasp_smoke.py` 仍输出 `GRASP_OK`。
