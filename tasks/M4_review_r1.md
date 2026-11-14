# M4 审查记录 r1（Claude）

- 对象：Cursor dispatch `ctx_e68be28bdd5e`（新对话，按 `tasks/handover_M4_1.md` 启动）；交付记录 `tasks/M4_delivery_r1.md`
- 基线：`10d094b`（M3）。审查以 `git diff 10d094b` 加未跟踪新文件为准
- 方式：只读。一次性验证脚本放在 Claude 的 scratchpad，不落盘到项目目录

## 合规

| 项 | 结果 | 依据 |
|---|---|---|
| 受保护文件 | 通过 | 派发前对 PLAN、WORKFLOW、`.gitignore`、`environment.yml`、`check_env.py`、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`、M2 计算模块、`aloha/*.xml` 记录了 sha256；交付后 `sha256sum -c` 全部成功。`.gitignore` 的未提交改动是派发前就有的，哈希没变 |
| git | 通过 | HEAD 仍是 `10d094b`，没有 commit |
| 环境 | 通过 | 交付记录中的命令都带 `conda run -n AIControl`；我用同一环境复跑 `pytest tests -q`：56 passed |
| 不改增益 / 上限 / 阈值 / 采样范围 / 夹爪开合量 | 通过 | `configs/m4_task.yaml` 只新增等待、斜坡、放置间隙等；`gains.py` 在默认 ω 下直接返回 M3 数组，有逐元素测试 |
| “M3 重跑结果放到单独目录，不要覆盖 `outputs/m3/`” | **未遵守** | 见 F4 |

## 已核实的交付主体

- **未来目标入口**：`JointStream.push(q_target, q_future=None)`、`TrackingController.set_target(..., next_action=None)`、`AlohaCanEnv.step(..., next_action=None)`。`next_action is None` 时 env 不传这个参数，M3 路径逐字未变。有 `q_future` 时本 tick 以当前目标为终点，终点速度 / 加速度用中心差分（`quintic.py` `_push_preview` / `_open_central`）。我用一条匀速斜坡验证：纯 preview 时每个 tick 末的 q_d 等于本 tick 的目标，滞后 1 tick。
- **夹爪跟随 `action[7]`**：`tracking.py` 用 `np.clip(action[7], ctrlrange)`；M3 动作夹爪恒为 `gripper_open`，M3 回归一致。
- **E4（滞后 / 纹波）**：等待模式六行与 M3 交付表一致（例如 fixed_1 fast 肩饱和 4.95%、力矩峰值 26.6 Nm）。preview 下相对原始参考的 TCP RMSE 降到约一半到六成，肩 q̈_d RMS 比值 1.002–1.009，medium 不饱和。**推测“滞后降到 50 ms 后纹波不回来”成立。**
- **E1 / E2**：六组固定位置和 50 个随机种子全部成功；E2 Wilson 95% 区间 [0.929, 1.000]，我按公式复算一致。E1、E2 全部日志 `ik_success` 为真。
- **IK 耗时**（交付记录没写，我从日志统计）：E1+E2 每 tick 当前目标 IK 中位数 77 µs、最大 0.22 ms；下一目标 IK 中位数 72 µs、最大 0.19 ms；两次合计最大 0.30 ms，远小于 50 ms。
- **腕部**：持 Can 段 wrist_angle 误差峰值 ≤ 7.2 mrad、力矩 ≤ 19% 上限；wrist_rotate ≤ 3.5 mrad、≤ 15%。**推测“腕部增益在持 Can 时够用”在本任务（11.5 g、居中夹持）下成立**，不外推到更重的负载。
- **测试与回归**：pytest 56 passed（我复跑）；`check_env` 通过、`GRASP_OK`、M3 汇总表 `cell_diffs 0`（交付记录）。

## 问题

### F1（中）放置误差的归因是错的

交付结论 1 写：“放置 xy 误差 3.8–5.8 mm……这几毫米主要来自松手后大约 6 mm 的下落。”

我按日志在 `open` 状态第一个物理步拆分（E1 fixed_1 medium，其余组相同量级）：

| 量 | 数值 |
|---|---|
| 松手时 TCP − 参考 | x +2.2 mm，y −0.5 mm，z +3.5 mm |
| 松手时 FK(q_d) − 参考 | ≈ 0（≤ 0.02 mm） |
| 松手时 q − q_d | 肩 −6.7 mrad，肘 −4.5 mrad，其余 ≈ 0 |
| 松手时 Can − 格子中心 | x +3.5 mm，z +9.6 mm（计划是 +6 mm） |
| 结束时 Can − 格子中心 | x +4.6 mm，z ≈ 0 |

结论：松手前 Can 在 x 上已经偏了 3.5 mm，下落只再加约 1 mm。误差主要来自**静止时肩、肘的稳态跟踪误差**。这和 frictionloss / Kp 的上界吻合：肩 2.0 / 259.5 = 7.7 mrad，肘 1.15 / 212.3 = 5.4 mrad（`aloha/aloha.xml:32,36`，`configs/m3_track.yaml`）。C3 没有积分项，静止时 PD 力矩小于静摩擦就停在摩擦带里。E2 的 50 个种子都落在 4.5 mm 左右，也说明这是系统误差，不是随机下落。“开合时刻 TCP 误差 ≈ 4 mm”和 Can 比计划高 3.5 mm 也是同一个原因。

**要求**：
1. 改正交付记录结论 1 的归因，用上表这种拆分作为依据。
2. `m4_analyze.py`（只读日志）对每个 episode 增加三列：松手时 TCP 相对放置路点的误差（xy、z）、松手时 Can 相对格子中心的 xy 误差、松手后 Can 的 xy 位移，汇总表和 Markdown 都要有。
3. 不改控制器、不加积分、不改规划器去补偿这个误差。这是“为什么不用积分项”（M3 C4 实验）的另一面，留给报告讨论。

### F2（中）参考模式切换后滞后不恢复，中途缺一次 `next_action` 会来回抖一下

我在 scratchpad 里用一条匀速斜坡（每 tick 0.02 rad，0.4 rad/s）验证 `JointStream`：

- **A：先等待模式，再切到 preview**（前 3 个 tick 不给未来目标，之后一直给）：之后每个 tick 末的 q_d 一直落后目标 2 个 tick，**永远停在 100 ms**。原因：`_push_preview` 中缓存结点与本次目标不同时，先播缓存结点，再把本次目标放进缓存；下一 tick 又是这样，缓存永远不空。
- **B：preview 中途有一个 tick 不给 `next_action`**（tick 5）：这个 tick 走等待模式的“保持段”，起止都是 q=0.1、q̇=0.4，结果 q_d 在 0.0971–0.1029 之间先前冲再退回（±2.9 mrad，q̇ 换向两次）；之后也一直停在 100 ms。

C¹ 连续性确实成立，交付记录说的“不丢 / 不重复结点”也成立，但没有写出这两个后果。M4 规划器一直传 `next_action`，所以 E1 / E2 结果不受影响；但 M6 只要有一次没传（例如 episode 开头或某个分支漏传），之后整段都会退回 100 ms，而且没有任何标记。

dispatch 里“不得丢结点”这条是我写的，它和“滞后能回到 50 ms”不能同时满足。**我修改要求如下**：
1. 有 `q_future` 时，本 tick 总是以**本次目标**为终点。等待模式欠下的缓存结点视为已被更新的指令取代，直接丢弃（不播放），并在 sample / 日志里留一个“丢弃了缓存结点”的标记。起点仍是当前 q、q̇，保持 C¹；如果这段超速度上限，按 M3 的制动 / 限幅处理并记速度超限标记。
2. 流已经处在 preview 后，某个 tick 没给 `next_action`：按 `q_future = q_target` 处理（以零速度、零加速度的中心差分停在本次目标上，滞后仍是 1 tick），不进入“保持 + 缓存”。只有从未给过 `next_action` 的流（M3 的用法）才走原来的等待模式，保证 M3 逐步一致。
3. 测试：A、B 两种切换之后滞后都回到 1 tick；B 的那个 tick 里 q̇_d 不换向（匀速斜坡上 q_d 单调）；纯 None 流与 M3 逐步一致；M3 回归表仍逐格一致；E1 / E2 / E4 的数值因为不经过这些分支，应当不变（重跑核对）。

### F3（低）E1 / E2 的“TCP 对原始参考”和 E4 / M3 口径不同，交付记录没有说明

- E4（M3 路径）：tick k 下发的是 t = kT 的位姿（`reference.pose_at(k·dt)`），原始参考是同一条时间律，preview 仍有 50 ms 滞后，所以 fast 档 TCP 对参考是 19.3 mm。
- 规划器（`planning/schedule.py` `_motion`）：tick k 下发的是 `blend((k+1)/n)`，即这个 tick **结束时**的位姿，命令比时间律提前一拍；`Tick.dense` 的原始参考没有平移。preview 又恰好在 tick 结束时到达，所以 50 ms 滞后被提前量抵消，E1 的 TCP 对参考（≈3 mm）几乎等于 TCP 对 FK(q_d)。

规划器知道整条轨迹，提前一拍下发是合理的前馈，不需要改代码。但这个数和 E4 / M3 的数不能直接比较。**要求**：交付记录和汇总表的列名 / 脚注写明“E1 / E2 的参考是规划器时间律，命令提前一拍，已抵消 preview 的 50 ms 滞后”；结论里比较滞后的地方只用 E4。另外记一句：M5 用规划器生成的示范同样带这一拍提前量。

### F4（低，流程）M3 回归覆盖了 `outputs/m3/`

dispatch 要求“重跑结果放到单独目录，不要覆盖 `outputs/m3/`”。交付记录如实说明了 `m3_run.py` 输出路径写死、同名日志和两条视频被覆盖，之后复制到 `outputs/m4/m3_regression/`。汇总表逐格相同，`r1_baseline.json` 还在，实际损失很小，但这是没有遵守 spec。**要求**：不需要改代码；以后遇到输出路径写死时，先把原目录复制备份再跑，或者在交付前提出。记录在案。

### F5（低）交付记录缺 IK 耗时

dispatch 要求“单次耗时照常统计”。日志里有 `ik_solve_s` / `ik_next_solve_s`，交付记录没给数。**要求**：补上中位数 / 最大值（我统计的结果见上，供核对）。

## 不构成问题的点（已核对）

- 规划器读真值 Can 位姿：dispatch 允许，交付记录已说明它是对照组。
- yaw 选择（`choose_yaw`，开口朝余量大的方向，放置沿用同一 yaw）：理由合理；E2 50 个种子 39 个 yaw=0、11 个 yaw=π/2，全部成功。贴壁失败的可能仍在（`random_can` 只按 Can 半径内缩），交付记录已列为存疑，不扩大采样范围。
- fast 档：任务路径 fast 不饱和（峰值 16.8 Nm），E4 的 M3 路径 fast 仍饱和约 5%。两者路径和时长不同，不矛盾；“只限速度不足以避免力矩饱和”按 E4 报告。
- 开合等待 = 1 个滞后 tick + `round(0.25/0.05)` = 6 tick，满足“滞后 + 稳定时间”。因为 F3 的提前量，那 1 个滞后 tick 实际上多余，但无害，不要求改。

## 上下文用量

- Cursor `ctx_e68be28bdd5e`：47.3%，本对话已完成 1 次任务。修改轮可以继续用它（第 2 次）。
- Codex：上一条链（`ctx_25483edd438e`）已完成 4 次复核，超过“3 次即换”的规则，M4 复核新开对话，附交接单 `tasks/handover_M4_2.md`。

## 结论（待 Codex 复核）

M4 主体完成，E1 / E2 / E4 结果可信。需要修改：F1（归因与分析列）、F2（切换语义），F3、F5 补说明，F4 记录在案。修改后由 Cursor 重跑 E1 / E2 / E4 和 M3 回归。
