# M1 审查记录 r3（Claude 验收：按审批结论修改后）

- 对象：Cursor 新对话的 dispatch `ctx_e4238745e8ef`（交接单 `tasks/handover_M1_2.md`，任务说明 `tasks/M1_dispatch_r2.md`），交付记录 `tasks/M1_delivery_r2.md`
- 方式：只读复核，不再经过 Codex（WORKFLOW 第 2 节第 7 步）

## 合规
- PLAN、WORKFLOW、`.gitignore`、`environment.yml`、`scripts/check_env.py` 的哈希与派发前一致（`sha256sum -c` 五份都显示“成功”）。
- 没有 commit；改动范围只包括 `M1_dispatch_r2.md` 列出的文件。

## 复跑
| 命令 | 结果 |
|---|---|
| `pytest tests -q` | 20 passed（包括 M0 的测试） |
| `scripts/m1_grasp_smoke.py` | 局部滑移 2.78 mm（每个物理步采样），`ik_failures 0`，`GRASP_OK` |
| `scripts/m1_ik_reach.py` | 退出码 0（左 bin 和右 bin 4 个格子的网格全部收敛，随机 200/200） |
| `scripts/m1_gravity_hold.py` | 最大漂移 1.76e-4 rad |
| `scripts/check_env.py` | ALL CHECKS PASSED |
| 吞吐（一次性测量：`with AlohaCanEnv()`，`fixed_2`，每步渲染两路 96×96） | 约 680 env step/s（实测值）。每个物理步后执行 `mj_forward` 的开销可以接受 |

## 逐项验收（对照 M1_review_r2 的最终结论）
| 项 | 结果 | 依据 |
|---|---|---|
| 1 TCP | 通过 | `left/tcp` 位于 `aloha.xml:153`；张开时以及开口等于圆柱直径时，到指垫中点的距离都是 3.8e-7 m；配置和 `robot.py` 中旧的 TCP 定义都已删除；有对应测试 |
| 2 agentview | 通过 | 8 个顶角投影的绝对值最大为 0.861；256×256 的 reset 图和抓取图中，两个 bin、Can、夹爪都完整入画（人工查看过 `outputs/m1/agentview_reset_256.png`、`agentview_grasp.png`） |
| 3 目标格按 mode | 通过 | `fixed_2` 用 −X−Y 格，中心 (0.264, −0.170)；有测试 `test_target_cell_follows_mode` |
| 4 已松手 | 通过 | `require_release` 默认开启；“夹着 / 松手 / 格外”三种情况都有测试 |
| 5 IK 状态 | 通过 | `step` 返回 `(obs, success, info)`；三个脚本都打印 `ik_failures` |
| 6 状态同步 | 通过 | `env.py` 中 step 的每个物理步之后都执行 `_sync_derived()`，`get_obs` 也会同步；动态 FK 一致性测试的误差 < 1e-9 |
| 7 物体中心 | 通过 | 改为使用 `can_collision` 的 geom 位姿；两个朝向下与 `geom_xpos` 的差 < 1e-12 |
| 8 IK 脚本 | 通过 | 采样边界由 geom 推导（右 bin 扣除隔板，分成 4 个格子）；注入失败点后退出码为 1 |
| 9 滑移度量 | 通过 | 在 TCP 局部坐标系中计算，每个物理步采样；有刚体零滑移测试和 6 mm 注入检出测试 |
| 10 低优先级 | 通过 | 删除了重复的常量，测试中的期望值保留；新增 `with` 支持及对应测试 |

## 上下文用量
- Cursor（`ctx_e4238745e8ef`）：状态栏显示 **65.6%**，超过 65%。下一次任务必须开新对话。
- Codex（`ctx_debe96a23585` 系）：约 22.9%，已完成 2 次任务。

## 结论
**M1 验收通过，可以提交。**
