# M4 审查记录 r3（Claude 验收：按审批结论修改后）

- 对象：Cursor follow-up dispatch `ctx_6ff391b6b603`（同一对话 `ctx_e68be28bdd5e`）；交付记录 `tasks/M4_delivery_r2.md`
- 方式：只读复核，不再经过 Codex。对照基准：修改前我保存的 r1 diff、r1 新增文件副本和 r1 `summary.csv`（Claude scratchpad）

## 合规

- 受保护文件（PLAN、WORKFLOW、`.gitignore`、`environment.yml`、`check_env.py`、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`、M2 计算模块、`aloha/*.xml`）`sha256sum -c` 全部成功。
- HEAD 仍是 `10d094b`，没有 commit。
- **`outputs/m3/` 没有被写**：目录下没有比 `M4_review_r2.md` 更新的文件。
- r1→r2 的改动只落在审批清单涉及的文件：`quintic.py`、`tracking.py`、`metrics.py`、`planner.py`（只新增 `place_tcp` 记录，3 行）、`m4_task.yaml`（注释）、`m4_run.py`、`m4_analyze.py`、`m3_run.py` / `m3_analyze.py`（`--out`，默认仍为 `outputs/m3`）、两份 M4 测试。`gains.py`、`schedule.py` 未变。

## 逐项验收（对照 M4_review_r2 最终结论）

| 项 | 结果 | 依据 |
|---|---|---|
| 1 F1 放置误差 | 通过 | 归因已改正；分析脚本新增 `open_*`（斜坡开始）和 `detach_*`（两指实际都脱离）两组拆分列及脱离后 xy 位移，E2 给出最小 / 中位数 / 最大。摩擦消融只在内存中置零（`m4_run.py:176`），`aloha/` 哈希不变；我读消融日志：松手时肩 / 肘 q−q_d 由 −6.7 / −4.5 mrad 降到 0.15 / 0.13 mrad，两组都成功，最终 xy 由 4.6–4.7 mm 降到约 0.5 mm。结论措辞区分了摩擦（主要来源，有消融支撑）和下落 / 重新瞄准（剩余约 0.5 mm，未拆分），slow 档脱离后位移 2.5–2.7 mm 如实写出 |
| 2 F2 切换语义 | 通过 | 我用 r1 审查时的同一脚本复测：A（前 3 拍 None 后转 preview）切换拍丢弃缓存结点（`dropped_knot` 仅该拍为 1），之后每拍末 q_d 等于本拍目标，滞后回到 1 tick；B（preview 中途缺一拍）q_d 全程单调、q̇_d 最小 0.0107 rad/s 不换向，之后仍是 1 tick。缺 `next_action` 按 `q_future = q_target` 的中心差分处理，交付说明写明终点速度不是零；连续缺拍时段内超调 1.65 mrad（单元测试），E1 / E2 中不出现。纯 None 流与 M3 逐步一致（测试 + M3 回归） |
| 3 N1 掉落判定 | 通过 | `metrics.py` 改为 `~(left \| right)`；单指失接触不算、两指 9 步不算 10 步算，均有测试；配置注释已同步 |
| 4 F3 口径 | 通过 | `outputs/m4/summary.md` 主表前后有脚注：E1 / E2 的参考是规划器时间律，命令提前一拍抵消 50 ms 滞后，比较滞后只用 E4，M5 示范同样带此提前量 |
| 5 F5 IK 耗时 | 通过 | 交付记录给出按 tick 去重的中位数 / 最大值（88 / 258 µs，下一目标 74 / 240 µs），与我在 r1 统计的量级一致（墙钟差异已说明） |
| 6 F4 与 M3 回归 | 通过 | M3 回归写到 `outputs/m4/m3_regression/`，汇总表与 M3 交付 40 行逐格一致（交付记录） |
| 7 重跑核对 | 通过 | 我对比 r1 与 r2 的 `outputs/m4/summary.csv`：69 行、65 个共有列、3112 个数值格差异为 0；新增 17 列为释放拆分等。E1 / E2 全部日志 `dropped_knot` 为 0 |
| 回归 | 通过 | 我复跑 `conda run -n AIControl python -m pytest tests -q`：63 passed。`check_env` 通过、`GRASP_OK`（交付记录） |

## 记录到后续里程碑的事项（不阻塞 M4）

- **放置误差约 4.5 mm 主要来自肩 / 肘摩擦带内的稳态误差**（frictionloss / Kp），C3 不用积分项是有意的设计。报告中与 M3 C4 接触实验一起讨论；M6 / M7 的策略如果需要更准的放置，可以靠闭环观测修正，而不是改底层控制器。
- **M6 使用参考接口时每拍都传 `next_action`**（动作块内用下一步，执行段最后一步用第 9 步）。漏传已经不会退回 100 ms，但会走“重复当前目标”的停止式差分。
- **规划器命令提前一拍**：M5 用规划器生成保底示范时，动作时序与重定向示范不同，数据集中要记录来源。
- `random_can` 只按 Can 半径内缩，E2 没有抽到贴壁失败；不扩大采样范围，报告中如实说明 50/50 的置信区间 [0.929, 1.000]。
- 摩擦消融只做了两组 medium；slow 档脱离后 2.5–2.7 mm 的位移未在零摩擦下重测。

## 上下文用量

- Cursor（`ctx_e68be28bdd5e` 系）：**91%**，本对话已完成 2 次任务。超过 65%，**下一个里程碑必须新开 Cursor 对话并写交接单**。
- Codex（`ctx_67512818f8e5`）：约 14.7%，已完成 1 次任务。

## 结论

**M4 验收通过，可以提交。**
