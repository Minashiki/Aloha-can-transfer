# M4 审查记录 r2（Claude 裁决与最终结论，待用户审批）

- 输入：`tasks/M4_review_r1.md`（Claude）、`tasks/M4_recheck_r1.md`（Codex，新对话 `ctx_67512818f8e5`）
- Codex 合规：复核前后 `git status` 只多了 `?? tasks/M4_recheck_r1.md`；`git diff 10d094b` 的哈希与复核前一致；`src/ scripts/ tests/ configs/ outputs/m3 outputs/m4` 下没有比交接单更新的文件；受保护文件 `sha256sum -c` 全部成功。**Codex 只写了 recheck 文件。**
- Codex 环境：`gpt-6.1-sol`、`reasoning_effort=medium`、`service_tier=default`（会话 jsonl）。

## 对 Codex 意见的逐条判断

| 项 | Codex | Claude 判断 | 依据 |
|---|---|---|---|
| 合规 / 主体 1–7 | 同意或部分同意（多为“无法独立重证历史”） | **采纳**（措辞层面） | 这些保留意见针对证据边界，不改变结论 |
| pytest 55 passed / 1 failed（CUDA 不可用） | 单列为环境限制 | **采纳为环境说明，不算回归** | 失败的是 `tests/test_smoke.py` 的 `torch.cuda.is_available()`。Codex 的沙箱没有 GPU；我在本机运行同一命令为 56 passed。验收时我会再在本机跑一次 |
| F1：归因不成立、补拆分列 | 同意 | **采纳** | — |
| F1：slow 档松手后位移 2.25–2.44 mm，不能统一说“约 1 mm” | 补充 | **采纳** | Codex 给出 E1 六组完整拆分表，我的 r1 只看了 medium / fast |
| F1：没有摩擦消融，不能写成唯一因果 | 补充 | **采纳，并要求做消融** | 在内存中把臂关节 `dof_frictionloss` 设为 0（不改模型文件）重跑两组 medium，代价很小，能把“吻合”变成因果证据 |
| F1：`open` 首步是斜坡开始，不是手指实际脱离 | 补充 | **采纳** | 拆分改用“实际脱离时刻”，同时保留斜坡开始时刻 |
| F2：现象 | 同意并独立复现 | **采纳** | Codex 的数值与我一致（B 段内 [0.09707, 0.10294]） |
| F2 修改①：丢弃被取代的缓存结点 | 同意方向；限幅时不能承诺精确到达；丢弃的结点不得再进入差分 | **采纳** | — |
| F2 修改②：我写的“`q_future = q_target` 即零速度、零加速度” | **反对**：中心差分给出 v=(target−prev)/(2T)、a=(prev−target)/T²，不是零 | **采纳 Codex 的反对，我的表述是错的** | 按公式代入 prev=0.10、target=0.12，v=0.2 rad/s、a=−8 rad/s²，Codex 正确。见下方裁决 |
| F2 修改③：测试范围扩大（连续 None、预告不一致、限幅段、reset） | 部分同意 | **采纳** | — |
| F3 / F4 / F5 | 同意 | **采纳** | — |
| N1：掉落判定用的是“任一指离开”，与声明的“两指都松开”不符 | 新问题（中） | **采纳，按声明修代码** | `planning/metrics.py:111–116` 中 `lost = carry & high & ~both`，`both = left & right`，确实是“任一指不接触”。只有两指都不接触才说明 Can 已脱离夹爪；单指短暂失接触在持罐中可能出现，不应判掉落。当前 56 条全成功，结果不受影响 |

**分歧项：无。** Codex 唯一的“反对”指出的是我在 r1 中的公式错误，我接受。

### F2 修改②的裁决（替代 r1 的表述）

流进入 preview 之后，某次调用没有 `next_action` 时，**按 `next_action = action`（把当前目标当作自己的下一目标）处理**，仍用中心差分：终点速度 (target−prev)/(2T)，不是零。理由：这正是规划器最后一个动作的做法（“最后一个动作重复自身”），E1 / E2 已经在用，语义统一；连续多次缺失时，第二次起 prev = target，终点速度和加速度自然归零，参考停在目标上。不另设“v1=a1=0 的停止段”。

## 最终结论（交用户审批）

M4 主体完成：E1 六组、E2 50/50 成功；E4 证实 preview 把滞后从 100 ms 降到 50 ms 且纹波不回来；腕部在本任务负载下够用。**需要 Cursor 修改后重跑**，修改清单如下（发给 Cursor 的 `ctx_e68be28bdd5e`，该对话第 2 次任务，用量 47.3%）：

1. **F1 放置误差**
   - 改正交付结论的归因：主要来源是静止时肩 / 肘停在摩擦带内的稳态误差，下落只在 fast / medium 档贡献约 0.6–1.1 mm，slow 档约 2.2–2.4 mm（以重算为准）。
   - `m4_analyze.py`（只读日志）每个 episode 新增：①`open` 斜坡开始时，以及②两指**实际**都脱离 Can 的第一个物理步，TCP 相对放置路点的误差（xy、z）、Can 相对格子中心的误差（xy、z）、肩 / 肘的 q−q_d；③从实际脱离到结束 Can 的 xy 位移（位移向量的模）。CSV 和 Markdown 都要有，E2 给分布（最小 / 中位数 / 最大）。
   - **摩擦消融**：E1 的 fixed_1 medium、fixed_2 medium 各重跑一次，只在内存中把 6 个臂关节的 `dof_frictionloss` 设为 0（不得修改 `aloha/` 文件），日志放 `outputs/m4/ablation_nofriction/`，与原结果对比上面这些量和最终放置误差。结论按消融结果写，不能证明的地方写“吻合”而不是“导致”。
   - 不改增益、不加积分、不在规划器里补偿这个误差。
2. **F2 参考模式切换**（`control/quintic.py`、`control/tracking.py`）
   - 有 `q_future` 时，本 tick 总以本次目标为终点；等待模式欠下的缓存结点视为被取代，直接丢弃，不播放、不进入任何后续差分；在 sample / 日志中记一个 `dropped_knot` 标记。
   - 流一旦用过 `q_future`，之后缺 `next_action` 的调用按“当前目标作为下一目标”处理（见上方裁决）；只有从未用过 `q_future` 的流（M3 用法）才走原等待模式，保证与 M3 逐步一致。`reset()` 清除这个状态。
   - 超速度上限时按 M3 的制动 / 限幅处理并记标记；文档不承诺限幅段精确一拍到达。
   - 测试：A（前几拍 None 后转 preview）切换后滞后回到 1 tick；B（匀速斜坡上 preview 中途缺一拍）q_d 单调、q̇_d 不换向；斜坡末尾连续 None 时参考停在目标上，并给出段内超调量；预告与实际下发不一致；限幅段标记；纯 None 流与 M3 逐步一致；reset 后状态清除。
3. **N1 掉落判定**：`lost` 改为两指都不接触（`~(left | right)`）连续 ≥10 步且 Can 仍高于格子中心 2 cm；保留“低于抓取高度 drop_z”的分支。补构造测试：单指失接触不算掉落，两指都失接触 ≥10 步算掉落。交付说明和配置注释与代码一致。
4. **F3 口径说明**：E1 / E2 的“TCP 对参考”列名或脚注写明“参考为规划器时间律，命令提前一拍，已抵消 preview 的 50 ms 滞后”；比较滞后只用 E4；注明 M5 用规划器示范时也带这一拍提前量。不改规划代码。
5. **F5**：交付记录补 IK 耗时（每 tick 去重后的中位数 / 最大值，当前目标和下一目标分开）。
6. **F4 与 M3 回归**：给 `scripts/m3_run.py`（以及 `m3_analyze.py`，如需要）加输出目录参数，默认值仍是 `outputs/m3`，行为不变；这次 M3 回归写到 `outputs/m4/m3_regression/`，**不得再写 `outputs/m3/`**。
7. **重跑与回归**：E1、E2、E4、摩擦消融、M3 回归；`pytest tests -q`、`check_env.py`、`m1_grasp_smoke.py`。核对 E1 / E2 / E4 原有指标与 r1 一致（这些实验不经过 F2 改动的分支，应完全不变；若有变化说明原因）。交付记录写 `tasks/M4_delivery_r2.md`，列出改动文件、命令与输出摘要、上面每一项的处理结果和存疑点。

约束不变：只用 `AIControl` 环境；不改 `docs/`、`.gitignore`、`environment.yml`、`check_env.py`、`aloha/`、`assets/scene/`、M2 计算模块、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`；不改增益、上限、阈值、采样范围、夹爪开合量；不做 git 写操作；不启动其他 agent。

## 上下文用量

- Cursor `ctx_e68be28bdd5e`：47.3%，已完成 1 次任务
- Codex `ctx_67512818f8e5`：69,750 / 475,000 ≈ 14.7%，已完成 1 次任务
