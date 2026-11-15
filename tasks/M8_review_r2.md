# M8 审查 r2（Claude 对 Codex 复核的裁决）

对象：`tasks/M8_recheck_r1.md`（Codex dispatch `ctx_4f42b0fa9e7d`，新对话，模型 `gpt-6.1-sol`、effort medium、service_tier default，已在 rollout 中核实）。

## 1. Codex 合规

- 派发前后对 `report/`、`scripts/m8_figures.py`、`tasks/M8_dispatch.md`、`tasks/M8_review_r1.md`、PLAN、WORKFLOW 做了 sha256 对比，结果一致；`report/` 和 `outputs/` 的文件清单也一致。
- `git status` 只多了 `tasks/M8_recheck_r1.md`。
- 这是该 Codex 对话的第 1 次任务。

## 2. 逐条裁决

| # | 意见 | 裁决 | 依据 | 处理 |
|---|---|---|---|---|
| K1 | SAC log π 的口径是归一化动作 $a=w/1.5$，不是 $w$ | **采纳** | `rl/nets.py:85` 没有减去 $\log 1.5$；`tasks/M7_recheck_r1.md` 已记录此口径 | §8.4 改为按 $a$ 的密度，注明与 $w$ 的密度相差 $128\log 1.5$，并说明目标熵和日志都用此口径；源代码不改 |
| K2 | +4.7 / 步是短程试验的值 | **采纳** | 我从正式日志复算 $-\alpha\log\pi$：12.2k 转移处 73.94，14.6k 处 10.74，100k 处 0.0069 | §8.6 改用正式日志的数值，因果表述标为“解释（未单独验证）” |
| K3 | “单调下降”不准确 | **采纳 → 用户否决** | 日志中 BC loss 和 Q 都有回升 | §7.5、§8.6 改为“总体下降、伴有波动”；BC 末值改为约 0.0020 |
| K4 | 16/16 的次数多算一次 | **采纳** | 全程 39 次，首次之后 38 次 | §8.6 已改 |
| K5 | “保证在流形上”“低维”表述过强 | **采纳 → 用户否决** | 噪声与动作块同为 128 维；去噪过程没有可行性保证 | §8.1 已改写 |
| K6 | 不能把摩擦带推广为所有方法的精度下限 | **采纳 → 用户否决** | 摩擦消融只覆盖规划器 E1 medium；DSRL last 的放置误差为 8.1 mm | §9.1 已限定表述范围 |
| K7 | C3 的逐关节二阶近似需要前提条件 | **采纳 → 用户否决** | `control/laws.py` 中的反馈项是 $K_p e + K_d\dot e$，增益按 home 构型的对角质量计算 | §4.2 补写三个前提条件，以及等效频率 $\sqrt{K_p/M_{ii}(q)}$ |
| K8 | 图 8-1 的坐标轴符号 | **采纳** | 与 K1 同源 | `scripts/m8_figures.py` 的坐标轴改为 `log π(a|f)（a = w/1.5）`，已重新生成插图 |

**分歧项：无。**

### 用户最终决定（2026-11-15）

只实施 K1、K2、K4、K8；**K3、K5、K6、K7 不写入报告**。已按用户要求把这四条对应的修改改回原文：

- K3：§7.5“训练损失单调下降到 0.0021”、§8.6 Q“之后单调下降”恢复原文。§8.6 中 K2 带来的熵项数值保留。
- K5：§8.1 恢复“好处一 / 好处二”原文。
- K6：§9.1 恢复“底层决定了精度的下限”原文。
- K7：§4.2 恢复“即各关节近似解耦为 …”原文。

## 3. 修改后核对

- PLAN / WORKFLOW 哈希不变；`outputs/` 的 1236 个文件清单与开工前一致。
- 报告中所有 figures / videos 链接都存在。
- 重跑 `scripts/m8_figures.py`：44 张图、20 段视频，打印的统计数值与第一次运行相同。
- 已跟踪文件没有任何改动。待提交的范围：`report/.gitignore`、`report/report.md`、`scripts/m8_figures.py`、`tasks/M8_dispatch.md`、`tasks/M8_review_r1.md`、`tasks/M8_recheck_r1.md`、`tasks/M8_review_r2.md`。

## 4. 交用户审批

8 条意见都是文字或图轴层面的修正，已经改在 Claude 自己新建的文件中，没有改动任何已有的代码或结果。请用户审阅报告，同意后提交（不 push）。
