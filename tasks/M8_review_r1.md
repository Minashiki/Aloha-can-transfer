# M8 审查 r1（Claude 自检记录）

对象：Claude 撰写的 `report/report.md`、`report/.gitignore`、`scripts/m8_figures.py`（用户 2026-11-15 批准由 Claude 撰写，不派 Cursor）。计划见 `tasks/M8_dispatch.md`。

## 1. 合规

| 项 | 结果 |
|---|---|
| HEAD | `e4ac982`，未提交 |
| PLAN / WORKFLOW 哈希 | 与开工前快照一致（`a90182e3…54d9`、`09f4d845…803b`） |
| `outputs/` | 1236 个文件的路径 / 大小 / mtime 与开工前清单逐项相同 |
| 现有文件 | 未修改任何已跟踪文件（`git status` 只有 4 个未跟踪项：`report/.gitignore`、`report/report.md`、`scripts/m8_figures.py`、`tasks/M8_dispatch.md`） |
| 插图 / 视频 | `report/figures/`（44 张）、`report/videos/`（20 段，约 1.6 MB）由 `report/.gitignore` 排除，`git check-ignore` 已确认 |
| 环境 | 全部命令在 `conda run -n AIControl` 中执行，未安装任何包 |

## 2. 数值来源核对

| 内容 | 依据 |
|---|---|
| M2 验证表 | 重跑 `scripts/m2_validate.py`，`OUT` 在内存中改到 scratchpad（不写 `outputs/m2`），15 项与 `tasks/M2_delivery_r2.md` 一致 |
| M3 跟踪表 | `outputs/m3/summary.csv`、`tasks/M3_delivery_r2.md` |
| M4 E1 / E2 / E4 | `outputs/m4/summary.csv`（脚本读取统计）、`tasks/M4_review_r3.md` |
| M5 | `tasks/M5_review_r4.md`、`configs/m5_retarget.yaml`、`src/aloha_pp/data/README.md` |
| M6 | `configs/m6_dp.yaml`、`outputs/m6/train/train_meta.json`、`train_log.jsonl`、`eval/eval.json` |
| M7 | `configs/m7_dsrl.yaml`、`outputs/m7/train/{train_log.jsonl,selected.json,timing.json}`、`outputs/m7/compare/compare.{md,json}`、各 `eval_*/episodes.csv` |
| 新图数值 | `scripts/m8_figures.py` 输出：成功率与 `compare.json` 一致；训练末 10 万转移 3241/3724；1.5M 后验证平均 12.85/16；首次 16/16 在 1 032 000 |

## 3. 自检中发现并已处理的问题

1. **交接单数值有误**：`tasks/handover_claude_3.md` §3.3 写“Q 均值从约 600（30 万步）降到约 0.93（70 万步）”。日志显示 Q 在 14.6k 转移处达峰值约 888，30 万转移时为 41.7。报告按日志写。
2. **`outputs/m2/validation_errors.csv` 是 NaN 注入测试留下的文件**：其中 `g` 一行为 `FAIL nonfinite`，不是正常结果。报告第 3.6 节加了说明，表中数值用重跑结果。没有改动该文件（`outputs/m2/` 属于已完成的里程碑）。
3. **用时口径**：`compare.md` 的 `duration_s_median` 是全部 50 局的中位数；图 9-3 是成功局的中位数。报告中分别注明。
4. **新发现**：M7 验证在 1.792M–1.816M 转移间降到 1–4/16，之后恢复。报告第 8.6 节如实写出，原因未分析。

## 4. 报告中标为“推测”的内容

熵项主导是崩溃原因；γ 折扣导致 DSRL 偏好更快的动作块；`last` 的放置误差较大是以精度换速度；`best` 部分得益于验证运气；Can 质量偏轻；`bc_dsrl` 跨进程差异的原因。

## 5. 待复核

请 Codex 只读复核一轮，写 `tasks/M8_recheck_r1.md`。复核 spec 放在 Claude 的 scratchpad。
