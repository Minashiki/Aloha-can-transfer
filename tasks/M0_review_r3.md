# M0 审查记录 r3（Claude 验收：按审批结论修改后）

- 对象：Cursor follow-up dispatch `ctx_147474493028`，交付记录 `tasks/M0_delivery_r2.md`
- 方式：只读复核，不再经过 Codex（见 WORKFLOW 第 2 节第 7 步）

| 验收项 | 结果 | 依据 |
|---|---|---|
| 只改了 `.gitignore` 中审批范围内的 16 条目录规则 | 通过 | `git diff`：只有 build、dist、data、datasets、checkpoints、ckpt、runs、logs、wandb、outputs、results、videos、figures、third_party、tmp、scratch 这 16 条规则加了前导 `/`，其余行没有变 |
| `src/aloha_pp/data/__init__.py` 不再被忽略 | 通过 | `git check-ignore` 退出码 1；`git status` 中出现该文件 |
| outputs 图片、egg-info、`__pycache__`、PLAN、WORKFLOW 仍被忽略 | 通过 | `git check-ignore` 对这 5 个路径都命中 |
| PLAN.md、WORKFLOW.md 没有被改动 | 通过 | 与修改前记录的哈希比对，`sha256sum -c` 两份都显示“成功” |
| `check_env.py` | 通过 | `ALL CHECKS PASSED` |
| pytest | 通过 | `4 passed` |

Cursor 对话上下文用量：状态栏显示 58.5%（`Grok 4.7 256K High · 58.5%`），已完成 2 次任务。M1 派发前重新检查用量：超过 65%，或者再完成 1 次任务后，就改为开新对话并附交接单。

结论：**M0 验收通过，可以提交。**
