# M0 审查记录 r2（Claude 对 recheck_r1 的回应与最终结论）

- 依据：`tasks/M0_review_r1.md`（Claude）、`tasks/M0_recheck_r1.md`（Codex，dispatch `ctx_debe96a23585`）
- Codex 只读核查：`docs/PLAN.md`、`docs/WORKFLOW.md` 的哈希与派发 Codex 前一致（`sha256sum -c` 两份都显示“成功”）；`git status` 中新增的文件只有 `tasks/M0_recheck_r1.md`。通过。
- 打回轮次：第 1 轮已经结束。下面的分歧全部采纳 Codex 意见，没有剩余分歧，所以不进入第 2 轮。

## 逐条回应 Codex

| 项 | Codex 意见 | Claude 回应 |
|---|---|---|
| 合规第 1、3、4 项 | 部分同意：Codex 没有派发前的哈希基准，也不能证明过去的操作全部合规 | 接受这个证据边界。补充依据：派发 Cursor 前，PLAN、WORKFLOW 的哈希记录在协调者的 scratchpad 里，交付后用 `sha256sum -c` 比对通过；作者邮箱的检查对象是提交时实际使用的配置。结论不变 |
| F1 | 同意；验收条件补充一条：`data/__init__.py` 必须真的出现在待提交文件清单里 | **采纳**，验收条件按补充后的内容写 |
| F2 torchvision | 部分同意：缺包属实，但 spec 没有要求，M6 也可以裁剪，不应在 M0 必须修 | **采纳**。撤销 F2 这一修改项，改为记在 M6 的要求里：安装 torchvision 时用约束文件钉住 `torch==2.15.0+cu132`，装完重新跑 `check_env.py` |
| F3 environment.yml | 反驳：AIControl 的 conda-meta 显示 python / setuptools / wheel / packaging 都来自 `repo.anaconda.com/pkgs/main`，版本与 yml 一致 | **采纳**，撤销 F3。完整重建环境仍未验证，这一点保留为已知边界 |
| F4 | 同意不处理 | 一致 |
| 补充 GPU 验证 | 部分同意：只做了有限测试，不能推广到所有负载 | 接受。表述改为“已验证的几项 GPU 功能可用” |
| 总评 | 主体功能通过，F1 解决前不能入库 | 一致 |

## Codex 对话上下文用量

Codex 会话日志最后一次请求的输入约 45.5K token，上下文窗口 475K，用量约 9.6%，低于 65%，后续轮次可以继续 `send` 给这个对话。

## 最终结论（提交用户审批）

M0 主体合格。**只需修改 1 项**：

**F1**：修改 `.gitignore`，给产物目录类规则加前导 `/`，只匹配仓库根目录：
`/build/`、`/dist/`、`/data/`、`/datasets/`、`/checkpoints/`、`/ckpt/`、`/runs/`、`/logs/`、`/wandb/`、`/outputs/`、`/results/`、`/videos/`、`/figures/`、`/third_party/`、`/tmp/`、`/scratch/`。
`*.hdf5`、`*.pt`、`*.mp4` 这类扩展名规则，以及 `__pycache__/`、`.venv/`、`venv/`、`env/`、`.conda/`、编辑器目录、`docs/PLAN.md`、`docs/WORKFLOW.md` 两条规则，都保持不变。

- 执行者：Cursor（发到原对话 `ctx_8e8bea6a3b0e`）。本次修改临时解除 M0 spec 中“不得修改 `.gitignore`”的限制，只允许改上面列出的规则。
- 验收：
  1. `git check-ignore src/aloha_pp/data/__init__.py` 不再命中；
  2. `git status --short --untracked-files=all` 里出现 `src/aloha_pp/data/__init__.py`；
  3. `outputs/m0_wrist_cam.png`、`src/aloha_pp.egg-info/`、`__pycache__/` 仍然被忽略；
  4. `docs/PLAN.md`、`docs/WORKFLOW.md` 仍然被忽略；
  5. `check_env.py` 和 pytest 仍然全部通过。

记入后续里程碑：M6 安装 torchvision 时钉住 torch 版本（原 F2）。
