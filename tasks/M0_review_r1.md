# M0 审查记录 r1（Claude）

- 审查对象：Cursor 交付 `tasks/M0_delivery_r1.md`，dispatch `ctx_8e8bea6a3b0e`
- 基线：`c3e6b77`；M0 全部为新增的未跟踪文件，审查以 `git status --untracked-files=all` 加文件内容为准（`git diff` 为空）
- 审查方式：只读。复跑交付中的命令，另跑一次性验证脚本（脚本放在协调者的 scratchpad 目录，不在项目内）

## 1. 合规检查

| 项 | 结果 | 依据 |
|---|---|---|
| `docs/PLAN.md`、`docs/WORKFLOW.md` 未被改动 | 通过 | `sha256sum -c` 对比派发前记录的哈希，两份都显示“成功” |
| `aloha/`、`.gitignore` 未被改动 | 通过 | `git status` 中只有新增的未跟踪文件，没有已跟踪文件被修改 |
| 没有 commit，git 作者配置未变 | 通过 | `git log -1` 仍是 `c3e6b77`；`git config user.email` = `minashikiii@gmail.com` |
| 只在 `AIControl` 环境中操作 | 通过 | 交付记录里每条命令都是 `conda run -n AIControl ...`；复跑时 torch / mujoco 均从该环境导入 |
| 没有安装 DPPO / DSRL / robosuite / robomimic | 通过 | 交付记录的安装清单中没有这几个包 |
| 交付记录包含规定的三项内容 | 通过 | 改动清单、命令及输出、存疑点都写了 |

## 2. 复跑验收命令

- `conda run -n AIControl python scripts/check_env.py`：退出码 0，`ALL CHECKS PASSED`；torch 2.15.0+cu132，CUDA 13.2，含 sm_120，`neq=1`，EGL 下腕部相机输出 96×96 图像。与交付记录一致。
- `conda run -n AIControl python -m pytest tests -q`：`4 passed`。

## 3. 一次性补充验证（后续里程碑要用到的 GPU 功能）

`conda run -n AIControl python -I <scratchpad>/m0_extra.py`，输出：

```
cudnn 92600 conv+backward OK
bf16 OK torch.bfloat16
compile max err 2.384185791015625e-07
torchvision False
```

结论：交付记录中提到，部分 NVIDIA 库版本高于 13.2（cublas 13.4 等）。实测在本机驱动下，cuDNN 卷积前向 + 反向、bf16 autocast、`torch.compile`（Triton）都能正常运行。

## 4. 问题

### F1【高】`.gitignore` 的目录规则没有锚定根目录，`src/aloha_pp/data/` 整个子包被 git 忽略

- 现象：`git status` 的未跟踪文件里看不到 `src/aloha_pp/data/__init__.py`。`git check-ignore -v` 显示命中的是 `.gitignore:20: data/`。
- 影响：提交后，别人 clone 下来没有 `aloha_pp.data` 子包，M5 的示范数据代码也会被静默忽略。同一类规则（`logs/`、`checkpoints/`、`outputs/`、`results/`、`figures/`、`videos/`、`runs/`、`tmp/`、`build/` 等）会忽略任何层级下的同名目录。实测 `src/aloha_pp/utils/logs/x`、`src/aloha_pp/policy/checkpoints/x`、`configs/data/x.yaml` 都会被忽略。
- 责任：`.gitignore` 是协调者在开工前写的，不是 Cursor 的交付缺陷。
- 修改建议：把“产物目录”类规则改成只匹配仓库根目录（加前导 `/`），例如 `/data/`、`/datasets/`、`/checkpoints/`、`/runs/`、`/logs/`、`/outputs/`、`/results/`、`/videos/`、`/figures/`、`/wandb/`、`/tmp/`、`/scratch/`、`/build/`、`/dist/`、`/third_party/`、`/ckpt/`。文件扩展名类规则（`*.hdf5`、`*.pt`、`*.mp4` 等）和 `__pycache__/`、`.venv/` 这类规则保持不变。
- 验收：修改后 `git check-ignore` 不再命中 `src/aloha_pp/data/__init__.py`，`outputs/m0_wrist_cam.png` 仍被忽略。
- 注意：执行这项修改需要临时解除 M0 spec 中“不得修改 `.gitignore`”的限制，由用户审批。

### F2【中】没有安装 torchvision

- 现象：`torchvision False`。spec 只要求装 torch，所以这不算交付违规。
- 影响：M6 Diffusion Policy 的图像编码器（DP / DPPO 常用 ResNet18）以及图像增广一般依赖 torchvision。torchvision 的版本必须和 torch 2.15.0+cu132 严格配对，到 M6 再装，有可能把 torch 一起换掉。
- 修改建议：现在就从同一个 cu132 索引安装与 torch 2.15.0 配对的 torchvision，并在 `environment.yml` 中钉死版本。装完后确认 `torch.__version__` 仍为 `2.15.0+cu132`、`check_env.py` 仍全部通过，再加一项 `torchvision.ops` 或 `resnet18` 的 GPU 前向冒烟测试。

### F3【低】`environment.yml` 能否复现，尚未验证（推测）

- 文件里 conda 部分钉了 `setuptools=83.0.0`、`wheel=0.47.0`、`packaging=26.3` 等确切版本。这些版本在 `defaults` 频道里是否都有，没有验证。pip 部分的写法看起来是对的。
- 规定禁止新建环境，所以不能用 `conda env create` 实测。建议改为：conda 部分只钉 `python=3.12.14` 和 `pip`，其余包都放进 pip 段钉版本，这样可复现的把握更大。此条为推测，复核时可以反驳。

### F4【信息】mujoco 停在 3.17.0，PyPI 最新是 4.0.1

按 PLAN 的“mujoco 3.x”执行，正确，不需要改。记录下来供用户知悉：如果以后要升 4.x，需要另行决定。

## 5. Cursor 对话状态

- 上下文用量：**无法读取**。状态栏没有显示用量，Cursor 配置里是 `showStatusIndicators: false`。
- 该对话已完成任务数：1。按保守规则，连续完成 3 次后换新对话。

## 6. 结论

M0 主体合格。待处理：F1（必须修）、F2（建议现在修）、F3（建议修）。F4 不处理。
