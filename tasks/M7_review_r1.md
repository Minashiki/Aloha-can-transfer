# M7 审查 r1（Claude）

对象：`tasks/M7_delivery_r1.md`，`git diff 006a65b` 及未跟踪的新文件，`outputs/m7/`。只读：一次性脚本和输出都在 Claude 的 scratchpad（`/tmp/claude-1000/.../scratchpad/`），没有写项目文件（本审查文件除外）。所有命令都在 `conda run -n AIControl` 中执行。

**结论：主体合格，需修改后再启动正式训练。** DSRL 主变体的设计、超参数出处、非对称输入、终止 / 截断、种子分离、并行环境都符合派发要求。Cursor 报告失败的“M6 冒烟回归”是 **Claude 在 spec 中写错了验收标准**，不算 Cursor 的缺陷；但我的复现实验发现，Cursor 给出的根因不完整：**录视频本身会破坏闭环复现**（R2），这会影响 M7 的正式对比，必须先修。另有三项较小的修改（R3、R4、R5）。

## 1. 合规核对

| 项 | 结果 | 依据 |
|---|---|---|
| PLAN / WORKFLOW 哈希 | 通过 | `sha256sum -c` 与派发前快照一致（PLAN `a90182e3…54d9`、WORKFLOW `f13164fa…1987`） |
| `outputs/m3`–`m6` 未写 | 通过 | 派发前后 590 个文件的路径 / 大小 / mtime 清单 `diff` 无差异 |
| tracked 改动范围 | 通过 | 只改 `policy/policy.py`（新增 `actor_feature_dim`、`encode_actor_obs`，`sample` 未动）、`policy/rollout.py`（新增可选参数 `chunk_noise`，不传时仍是 `sample(stacked, noise=None)`，与原 `sample(stacked)` 同一分支）、`rl/__init__.py`（仅 docstring）。扩散核心、配置、`retarget.py`、数据集未动 |
| 不装包 | 通过 | `AIControl` 的 site-packages 最新条目为 2026-11-13，conda 修订只有 rev 0 |
| 不依赖 DSRL 官方仓库 | 通过 | `third_party/dsrl`（commit `8d21b9c`）只读，`.gitignore` 第 49 行排除；`src/` 中无 import |
| git | 通过 | HEAD 仍为 `006a65b`，无暂存改动 |
| 测试 | 通过 | 本机 `python -m pytest tests -q -p no:cacheprovider`：**111 passed**（97 + 14） |
| 其余回归 | 接受交付记录 | `check_env.py`、`GRASP_OK`、M4 E1 逐项一致：本次未重跑 |
| Cursor 上下文 | 54.6%，已完成 1 次任务 | 状态栏 `Grok 4.7 256K High · 54.6%`；可继续用于本轮修改 |

## 2. 设计逐项核对

| 派发要求 | 判断 | 说明 |
|---|---|---|
| 潜动作 `tanh(u)·noise_scale`，形状 (16, 8) | 通过 | `noise_scale = 1.5`，与官方 `dsrl_can.yaml` 的 `action_magnitude` 一致（已对照 `third_party/dsrl`） |
| actor 输入无物体真值 | 通过 | 272 维 = 两路 128 维 SpatialEmb + 2 帧 8 维本体；测试覆盖“塞入特权键不改变特征” |
| 非对称 critic | 通过 | 30 维特权（Can 位姿 / 线速度、TCP、夹爪、q、q̇）+ actor 特征 + w，共 430 维；字段已写明 |
| SAC 超参数出处 | 通过 | 已对照官方 `cfg/robomimic/dsrl_can.yaml` 与 `train_dsrl.py`：lr 3e-4、batch 256、τ 0.005、γ 0.99、`utd` 20 即 SB3 `gradient_steps`（每个 vec step）、`target_ent` 0、2048×3 + LayerNorm、双 Q 取 min、`init_rollout_steps` 1501 |
| 一个动作块 = 一个 RL 步；成功 terminal、超时 truncated | 通过 | `chunk_env.py`；`td_target` 只在 terminated 时停止 bootstrap |
| 自动重置时 next_obs 不被覆盖 | 通过 | `trainer.py` 先算 `next_feat` 并写入回放，再用重置观测替换 `state` |
| `ChunkEnv`（训练 / 验证）与 `rollout_episode`（评测）执行规则一致 | 通过（代码对读） | reset → `reset_controller_after_settle` → `hist=[obs]*2` → `ChunkExecutor` 每 8 拍取块、第 8 拍预览第 9 拍 → 成功即停，上限 500 拍；两边一致 |
| 分阶段奖励默认关闭、每阶段至多一次 | 通过 | 关闭时奖励只有 0 / 1；有测试 |
| 训练 / 验证 / 评测种子两两不相交 | 通过 | 10000 起 / 20000–20015 / 0–49；`selected.json` 写明评测种子不参与选择 |
| 并行环境与单环境一致 | 通过 | 测试：低维 1e-5 内、像素差 ≤ 1 |
| 正式训练时长 ≤ 约 24 h | 通过 | 2e6 步、8 环境，估计 17.1–17.6 h；与官方不同的两项（步数 2e6 vs 20e6、回放 2e6 vs 20e6）以及“每个转移的更新次数 2.5 vs 官方 5”都已写明 |

## 3. 需要修改的问题

### R1 M6 冒烟回归的验收标准是错的（Claude 的 spec 错误）

- **事实**：我在 spec 中要求“M6 冒烟 checkpoint 在种子 0–2 上评测，与 `outputs/m6/smoke_r2/eval/` 一致（成功标志、主原因、IK 拍数）”。这个标准不可能满足：`m6_eval.py` 用 `sample(noise=None)`，而 torch 2.15 每个进程的默认种子不同（我实测两个进程的 `torch.initial_seed()` 分别为 18297884036280286209 和 7307011323489312162）。
- 更进一步：即使在调用前固定 `torch.manual_seed(123)`，**M6 基线代码自己**（`git archive 006a65b` 导出到 scratchpad）连跑两次也不一致：种子 1 的 IK 拍数 126 对 136，种子 2 的 231 对 164（`runs/base_a`、`runs/base_b`）。原因见 R2。
- **处理**：Cursor 把这一项报为失败、没有为了凑数去改 `m6_eval.py`，做法正确。本项改判为“标准不适用”。M6 的回归改为：
  1. `m6_eval.py` 能跑完，输出格式和字段与 `smoke_r2/eval` 相同（已满足）；
  2. 代码对读：`noise=None` 分支与原调用相同（已满足，见第 1 节）；
  3. 新增确定性回归（见 R2 第 3 点），用来代替逐位对比。

### R2 录视频会破坏闭环复现；正式对比必须不录像（重要）

**证据**（scratchpad 中的 `nd_probe*.py`；同一种子、同一 checkpoint，在两个独立进程中各跑一次 500 拍，每 50 拍比较 `qpos` 哈希）：

| 条件 | 两个进程的轨迹 |
|---|---|
| 正式 checkpoint，种子 3，w = 0，不录像，80 拍 | 图像、特征、动作块、qpos **逐位相同** |
| 冒烟 checkpoint，种子 1，w = 0，不录像，500 拍 | **逐位相同** |
| 冒烟 checkpoint，种子 1，`noise=None` + `manual_seed(123)`，不录像 | **逐位相同** |
| 同上，但每 2 拍插入一次 `env.render("agentview", 256, 256)`（即评测录像的做法） | 第 99 拍仍相同，**第 149 拍起不同**；而且第 99 拍已经和不录像的轨迹不同 |

在复位状态下静态交替渲染 256 和 96 的图像，96 的观测并没有变化（`render_probe.py`）。所以差异出在闭环执行过程中，录像渲染与观测渲染交替进行时；具体机制没有查清。

**结论**：
- 不录像、显式给噪声时，闭环评测在本机**跨进程逐位可复现**。所以 `bc_zero`、`bc_seeded` 和确定性 `bc_dsrl` 的评测本来可以完全复现。
- 录像的 episode 既不能复现，也和不录像时不是同一条轨迹。`evaluate_latent` 在 `auto` 模式下会给前几个种子录像（直到成功、失败都出现过），这些种子的结果因此带随机性。
- M6 正式评测的 0.62，种子 0–3 是录了像的，噪声也未固定；它只作为“随机噪声下 BC 的一次抽样”参考。M7 的主 BC 基线应是 `bc_seeded` 和 `bc_zero`。
- Cursor 在第 8 节写的根因（进程种子）对 `noise=None` 成立，但不是全部原因。

**要求修改**：
1. `scripts/m7_eval.py` 增加 `--videos {none,auto,all}`。正式对比的三条（四条，见 R3）评测命令一律用 `none`。视频另外用 `--videos all --seeds <少量种子> --out outputs/m7/videos_<method>` 单独录一遍，交付和 `compare.md` 中注明“录像 episode 与统计用的 episode 不逐位相同”。
2. `m7_compare.py` 读入的评测如果含录像 episode，在表中标注；M6 正式评测一行标为“随机噪声（未固定）、种子 0–3 录像，仅作参考”。
3. **确定性回归**（代替 R1 中的逐位对比）：在两个独立进程中，各用 `m7_eval.py --videos none` 对 `bc_zero` 和 `bc_seeded` 跑种子 0–2（正式 BC checkpoint，`--out outputs/m7/regression/det_a`、`det_b`），`episodes.csv` 中的成功标志、主原因、IK 拍数、`place_xy`、时长逐项相同。写成脚本或在交付中给出命令和比对结果。
4. （可选，限时 30 分钟以内）查一下录像为什么影响观测；查不清就如实写明，不要求修。训练中验证只给第一条 episode 录像，这一条结果可能有随机性，可以接受，在交付中写明即可。

### R3 正式对比同时报告 `ckpt_best` 和 `ckpt_last`

- `best_val` 是在约 500 次、每次 16 条的验证中取最大值（严格大于、平局留旧的）。前约 12000 个环境步还处在热身期，这时的 actor 没训练过，也参与竞争。最大值本身会被验证噪声抬高，选中的 checkpoint 可能只是运气好的那一个。
- 这两个 checkpoint 都不是用评测种子选出来的，所以同时报告两者不会把评测种子泄漏进模型选择。
- **要求**：交付中的正式评测命令改为两条 `bc_dsrl`（`--actor-checkpoint` 分别为 `ckpt_best.pt` 和 `ckpt_last.pt`，输出目录分开）。`m7_compare.py` 能接受多个 DSRL 结果并分别标注（例如 `bc_dsrl_best`、`bc_dsrl_last`），逐种子的转移对两者都给出。`selected.json` 保留，作为事先声明的主结果。

### R4 缺少 actor 改进方向的测试

- 现有的 SAC 测试覆盖了 critic 损失下降、目标网络软更新、α 的更新方向，但**没有测试 actor 会朝高回报的方向移动**。冒烟中的 `bc_dsrl` 用的是热身权重，actor 从未更新，所以“actor 经过 tanh 和缩放后的梯度方向正确”至今没有证据。
- **要求**：加一个小测试。单步 bandit：特征固定，奖励只取决于 w 的某个分量的符号（例如 `w[0] > 0` 得 1，否则得 0），终止转移，用小网络更新几百次，断言确定性动作的该分量从约 0 移到明显大于 0。要求在 CPU 上几秒内跑完。

### R5 对比表的标注

- `compare.md` 的表头或注释写明各方法的噪声来源：`bc` 是 M6 正式评测（随机、未固定），`bc_zero` 是 w = 0，`bc_seeded` 是按种子固定的 N(0, I)，`bc_dsrl_*` 是确定性 actor；再加上是否录像（R2）。
- 写明两个比较结论的口径：Wilson 区间是否重叠；逐种子转移只看 `bc_seeded` / `bc_zero` 与 `bc_dsrl` 之间（都可复现），不要把 M6 的 0.62 当作配对基线。

## 4. 不需要修改、但需要在报告中写明的事项

- **每个转移的更新次数 2.5（官方 5）**：8 个环境 × UTD 20 是在 24 h 预算下的取舍。Cursor 已写明，我接受，报告中照写。
- **目标熵 0、128 维**：初期 `−α·log π` 约 +180，critic 损失在几千量级；α 会随 `log π < 0` 下降。官方 Can 配置的动作维度是 4×7 = 28，这里是 16×8 = 128，初期熵奖励的量级约为官方的 4–5 倍。我判断不是缺陷，交 Codex 复核。
- 冒烟没有给出任何“DSRL 学到了东西”的证据（预期如此）。正式训练要由用户启动，约 17 h。

## 5. 给 Cursor 的修改清单（经用户审批后发送）

R2（1–3 必做，4 可选）、R3、R4、R5。修改后：`pytest tests -q` 全部通过；R2-3 的确定性回归结果；更新 `tasks/M7_delivery_r2.md`（改动文件、命令、结果，以及更新后的正式训练、评测、对比命令）。约束不变：只在 `AIControl` 中运行，不装包，不写 `outputs/m3`–`m6`，不做 git 写操作。
