# M7 交付 r1：DSRL（噪声空间 SAC）

基线 commit `006a65bcf52e2b11011f26cc0159d7377df8bbd0`。实现参考 `third_party/dsrl` commit `8d21b9cf55459f022c0e769bca1343edd9d30e5a`（只读，运行时不 import）。没有安装包，没有逐段复制官方源码，所以没有改 `THIRD_PARTY_LICENSES`。DSRL-NA 没有做。

正式训练没有跑。冒烟、评测和回归都在 `conda run -n AIControl` 里执行。

## 结论里必须先看的一项

M6 冒烟回归的 IK 拍数对不上，而且同一条命令重跑也不稳定。这是本交付没有通过的验收项，见第 8 节。其余冒烟路径是跑通的。

## 1. 改动文件

新增：

- `configs/m7_dsrl.yaml`
- `scripts/m7_train.py`、`scripts/m7_eval.py`、`scripts/m7_compare.py`
- `src/aloha_pp/rl/`：`checkpoint.py`、`chunk_env.py`、`compare.py`、`evaluate.py`、`nets.py`、`noise.py`、`replay.py`、`reward.py`、`sac.py`、`spec.py`、`trainer.py`、`vec.py`
- `tests/test_m7_dsrl.py`

修改（M6 扩散核心 `diffusion.py`、`unet1d.py`、`vit.py`、`modules.py` 未动）：

- `src/aloha_pp/policy/policy.py`：只加了 `actor_feature_dim` 和 `encode_actor_obs`。`sample` 的函数体没改。
- `src/aloha_pp/policy/rollout.py`：`rollout_episode` 增加可选参数 `chunk_noise`。不传时仍走 `sample(obs, noise=None)`，和原来的 `sample(obs)` 是同一个分支。
- `src/aloha_pp/rl/__init__.py`：只有说明文字，不 import MuJoCo，避免子进程在设置 `MUJOCO_GL` 之前加载 MuJoCo。

`outputs/` 不进 git。文件大小见第 7 节。

## 2. DSRL 设计

主变体是 DSRL-SAC：M6 扩散策略冻结，SAC 的动作是 DDIM 的初始噪声。一个动作块是 RL 的一步：预测 16 步，执行前 8 步，第 8 步预览第 9 步，沿用 `ChunkExecutor`。

### 潜动作

`DiffusionPolicy.noise_shape = (16, 8)`，展平 128 维。actor 输出高斯，经 tanh 再乘 `noise_scale`。配置值 **1.5**，对应官方 `cfg/robomimic/dsrl_can.yaml` 的 `action_magnitude`。均值头用正交初始化、增益 0.01，所以没训练时的确定性动作接近 0，和 `bc_zero` 是同一类起点，但不是逐位为 0。

三种噪声不要混：

- 热身（`init_rollout_steps` 个 vec step）：`w ~ N(0, I)` 再裁到 `±noise_scale`，写入回放池。这是官方 `collect_rollouts` 的做法。
- 热身之后：actor 的 tanh 高斯，范围同样是 `±noise_scale`。
- `bc_seeded` 评测：`w ~ N(0, I)`，不裁剪。随机数发生器是 `np.random.default_rng(SeedSequence([eval_seed, 0xB7C0]))`，每个动作块一次 `standard_normal`。不改 `sample(noise=None)`。

### actor 输入

不含物体真值。`encode_actor_obs` 在 eval 模式、关闭增强下取 EMA 的 ViT + SpatialEmb（两路相机），再拼上归一化后的本体：2 帧 `pose_state(tcp_pose, gripper_opening)`，规则与 M6 相同。正式网络是 128×2 + 8×2 = **272** 维。测试确认：调用它之前和之后，`sample` 的输出逐位相同；多塞 `can_pose`、`can_linvel`、`q`、`qvel` 不改变这个向量。

### critic 输入

非对称。顺序是特权状态、actor 特征、潜动作 w。特权状态 30 维，字段和宽度：

| 字段 | 维 | 来源 |
| --- | --- | --- |
| `can_pose` | 7 | `get_privileged` |
| `can_linvel` | 3 | `get_privileged` |
| `tcp_pose` | 7 | `get_obs` 的原始位姿，不是 `pose_state` |
| `gripper_opening` | 1 | `get_obs` |
| `q` | 6 | `ARM_JOINTS` |
| `qvel` | 6 | `get_obs` |

actor 主干只吃 272 维特征。critic 输入是 30+272+128 = **430**。双 Q，取较小值做 backup，目标网络按 `tau` 软更新。

### 网络和 SAC 超参数

出处：`third_party/dsrl` 的 `cfg/robomimic/dsrl_can.yaml` 和 `train_dsrl.py` 的 `policy_kwargs`。该 yaml 的 `algorithm` 是 `dsrl_na`，但 `train:` 块与 `dsrl_sac` 共用。这里只实现 SAC。

| 项 | 值 | 出处 |
| --- | --- | --- |
| 隐藏层 | 3 × 2048，Linear → LayerNorm → Tanh | `layer_size`、`num_layers`、`use_layer_norm`，激活 Tanh |
| 学习率 | 3e-4，actor / critic / 温度共用 | `actor_lr`，与 SB3 的单一学习率一致 |
| batch | 256 | `batch_size` |
| tau | 0.005 | `tau` |
| γ | 0.99 | `discount` |
| UTD | 20 | `utd`：每个 vec step 做 20 次梯度，不是每个转移 20 次 |
| 目标熵 | 0.0 | `target_ent`。SB3 的 `"auto"` 会设成 `-action_dim`，官方没有用那个 |
| log_std | 初值 0，夹在 [-20, 2] | `log_std_init` 和 SB3 的默认夹取 |
| 备份 | 双 Q 的较小值 | `n_critics: 2` |
| 热身 | 1501 个 vec step 的裁剪标准正态 | `init_rollout_steps` |

`target_entropy = 0` 时，平衡点是 `E[log π] ≈ 0`。128 维标准正态的 log 概率大约是 -180，所以训练最初几步的 TD 目标会被 `-α log π` 抬到一两百，critic 损失会到几千。这是这个目标熵的直接结果，不是把奖励尺度弄错了。冒烟里损失从 3593 升到约 5200，只更新了大约 30 次，不能当成学没学动的证据。单元测试在一个小网络、终止转移（不 bootstrap 熵项）上看到 critic 损失降到初值的一半以下。

和官方数字不同、并且写进配置的两项：

- `total_env_steps: 2000000`。官方 `model.learn(total_timesteps=20000000)`。20e6 按实测要大约 171 小时。
- `replay_capacity: 2000000`。官方 buffer 是 20e6。特征存两份，每条约 3 KB，20e6 大约 60 GB，这台机器大约 30 GB 内存放不下。2e6 大约 6 GB，和步数预算一样长，不会挤掉早期转移。

`n_envs` 官方是 4。实测 8 个环境的转移吞吐更高，所以配置改成 8。代价是：UTD 20 是「每个 vec step 20 次更新」，每个转移分到的更新从官方的 20/4 = 5 变成 20/8 = 2.5。把 UTD 改成 40 才能恢复每个转移 5 次更新，更新时间大约翻倍，总时长会超过 24 小时。当前配置保留官方的 `utd: 20` 这个数字，并接受更稀的更新。

### 奖励、终止、截断

默认稀疏。成功判定通过的那个 RL 步奖励 1，`terminated`，TD 不再 bootstrap。超时默认 25 s（控制频率 20 Hz，500 个控制拍），`truncated`，`terminated` 为假，可以 bootstrap。一个 RL 步最多 8 个控制拍；正好打满 500 拍的那一步是截断。成功如果出现在块的中途，这一步的控制拍数小于 8。

分阶段奖励默认关闭。关闭时奖励只有 0 或 1。打开后用特权信息，每段每个 episode 最多给一次，反复进出不加分：

| 阶段 | 奖励 | 条件 |
| --- | --- | --- |
| `both_contact` | 0.1 | 两指都碰到 Can，且夹爪指令与 `gripper_close`（0.015）的差不超过 `grip_closed_tol`（1e-4） |
| `lifted` | 0.2 | 已经接触过，接触之后的 Can 最高 z ≥ 接触时 z + 0.02 m |
| `above_cell` | 0.3 | 已经提起，块结束时 xy 在目标格半宽内，且 z ≥ 格中心 z + 0.02 m |
| `released_in_cell` | 0.4 | 已经提起，块结束时 Can 在格内（完整 3D）且两指都已松开 |

这四个数不是论文里的。成功的 +1 另算。提起用接触之后的最高 z，避免接触前的高度误触发。格子在构造环境时按 `random` 模式读 `m1` 的目标格；训练和评测都用 `random`。

DSRL-NA（噪声混叠）没做。官方 README 把它写成更省样本的变体。主变体如果在 2e6 步后仍然接近 `bc_zero`，再考虑它，而不是先把奖励改成分阶段。

## 3. 环境、并行、种子

环境用 `make_c3_env`，`reset` 之后调用 `reset_controller_after_settle`，和 M6 评测相同。并行是 `spawn` 子进程，每个进程在 import MuJoCo 之前设置 `MUJOCO_GL=egl`。父进程批量做特征提取和 `sample`。`--live-viewer` 强制 1 个进程内环境，默认关闭，这次没有显示器，没有实测。

同一种子、同一潜动作，子进程和进程内环境的低维轨迹（tcp、关节、Can 位姿）在 1e-5 以内，图像最大像素差 ≤ 1。测试覆盖了这一点。

| 环境数 | 转移/秒 | 一个 vec step |
| --- | --- | --- |
| 1 | 14.7 | 0.068 s |
| 4 | 66.4 | 0.060 s |
| 8 | 105.0 | 0.076 s |

测量是正式 BC checkpoint、`w = 0`、每种 20 个 vec step，不含 SAC。一次 SAC 更新（batch 256、宽度 2048、预计算特征）是 **0.0079 s**。推荐 **8** 个环境。

种子三组互不相交，测试检查了配置：

- 训练 episode：从 10000 起，跳过验证集和评测集。`seed: 0` 只给网络初始化和热身的 numpy 发生器，不给环境。
- 验证：20000 起，默认 16 个，即 20000–20015。不用 M6 的 0–3。
- 评测：`random` 模式种子 0–49。

checkpoint 只按验证成功率选。`best_val` 用严格大于，平局留更早的那个文件。评测种子不参与选择。`selected.json` 里 `eval_seeds_used_for_selection` 为 false。

训练中验证默认每 4000 个环境步做 16 条，确定性均值噪声，并行跑，并至少写一条 ≤ 2 MB 的视频，路径进 jsonl。`val_at_end` 为 true。步数跳跃如果跨过间隔也会触发，避免 `n_envs` 不整除 `val_every` 时永远不验证。

## 4. 跑过的命令和摘要

环境前缀都是 `conda run -n AIControl`。长任务加了 `--no-capture-output python -u`。

1. `python -m pytest tests/test_m7_dsrl.py -q`：14 passed。
2. `python -m pytest tests -q`：111 passed（交接时 97，新增 14 个 M7 测试，原有测试没改）。
3. `python -u scripts/m7_train.py --config configs/m7_dsrl.yaml --benchmark --benchmark-steps 20 --out outputs/m7`  
   吞吐见上表。当时配置还是占位的 200000 步，估计 1.65 h。随后把 `total_env_steps` 改成 2000000、`n_envs` 改成 8、`replay_capacity` 改成 2000000。
4. 冒烟训练：`--smoke --out outputs/m7/smoke/train`。10.6 s 墙钟，280 个环境步，4 个并行环境。6 条 episode 里 5 条奖励 1 并终止（种子 10000、10001、10003、10005、10006），1 条超时截断、奖励 0（种子 10002，63 个 RL 步、500 个控制拍）。成功路径出现了。  
   验证在第 140 步和 280 步都是 2/2，Wilson [0.34, 1.0]。视频：  
   `outputs/m7/smoke/train/val_step00000140/videos/seed20000.mp4`（75.0 KB）  
   `outputs/m7/smoke/train/val_step00000280/videos/seed20000.mp4`（74.9 KB）  
   热身是 55 个 vec step（220 个环境步）。SAC 从大约第 240 步开始有损失：critic 3592.6 → 5039.0，actor -103.3 → -123.2，α 0.997 → 0.991。
5. 续训第一次失败：`torch.load(..., map_location=cuda)` 把 RNG 状态变成 CUDA 张量，`torch.set_rng_state` 拒绝。已在 `rl/checkpoint.py` 的 `restore_rng` 里转回 CPU `ByteTensor`。第二次：`--resume outputs/m7/smoke/train/ckpt_last.pt --total-env-steps 312 --val-every 100000 --no-val-at-end`。日志是 `resumed env_steps 280`、`loaded replay size 280`，步数走到 312，回放池到 300，critic 5201，没有再验证。
6. 冒烟评测，种子 0–4，三种方法都是 5/5，IK 失败拍数都是 0。`bc_dsrl` 用的是 `selected.json` 指向的 `ckpt_best.pt`。该文件保存在第 140 步，还在热身里，actor 没有梯度更新。所以这 5/5 是未学习的确定性噪声，不是 DSRL 学会了。放置误差和 `bc_zero` 不完全相同，说明确定性均值不是逐位 0。  
   种子 3 和 4 在 M6 正式评测里是 `not_grasped`，这里三种噪声都成功。只有 5 个种子，不能据此说 `w = 0` 高于 0.62。种子 0–49 的正式评测才能比。  
   这 5 个种子没有失败，所以没有失败视频。训练里那条超时也没有单独录视频（视频只在验证时录，两次验证都是成功）。
7. 对比：`scripts/m7_compare.py`，读 `outputs/m6/train/eval/eval.json`（只读）、三种冒烟评测和 `outputs/m4/summary.csv`，写到 `outputs/m7/smoke/compare/`。表见第 6 节。
8. `scripts/check_env.py`：退出码 0，`ALL CHECKS PASSED`。
9. `scripts/m1_grasp_smoke.py`：`GRASP_OK`。
10. `scripts/m4_run.py --e1 --out outputs/m7/regression/m4_e1`，再 `m4_analyze.py --runs outputs/m7/regression/m4_e1/runs --out outputs/m7/regression/m4_e1`。6 行 E1 与 `outputs/m4/summary.csv` 的共有非空列逐项一致，mismatch 0。
11. `scripts/m6_eval.py --checkpoint outputs/m6/smoke_r2/ckpt_last.pt --seeds 0-2 --out outputs/m7/regression/m6_eval`。成功标志都是 false，主原因都是 `not_grasped`，和 `outputs/m6/smoke_r2/eval/` 一致。IK 拍数和 `place_xy` 不一致。紧接着用 `--out outputs/m7/regression/m6_eval_repeat` 再跑一次，连主原因都不稳定。见第 8 节。

## 5. 正式训练

配置就是 `configs/m7_dsrl.yaml`：8 个环境，2e6 环境步，UTD 20，宽度 2048，验证 16 条、每 4000 步一次。

按实测相加（采集和更新是串行的）：

- 采集：2e6/8 × 0.0762 s ≈ 5.3 h
- 更新：(250000 − 1501) × 20 × 0.00792 s ≈ 10.9 h
- 验证：500 次 × 2 波。每条验证按 40 个 RL 步约 0.85 h，按超时 63 步约 1.3 h
- 合计约 **17.1 h**（验证偏短）到 **17.6 h**（每次验证都超时）

20e6 步大约 171 h。2.5e6 步、验证按 40 RL 步估计约 21.4 h，仍在 24 h 内，但余量小；配置留在 2e6。若日志里的步速明显低于这次 benchmark，先减 `total_env_steps`，不要先改 UTD 或宽度。若要坚持每个转移 5 次更新（UTD 40、8 环境），更新时间大约翻倍，会超出 24 h。

启动：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_train.py \
  --config configs/m7_dsrl.yaml \
  --out outputs/m7/train
```

断点续训默认不保存回放池（`save_replay: false`，与官方一致），续上的是网络、优化器、温度、步数和种子流，回放池是空的。要连回放池一起续，把 `save_replay` 设为 true，会在输出目录写 `replay.pt`。续训命令：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_train.py \
  --config configs/m7_dsrl.yaml \
  --resume outputs/m7/train/ckpt_last.pt \
  --out outputs/m7/train
```

评测用 `outputs/m7/train/selected.json` 里的文件。`checkpoint_select: best_val` 时那就是验证成功率严格更高的 `ckpt_best.pt`；如果一次验证都没完成，用 `ckpt_last.pt`。不要用种子 0–49 挑 checkpoint。冒烟里两次验证都是 2/2，严格大于留住了第 140 步的热身权重。正式的 16 条验证不容易一开始就满分，但平局同样会留更早的文件。

种子 0–49，三种方法：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_zero --seeds 0-49 --out outputs/m7/eval_bc_zero

conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_seeded --seeds 0-49 --out outputs/m7/eval_bc_seeded

conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_dsrl \
  --actor-checkpoint outputs/m7/train/ckpt_best.pt \
  --seeds 0-49 --out outputs/m7/eval_bc_dsrl
```

第三条的 `--actor-checkpoint` 换成 `selected.json` 里的那个文件。对比：

```bash
conda run -n AIControl python -u scripts/m7_compare.py \
  --bc outputs/m6/train/eval/eval.json \
  --bc-zero outputs/m7/eval_bc_zero/eval.json \
  --bc-seeded outputs/m7/eval_bc_seeded/eval.json \
  --dsrl outputs/m7/eval_bc_dsrl/eval.json \
  --m4 outputs/m4/summary.csv \
  --out outputs/m7/compare
```

日志格式是 `aloha_pp.eval.v1`，`method` 分别为 `bc_dsrl`、`bc_zero`、`bc_seeded`。

## 6. 对比表示例

冒烟种子 0–4，加上 M6 正式 50 种子和 M4 规划器 E2。全文在 `outputs/m7/smoke/compare/compare.md`。

| method | n | n_success | success_rate | wilson_95 | failure_counts | ik_fail_ticks_sum | place_xy_median_success | duration_s_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc | 50 | 31 | 0.6200 | [0.4815, 0.7414] | not_grasped 15, not_lifted 4 | 0 | 0.0049 | 12.8500 |
| bc_zero | 5 | 5 | 1.0000 | [0.5655, 1.0000] |  | 0 | 0.0064 | 12.6000 |
| bc_seeded | 5 | 5 | 1.0000 | [0.5655, 1.0000] |  | 0 | 0.0092 | 13.2000 |
| bc_dsrl | 5 | 5 | 1.0000 | [0.5655, 1.0000] |  | 0 | 0.0070 | 12.9500 |
| planner_e2 | 50 | 50 | 1.0000 | [0.9286, 1.0000] |  |  | 0.0045 | 6.8000 |

相对 M6 正式评测、只比较共有种子：`bc_zero`、`bc_seeded`、`bc_dsrl` 都是种子 3 和 4 从失败变成成功，种子 0、1、2 两边都成功。规划器相对 BC 的失败转成功是种子 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 17, 29, 31, 34, 35, 41, 43, 46, 47。

## 7. outputs/m7 文件

冒烟权重：`ckpt_best.pt` 178 MB（热身，优化器状态还是空的），`ckpt_last.pt` 392 MB（续训到 312 步）。`replay.pt` 1.3 MB。验证视频两条，各约 75 KB。三种评测的 `eval.json` 约 9 KB，视频每条约 75–92 KB，都低于 2 MB。`benchmark.json` 1.0 KB。`compare.md` 1.6 KB。回归的 M4 E1 在 `outputs/m7/regression/m4_e1/`（`summary.csv` 8.3 KB，六个 npz 约 1–3 MB）。M6 两次重跑在 `outputs/m7/regression/m6_eval/` 和 `m6_eval_repeat/`。

## 8. 未完成和存疑

**M6 冒烟回归没有通过。** `outputs/m6/smoke_r2/eval/` 与这次 `outputs/m7/regression/m6_eval/`：

| 种子 | 存放的结果 | 这次 |
| --- | --- | --- |
| 0 | not_grasped，IK 123/115/198，place_xy 0.0826 | not_grasped，135/130/214，0.2497 |
| 1 | not_grasped，131/119/194，0.2569 | not_grasped，133/122/214，0.2569 |
| 2 | not_grasped，125/108/204，0.4631 | not_grasped，129/125/219，0.1909 |

成功标志和主原因这次是齐的，IK 拍数不齐。立刻再跑一次到 `m6_eval_repeat`：种子 0、2 仍是 `not_grasped`，种子 1 变成 `not_lifted`，IK 分别是 113/105、126/126、123/114。两次重跑彼此也不一致。

原因：torch 2.15 每个新进程的 `torch.initial_seed()` 都不同（实测两次是 16661029662002857589 和 9507265242052470238）。M6 评测调用 `sample(noise=None)`，噪声来自这个进程随机源。同一进程里固定噪声的 `sample` 逐位相同；不给噪声的两次 `sample` 不同。所以存放的那份 IK 拍数无法复现，也没有一个种子可以填回去。没有改 `m6_eval.py` 去强行播种，那样既改变 M6 的行为，也对不上旧文件。M7 的 `bc_seeded` 就是为了避开这件事。

其余存疑：

- 稀疏奖励能不能让 DSRL 超过 `bc_zero`，这次没有证据。冒烟的 `bc_dsrl` 是热身权重。
- `w = 0` 在 50 个种子上相对 0.62 是高是低，未知。种子 0–4 上它是 5/5，含 M6 失败的 3 和 4。
- 分阶段奖励和 DSRL-NA 都没开。主变体学不动时再考虑，顺序是先看 2e6 步的验证曲线。
- `--live-viewer` 没在有显示器的情况下跑过。
- 冒烟评测没有失败样本，因此没有失败视频。
