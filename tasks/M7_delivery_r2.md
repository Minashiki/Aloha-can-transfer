# M7 交付 r2：按审查结论修改

正式训练没有跑，种子 0–49 的正式评测没有跑。冒烟、确定性回归、真实循环短程计时和测试都在 `conda run -n AIControl` 里执行。没有安装包，没有写 `outputs/m3`–`outputs/m6`，没有做 git 写操作。

`total_env_steps` 仍是 2_000_000。按这次真实循环外推约 **18.5 h**，在 24 h 以内，所以没有改步数，也没有改 UTD 和网络宽度。

## 1. 改动文件

修改：

- `scripts/m7_eval.py`：`--videos {none,auto,all}`，默认 `none`。
- `scripts/m7_compare.py`：可重复的 `--dsrl LABEL=eval.json`，`--selected` 把事先声明的 checkpoint 标成主结果。
- `scripts/m7_train.py`：续训时调用不可变配置比对。
- `src/aloha_pp/rl/evaluate.py`：`none` 不录像；`eval.json` 写入请求的录像模式、实际录像种子和运行时版本。
- `src/aloha_pp/rl/trainer.py`：16 条验证不录像；另开一个环境录 1 条，不进成功率和选模；训练日志增加数值健康字段；分项计时写入 `timing.json`。
- `src/aloha_pp/rl/sac.py`：每次更新返回 mean log π、Q / target Q 的均值和最大值、α、确定性动作的均值绝对值。
- `src/aloha_pp/rl/compare.py`：噪声来源、是否录像、配对提升 / 退化和精确 McNemar p 值。Wilson 区间只作描述。
- `src/aloha_pp/rl/checkpoint.py`：`assert_resume_config`。
- `src/aloha_pp/policy/policy.py`：`encode_actor_obs` 拒绝 `state`，并要求两路图像、`tcp_pose`、`gripper_opening`。`_cond` 和 `sample` 没改。
- `configs/m7_dsrl.yaml`：只改了时长注释。`total_env_steps`、`utd`、`layer_size` 的数值没改。
- `tests/test_m7_dsrl.py`：新增 6 个测试。原有测试没有删改。

新增：

- `scripts/m7_det_compare.py`：比对两次 `episodes.csv`。
- `tasks/M7_delivery_r2.md`：本文件。

## 2. 跑过的命令和摘要

环境前缀都是 `conda run -n AIControl`。长任务用了 `--no-capture-output python -u`。

1. `python -m pytest tests -q -p no:cacheprovider`：**117 passed**（13.76 s）。交接时是 111，这次新增 6 个，原有测试都在。
2. `python -u scripts/check_env.py`：退出码 0，`ALL CHECKS PASSED`。python 3.12.14，torch 2.15.0+cu132，CUDA 13.2，mujoco 3.17.0。
3. `python -u scripts/m1_grasp_smoke.py`：`GRASP_OK`。
4. 冒烟训练：`--smoke --out outputs/m7/smoke_r2/train`。墙钟 14.7 s，280 个环境步，7 条训练 episode 里 6 条成功。两次验证都是 **2/2**（种子 20000、20001），不是 3 条。视频单独写出，`video_in_success` 为 false：
   - `outputs/m7/smoke_r2/train/val_step00000140/videos/seed20000.mp4`（76416 字节）
   - `outputs/m7/smoke_r2/train/val_step00000280/videos/seed20000.mp4`（76410 字节）
   第 240 步起的训练日志有 `logp_mean`、`q_mean`、`q_max`、`target_q_mean`、`target_q_max`、`alpha`、`actor_abs_mean`。第 240 步：logp_mean −71.46，q_mean 31.09，q_max 31.30，target_q_mean 73.34，target_q_max 86.40，α 0.997，actor_abs_mean 0.161。
5. 续训：`--smoke --resume outputs/m7/smoke_r2/train/ckpt_last.pt --out outputs/m7/smoke_r2/train --total-env-steps 312 --val-every 100000 --no-val-at-end`。日志是 `resumed env_steps 280`、`loaded replay size 280`，走到 312。第 300 步仍有上述健康字段（logp_mean −83.41，q_max 46.39，α 0.988，actor_abs_mean 0.145）。没有再验证。不可变配置一致，没有被拒绝。
6. 确定性回归：两个独立进程，`--videos none`，种子 0–2，正式 BC checkpoint 只读。比对脚本退出码 0，打印 `DET_OK`。见第 4 节。
7. 真实循环计时：`--out outputs/m7/benchmark_real --n-envs 8 --init-rollout-steps 2 --total-env-steps 3216 --val-every 2400 --no-val-at-end`。网络宽度 2048、UTD 20、batch 256 都来自正式配置。墙钟 104.6 s。验证 **12/16**，录像不计入。见第 5 节。

## 3. 清单 1–10

### 1. 评测录像隔离

`evaluate_latent` 在 `videos="none"` 时 `record=False`，不会走到 `rollout_episode` 里的 `env.render("agentview", 256, 256)`。`eval.json` 有 `videos` 和 `video_seeds`。

测试 `test_videos_none_does_not_render`：三局的 `record` 都是 false，录像渲染调用次数是 0，`video_seeds` 是 `[]`。确定性回归的三份 `eval.json` 也都是 `"videos": "none"`、`"video_seeds": []`，日志里 `video None`。

### 2. 跨进程确定性回归

见第 4 节。`scripts/m7_det_compare.py` 逐种子比较成功标志、主原因、`ik_current_fail_ticks`、`ik_preview_fail_ticks`、`ik_fail_ticks`、`place_xy`、`duration_s`。两次一致。

### 3. 验证录像隔离

统计用的验证波次固定 `record=False`。若统计结果里出现视频路径，训练直接报错。每次验证之后，用单独的 1 个环境再跑种子 `val_seed_start`，只写视频。这条的成功与否记在 `video_success`，`video_in_success` 为 false，不加入 `n_success`，选模只用统计成功率。

冒烟两次验证的 `n` 都是 2。正式宽度的短程计时验证 `n` 是 16（种子 20000–20015），12/16，视频：

`outputs/m7/benchmark_real/val_step00002400/videos/seed20000.mp4`（79 KB）

测试 `test_validate_without_record_does_not_write_video`：`record=False` 时 `write_video` 调用次数为 0；`record=True` 时写一次。

### 4. best / last 都评测

正式命令在第 6 节，两条 `bc_dsrl` 分别指向 `ckpt_best.pt` 和 `ckpt_last.pt`，输出目录分开。`m7_compare.py` 用重复的 `--dsrl 名字=eval.json`。`--selected` 读 `selected.json` 的 `selected`：`ckpt_best.pt` 对应标签 `bc_dsrl_best`，`ckpt_last.pt` 对应 `bc_dsrl_last`。对上的那一行 `role` 是「主结果（事先声明）」。当前配置 `checkpoint_select: best_val`，所以主结果是 best。标签对不上会退出，不会静默漏标。

测试 `test_paired_comparison_uses_mcnemar_and_marks_the_declared_result` 覆盖了这个标记。正式 0–49 评测没有跑。

### 5. 对比口径

表的列包括 `role`、`noise`、`recorded`。M6 那一行的噪声栏是「随机噪声（未固定）、种子 0–3 录像，仅作参考」。`bc_zero` 是 `w = 0`，`bc_seeded` 是按种子固定的 N(0, I)，`bc_dsrl_*` 是确定性 actor。`videos=none` 的结果写成「不含录像」。

配对只做 `bc_seeded` 对每个 `bc_dsrl_*`，以及 `bc_zero` 对每个 `bc_dsrl_*`。每对给出提升数（右成功、左失败）、退化数（右失败、左成功）和精确 McNemar p 值：不一致的种子数 n = 提升 + 退化，p 是 Binomial(n, 1/2) 较小一侧概率的两倍，超过 1 时截为 1。Wilson 区间留在表里，报告开头写明它不作为有无提升的判据。M6 的 0.62 不进入配对。

### 6. actor 方向测试

`test_actor_moves_the_rewarded_component_positive`：CPU，`torch.manual_seed(0)`，`torch.set_num_threads(1)`，特征固定为 0，特权维 1，动作维 1，两层宽 32，学习率 3e-3。2048 条均匀动作，奖励是 `w[0] > 0`，全部 `terminated`。500 次 batch 128 的 SAC 更新。断言确定性 `w[0]` 的绝对值开始小于 0.05，结束大于 0.3。该测试在全套里通过，耗时包含在 13.76 s 的整次 pytest 中；单独跑新增测试时 6 个测试 2.37 s。

### 7. 真实吞吐

见第 5 节。短程用的就是正式的 8 环境、2048 宽、UTD 20、batch 256。热身改成 2 个 vec step，热身之后是 400 个 vec step（其中 29 步在凑满 batch，371 步做了更新）。包含两次 `encode_actor_obs`、写回放、7420 次 SAC 更新、一次 16 条验证加 1 条录像、两次 checkpoint（`ckpt_best.pt` 和结束时的 `ckpt_last.pt`）。

### 8. 数值健康日志

`log_every` 的 `event=train` 记录现在带有 `logp_mean`、`q_mean`、`q_max`、`target_q_mean`、`target_q_max`、`alpha`、`actor_abs_mean`。还没更新时这些字段是 null，冒烟第 20–220 步就是这样。更新开始后是数。

正式宽度短程的第一笔非空记录在第 400 步：logp_mean **−86.91**，q_mean 174.1，q_max 174.1，target_q_mean 188.0，target_q_max 193.8，α 0.890，actor_abs_mean 0.067。到第 3200 步：logp_mean −37.67，q_mean 820.2，q_max **945.3**，target_q_max 945.3，α 0.126，actor_abs_mean 0.481。Q 在这段里一直上升，见第 7 节。

### 9. encode_actor_obs 输入

出现 `state` 就 `ValueError`。缺少 `tcp_pose`（以及两路图像或 `gripper_opening`）也 `ValueError`。`can_pose` 这类多余键仍然被忽略，原有的「多塞特权字段不改变特征」测试没改，仍然通过。`DiffusionPolicy._cond` 没改，M6 的 `sample` 仍可以在没有 `tcp_pose` 时走 `state`。

测试 `test_encode_actor_obs_rejects_state_and_missing_tcp`。

### 10. 续训比对不可变配置

比对 `noise_scale`、`seed`、`train_seed_start`、`val_seed_start`、`n_val_episodes`、`eval_seed_start`、`eval_seed_count`，以及 SAC 的学习率、batch、tau、discount、utd、`layer_size`、`num_layers`、layer norm、`n_critics`、目标熵和 log_std 夹取。不一致就 `SystemExit`。`n_envs`、`total_env_steps`、`val_every`、`save_replay` 不在这组里。

测试 `test_resume_rejects_changed_noise_scale_and_allows_n_envs`：改 `noise_scale` 或 `layer_size` 会退出；只把 `n_envs` 从 8 改成 4 可以通过。冒烟续训的配置与 checkpoint 一致，所以跑通了。

`save_replay: false` 时的续训行为，代码没有改，这里写明：

- 回放池是新建的空池。只有 `save_replay` 为 true 且旁边有 `replay.pt` 才加载。
- 不把热身计数清零。`warming` 仍是 `vec_steps < init_rollout_steps`。正式训练续到 1501 个 vec step 之后，热身不会再来一遍。
- 空池要先攒满一个 batch（正式配置是 256 条转移）。`n_envs` 为 8 时是 32 个 vec step。
- 攒满之后，每个 vec step 立刻做 `utd` 次更新。正式配置是 20，不会退回热身的「只采集、不更新」。

这次冒烟续训走的是另一支：`--smoke` 把 `save_replay` 设成 true，所以加载了 280 条回放，日志是 `loaded replay size 280`，`warmup_vec_steps` 为 0。

## 4. 确定性回归

命令（`det_b` 把目录里的 `det_a` 换成 `det_b`，其余相同）：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_zero --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --seeds 0-2 --videos none --out outputs/m7/regression/det_a/bc_zero

conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_seeded --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --seeds 0-2 --videos none --out outputs/m7/regression/det_a/bc_seeded

conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_dsrl --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --actor-checkpoint outputs/m7/smoke/train/ckpt_last.pt \
  --seeds 0-2 --videos none --out outputs/m7/regression/det_a/bc_dsrl
```

比对：

```bash
conda run -n AIControl python -u scripts/m7_det_compare.py \
  --a outputs/m7/regression/det_a --b outputs/m7/regression/det_b
```

输出 `bc_zero 3 episodes match`、`bc_seeded 3 episodes match`、`bc_dsrl 3 episodes match`、`DET_OK`。两个进程打印的成功标志、主原因、IK 拍数、`place_xy`、时长逐项相同：

| 方法 | 种子 | 成功 | 主原因 | IK current/preview | place_xy | 时长 s |
| --- | --- | --- | --- | --- | --- | --- |
| bc_zero | 0 | true | 无 | 0/0 | 0.0009 | 12.65 |
| bc_zero | 1 | true | 无 | 0/0 | 0.0198 | 13.20 |
| bc_zero | 2 | true | 无 | 0/0 | 0.0066 | 11.95 |
| bc_seeded | 0 | true | 无 | 0/0 | 0.0194 | 13.75 |
| bc_seeded | 1 | true | 无 | 0/0 | 0.0194 | 13.00 |
| bc_seeded | 2 | true | 无 | 0/0 | 0.0084 | 16.35 |
| bc_dsrl | 0 | true | 无 | 0/0 | 0.0046 | 13.30 |
| bc_dsrl | 1 | true | 无 | 0/0 | 0.0202 | 13.30 |
| bc_dsrl | 2 | true | 无 | 0/0 | 0.0072 | 11.65 |

`bc_dsrl` 用的是第一版冒烟 actor（`outputs/m7/smoke/train/ckpt_last.pt`），还在热身附近，不是学完的策略。三组都成功只说明这 3 个种子在这三种显式噪声下走通了，不能外推到 0–49。

两边 `eval.json` 的运行时相同：torch 2.15.0+cu132，CUDA 13.2，cuDNN 92600，驱动 595.99.02，GPU NVIDIA GeForce RTX 5080，mujoco 3.17.0，`MUJOCO_GL=egl`。`torch.get_num_threads()` 是 8，interop 线程 16，`cpu_count` 16。

这只覆盖 `--videos none`。第一轮在这台 GPU + EGL 上看到的「录像渲染会改变后续观测」没有在这一轮重测。

## 5. 真实吞吐和新的时长

短程目录 `outputs/m7/benchmark_real/`。402 个 vec step，8 个环境，3216 条转移。热身 2 步，凑 batch 29 步，学习 371 步，7420 次更新。一次验证（16 条统计 + 1 条录像，11.75 s）和两次 checkpoint。计时在每个分项边界上 `cuda.synchronize`，所以下面的秒数包含了 GPU 等待，和将要跑的训练循环是同一套代码。

| 分项 | 这次短程 | 每个 vec step | 外推到 2e6 |
| --- | --- | --- | --- |
| 采集（采样噪声、扩散 `sample`、环境 step） | 27.487 s | 0.0684 s | 4.75 h |
| 编码（每步两次 `encode_actor_obs`） | 2.860 s | 0.0071 s | 0.49 h |
| 更新（371 个学习步 × UTD 20） | 60.322 s | 0.1626 s / 学习步，0.00813 s / 次 | 11.22 h |
| 验证（16 + 1 录像） | 11.755 s | 一次 11.75 s | 1.63 h（500 次） |
| I/O（回放写入 + checkpoint） | 0.401 s | 回放 0.019 s；每次保存 0.191 s | 0.06 h |
| 未分进上面五类的循环开销 | 1.793 s | 0.0045 s | 0.31 h |
| 合计 | 104.618 s |  | **18.46 h** |

外推用的正式设置：250000 个 vec step，前 1501 个是热身、不做更新，其余 248499 个 vec step 每个做 20 次更新（共 4,969,980 次）。验证每 4000 条转移一次，一共 500 次，末次和 `val_at_end` 落在同一步，不会再加一次。I/O 按 501 次 `ckpt_last` 写入（每 4000 步一次，结束时再写一次）加上每次验证都存一次 `ckpt_best` 的上界来算。best 不会真的每次都变好，所以 0.06 h 是上界，改不了「是否超过 24 h」。

一次 SAC 更新约 8.1 ms，和 r1 只测更新时的 7.9 ms 接近。r1 把 2e6 估成 17 h，漏的主要是验证录像和循环里的其它开销；编码本身大约半小时，不是大头。采集这一项和 r1 的 0.076 s/vec step 同一量级。

若以后每次验证都比这次慢一倍（这次 12/16 成功，失败的会打满 25 s），验证从 1.63 h 变成 3.27 h，总时长约 20.1 h，仍低于 24 h。因此没有把 `total_env_steps` 改小。代价是：保持 2e6 就接受每个转移平均约 2.5 次更新，而不是为了缩短时间去砍 UTD 或宽度。

## 6. 正式命令

都在 conda 环境 `AIControl` 里跑。四条评测都是 `--videos none`，输出都在 `outputs/m7/` 下，不会写到 `outputs/m6/`。

正式训练：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_train.py \
  --config configs/m7_dsrl.yaml --out outputs/m7/train
```

续训（`save_replay: false`，回放池为空，不重新热身，攒满 256 条后每个 vec step 做 20 次更新）：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_train.py \
  --config configs/m7_dsrl.yaml \
  --resume outputs/m7/train/ckpt_last.pt \
  --out outputs/m7/train
```

四条正式评测：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_zero --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --seeds 0-49 --videos none --out outputs/m7/eval_bc_zero

conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_seeded --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --seeds 0-49 --videos none --out outputs/m7/eval_bc_seeded

conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_dsrl --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --actor-checkpoint outputs/m7/train/ckpt_best.pt \
  --seeds 0-49 --videos none --out outputs/m7/eval_bc_dsrl_best

conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_dsrl --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --actor-checkpoint outputs/m7/train/ckpt_last.pt \
  --seeds 0-49 --videos none --out outputs/m7/eval_bc_dsrl_last
```

单独录像，不作为上面四条的成功率，也不要拿这个目录去做配对：

```bash
conda run -n AIControl --no-capture-output python -u scripts/m7_eval.py \
  --method bc_dsrl --bc-checkpoint outputs/m6/train/ckpt_last.pt \
  --actor-checkpoint outputs/m7/train/ckpt_best.pt \
  --seeds 0-3 --videos all --out outputs/m7/video_bc_dsrl_best
```

训练过程里的验证视频已经按第 3 项单独写出，路径在 `train_log.jsonl` 的 `video` 字段。那一条不参与选模。

对比。`selected.json` 若仍是 `ckpt_best.pt`，表里 `bc_dsrl_best` 标为「主结果（事先声明）」：

```bash
conda run -n AIControl python -u scripts/m7_compare.py \
  --bc outputs/m6/train/eval/eval.json \
  --bc-zero outputs/m7/eval_bc_zero/eval.json \
  --bc-seeded outputs/m7/eval_bc_seeded/eval.json \
  --dsrl bc_dsrl_best=outputs/m7/eval_bc_dsrl_best/eval.json \
  --dsrl bc_dsrl_last=outputs/m7/eval_bc_dsrl_last/eval.json \
  --selected outputs/m7/train/selected.json \
  --m4 outputs/m4/summary.csv \
  --out outputs/m7/compare
```

## 7. 只记录、不改代码的差异

和 SB3 SAC 的差别，来自 `tasks/M7_recheck_r1.md` 的 SAC 专项，这次没有改实现：

- critic 损失是两个 MSE 直接相加，没有 0.5 因子。
- 更新顺序是 critic → actor → α。SB3 先更新 α。
- critic 吃的是 w，不是 w / 1.5。
- 初始化不同：均值头正交增益 0.01，log_std 头也是正交初始化；`log_std_init` 只写在偏置上。

每个转移的更新次数是 `utd / n_envs = 20 / 8 = 2.5`。官方 4 个环境、UTD 20 时是每个转移 5 次。这次没有把 UTD 改成 40。

目标熵 0 在 128 维下，tanh 修正后的初期 log π 约 −86。短程计时第 400 步的 `logp_mean` 是 −86.91，和这个量级一致。之后策略变尖，第 3200 步是 −37.7，α 从 0.89 降到 0.13。

## 8. 未完成和存疑

- 正式训练和种子 0–49 的四条评测没有跑。上面的命令是交给后续启动的。
- 没有任何证据说明 DSRL 的成功率高于 `bc_zero` 或 `bc_seeded`。确定性回归里的 `bc_dsrl` 用的是未学习的冒烟 actor。短程计时里 actor 的均值绝对值从 0.07 升到 0.48，只说明权重在动。
- 短程里 Q 和 target Q 的最大值在 3200 步升到约 945，而且还在涨。目标熵 0 配上未终止转移的 bootstrap，初期 `-α log π` 会把 TD 目标抬高，这和复核里「不是公式写反」的判断一致，但这次短程看不出 Q 会不会在更后面稳住。健康字段会把这个过程记下来。没有因此改目标熵或奖励。
- 18.5 h 是按这一次 12/16 的验证耗时线性外推的。验证若系统性变慢，上界大约再加 1.6 h。分项边界上的 CUDA 同步留在训练循环里，正式训练会付同样的代价。
- 录像会不会改变本机 GPU / EGL 上的后续观测，这一轮没有再做实验。处理办法是统计和录像分开，确定性回归只跑 `videos=none`。
- 16 个固定验证种子会在 2e6 步里被验证大约 500 次。`best_val` 仍可能偏高。主结果事先定为 `checkpoint_select` 指向的那个文件，best 和 last 都评，但不能靠 0–49 的结果回头改选择。
