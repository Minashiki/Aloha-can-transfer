# M6 交付记录 r2

按 `tasks/M6_review_r2.md` 最终结论第 1–7 项改完并重跑。环境仍是 `conda run -n AIControl`。未装包，未 import DPPO，未改受保护文件和 M5 数据集，未写 `outputs/m3`–`outputs/m5`，未做 git 写操作。

## 改动文件

- `src/aloha_pp/policy/quaternion.py`：四元数只做无状态的 `w >= 0`。`pose_state` 是训练窗口和在线条件共用的状态。
- `src/aloha_pp/policy/sequences.py`：观测和动作都走这个规则，再拟合归一化。
- `src/aloha_pp/policy/policy.py`：在线 `tcp_pose` 先走 `pose_state`，再用和训练集相同的 float32 归一化。
- `src/aloha_pp/policy/failure.py`：主原因按任务过程，覆盖每一次闭合尝试。IK 不再作为主原因。
- `src/aloha_pp/policy/rollout.py`：视频只渲染要写入的拍；评测在沉降后第二次 `controller.reset`；IK 按控制拍分列；日志写入渲染环境。
- `src/aloha_pp/sim/env.py`：渲染器按分辨率缓存，`close()` 全部释放。`ik_success` 的含义不变，另外给出当前目标和预览目标。
- `src/aloha_pp/sim/c3.py`：`reset_controller_after_settle`。
- `scripts/m6_train.py`、`configs/m6_dp.yaml`：`val_loss_every`，训练日志写入渲染环境。
- `tests/test_m6_diffusion.py`、`tests/test_m6_replay.py`。

`env.reset` 的默认过程没有改。M4 / M5 的分类函数没有改。

## 1. N1 四元数

120 条示范、29257 个时间步：动作 `|w|` 最小 0.500，观测 `|w|` 最小 0.473，没有一帧小于 0.2。半球规则不会在 0 附近来回翻。观测有 34 帧 `w < 0`；动作半球翻转 0。连续性翻转不再计算，这条数据上原先也是 0。

离线窗口和在线 `_cond` 都调用 `pose_state`：每个四元数单独把 `w` 翻到非负，不看前一帧。归一化在这之后、用训练集重新拟合。测试 `test_stored_window_matches_online_condition_and_sign_flip` 用 `demo_0004` 里一个 `w < 0` 的存储窗口：训练路径和在线路径的条件向量逐项相同，把该窗口四元数换成 `-q` 后在线条件向量仍逐项相同。

## 2. F1 渲染器

`_renderers[(height, width)]` 各留一个。`close()` 关掉并清空。默认分辨率仍可通过 `_renderer` 读到，`tests/test_m1_env.py` 不用改。同一状态下 96 和 256 来回渲染，96 的第二次和第一次像素差为 0。

视频在 20 Hz、`video_fps = 10` 时每 2 拍渲染一帧 256×256，不再先渲染再丢掉。

同一条评测设定（冒烟 checkpoint、`random`、种子 0、超时 5 s、录视频，都跑满 100 个控制拍）：

| | 修改前 | 修改后 |
| --- | --- | --- |
| 墙钟 | 20.400 s | 1.707 s |

不录视频时每控制拍（修改后，100 拍）：物理 + 控制 6.52 ms，两路 96 观测渲染 0.61 ms，策略推理 22.17 ms / 块（13 块）。修改前同口径是 5.39 ms、0.57 ms、20.34 ms / 块。无视频路径本来就只用 96，缓存对它几乎没有影响；时间差在反复创建 256 和 96 两个渲染器。

## 3. F2 失败原因

主原因不再返回 `ik_fail`。判定用全部闭合段，不是只看第一段：

- 没有任何一段双指接触：`not_grasped`
- 有接触，但没有任何一段相对该段起点提起 `lift_thresh`：`not_lifted`
- 某一段提起过，并且仍在该段闭合期间丢失接触达到 `lost_steps` 或下落 `drop_z`：`dropped_after_lift`。段结束后松开不算掉落。
- 结束时仍有接触：`not_released`；否则按 M5 的顺序看 `outside`、`not_still`、`timeout`

IK 按控制拍另计，成功 episode 也记：`ik_current_fail_ticks`（当前目标）、`ik_preview_fail_ticks`（预览目标）、`ik_fail_ticks`（这一拍里至少一个失败，每拍最多计 1）。汇总同时有任务原因分布和 IK 拍数。冒烟验证种子 0：主原因 `not_grasped`，当前 IK 失败 139 拍，预览 132 拍。

## 4. F3 渲染环境

训练写 `outputs/m6/smoke_r2/render.json`，评测写进 `eval.json` 的 `render`：

- MuJoCo 3.17.0
- `MUJOCO_GL=egl`
- GL renderer `NVIDIA GeForce RTX 5080/PCIe/SSE2`，vendor `NVIDIA Corporation`
- GPU NVIDIA GeForce RTX 5080，驱动 595.99.02

图像重放阈值仍是像素差 ≤ 1。`tests/test_m6_replay.py` 的说明写明这只在这台 GPU 的 EGL 上成立。

## 5. N2 时长

默认 batch 64 的完整验证损失：22023 个窗口、345 批，**3.555 s**，随机初始化网络的一次前向（损失数值没有意义）。一个训练 epoch 约 1.7 s，所以完整验证损失没有每 epoch 都跑。

`val_loss_every: 20`（最后一轮也跑）。2000 epoch 约 101 次，验证损失约 **0.10 h**。若改成 1，则约 **1.98 h**。闭环评测仍由 `val_every: 100` 控制。

这次冒烟的稳态训练步时 0.0129 s。脚本给出的 2000 epoch、batch 64 训练时间是 **0.82 h**。F1 之后，3 条 25 s 超时且录视频的 episode 共 19.6 s，约 6.5 s / 条。20 次 × 4 条约 **0.15 h**。checkpoint 约 22 MB，每 100 epoch 存一次，相对上面可以忽略。

正式训练合计约 **1.1 h**（0.82 训练 + 0.10 验证损失 + 0.15 闭环），另加约 8 s 读数据。若验证 episode 提前成功，闭环会短于这个上界。

```sh
conda run -n AIControl python -u scripts/m6_train.py \
  --config configs/m6_dp.yaml --out outputs/m6/train
```

## 6. N3 沉降后的 controller.reset

`rollout_episode` 在 `env.reset` 返回后调用 `reset_controller_after_settle`，也就是 M5 在沉降后再做的那次 `controller.reset`。`env.reset` 本身仍在沉降前重置控制器。测试用种子 3 比较两条路径的 `home_q`、IK 种子、参考流的 `q` 和 `v`、`reference_mode`，以及沉降后的关节角，逐项相同，并且 `home_q` 等于沉降后的关节角。

## 7. 重跑

冒烟（损失用 2 个验证 batch，每个 epoch 都算）：

```sh
conda run -n AIControl python -u scripts/m6_train.py \
  --config configs/m6_dp.yaml --out outputs/m6/smoke_r2 \
  --epochs 6 --batch-size 64 --warmup-epochs 0 \
  --val-every 6 --n-val-episodes 1 --val-batches 2 --val-loss-every 1
```

训练损失 0.849215 → 0.321830。验证损失 0.976302 → 0.370120。训练中验证：成功 0/1，Wilson [0, 0.793]，主原因 `not_grasped`，IK 当前/预览 139/132，视频 `outputs/m6/smoke_r2/val_epoch0005/videos/seed00.mp4`（238060 字节）。整次进程 29 s。

```sh
conda run -n AIControl python -u scripts/m6_eval.py \
  --checkpoint outputs/m6/smoke_r2/ckpt_last.pt \
  --seeds 0-2 --videos all --out outputs/m6/smoke_r2/eval
```

成功 0/3，Wilson [0, 0.562]，主原因全是 `not_grasped`。IK 合计当前 379 拍、预览 342 拍，3 条都有 IK 失败。推理中位约 0.022 s。三条视频 238325、246103、238834 字节。没有成功 episode，所以没有成功视频。

回归：

- `pytest tests -q`：97 passed，9.99 s。
- `scripts/check_env.py`：退出码 0，`ALL CHECKS PASSED`。
- `scripts/m1_grasp_smoke.py`：`GRASP_OK`。
- `scripts/m4_run.py --e1 --out outputs/m6/regression/m4_e1`：6 条都成功，7.93 s。分析后与 `outputs/m4/summary.csv` 的 E1 共有非空列 mismatches 0。
- 两条 M5 示范重放在上述 pytest 里通过：再次成功，低维观测逐位相同，像素差 ≤ 1。

冒烟产物（`outputs/` 不进 git）：

| 文件 | 字节 |
| --- | --- |
| `outputs/m6/smoke_r2/ckpt_last.pt` | 22587403 |
| `outputs/m6/smoke_r2/ckpt_epoch0005.pt` | 22591558 |
| `outputs/m6/smoke_r2/loss.png` | 30899 |
| `outputs/m6/smoke_r2/train_log.jsonl` | 1274 |
| `outputs/m6/smoke_r2/train_meta.json` | 1459 |
| `outputs/m6/smoke_r2/render.json` | 198 |
| `outputs/m6/smoke_r2/eval/eval.json` | 8718 |
| `outputs/m6/smoke_r2/eval/episodes.csv` | 925 |
| `outputs/m6/smoke_r2/val_epoch0005/eval.json` | 3502 |

训练子集种子 0、30 条，`source_id` 与 r1 相同：`0, 3, 7, 12, 31, 43, 49, 50, 74, 78, 85, 98, 102, 107, 108, 111, 115, 127, 132, 142, 155, 163, 165, 173, 176, 177, 180, 183, 196, 197`。

## 存疑

- 修改前后的 5 s 录像对比用的是同一个 checkpoint 和种子，都跑满 100 拍。在线四元数规则同时改了，所以动作序列不是逐拍同一条；墙钟差（20.400 s 对 1.707 s）来自渲染器不再按分辨率反复创建。
- 冒烟成功率仍是 0。主原因现在能看出是 `not_grasped`，同时 IK 失败拍数单独记着。
- `--live-viewer` 仍未在无显示环境下打开。
- 正式 2000 epoch 训练没有跑。
