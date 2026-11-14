# M6 交付记录 r1

Diffusion Policy（图像，行为克隆）已接入。冒烟训练的损失明显下降，训练中验证和 `m6_eval.py` 都写出了成功率、失败分布和 mp4。`sample(obs, noise)` 在 DDIM η = 0 下对同一观测和同一初始噪声逐位相同。`pytest tests -q` 为 93 passed。M4 E1 共有指标与 `outputs/m4/summary.csv` 逐项相同。两条 M5 示范用共享 C3 环境重放后再次成功，低维观测与存储值逐位相同。

环境：`conda run -n AIControl`。未安装任何包。未执行 git commit / push / add / config。未改 `docs/`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、M2、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`、`configs/m4_task.yaml`、`configs/m5_*.yaml`、`src/aloha_pp/data/retarget.py` 或 `outputs/m5/dataset/`。

记录时的 HEAD 是 `4d4f02b16abc184c737c23c319f98157e4ceae05`，工作区 dirty（checkpoint 里的 `git_dirty` 为 true）。

## 改动文件

新增：

- `src/aloha_pp/sim/c3.py`：C3 电机模型、增益、控制器和 `make_c3_env`。内容与原先 `scripts/m4_run.py` 里的组装相同。
- `src/aloha_pp/policy/modules.py`、`vit.py`、`unet1d.py`、`diffusion.py`：从 DPPO 摘出的网络、噪声调度和 DDIM。
- `src/aloha_pp/policy/THIRD_PARTY_LICENSES`：来源说明和 DPPO MIT 全文。
- `src/aloha_pp/policy/quaternion.py`、`normalizer.py`、`sequences.py`、`chunk.py`、`failure.py`、`policy.py`、`checkpoint.py`、`rollout.py`。
- `src/aloha_pp/policy/__init__.py`：导出 `DiffusionPolicy`。
- `configs/m6_dp.yaml`、`scripts/m6_train.py`、`scripts/m6_eval.py`。
- `tests/test_m6_diffusion.py`、`tests/test_m6_replay.py`。

修改：

- `scripts/m4_run.py`：`_gains`、`_controller`、`_motor_cfg` 改为调用 `aloha_pp.sim.c3`。`m5_generate.py` 和既有测试仍从 `m4_run` 导入这三个名字。

`third_party/dppo` 是只读 clone，已被 `.gitignore` 排除，不进 git。

## DPPO 来源

- URL：https://github.com/irom-princeton/dppo
- Commit：`cc7234ad7ff39a8f32de3af903606723a16f0648`
- 许可证：MIT，Copyright (c) 2024 Intelligent Robot Motion Lab。全文在 `src/aloha_pp/policy/THIRD_PARTY_LICENSES`。
- 运行时不 import DPPO，也不把它当依赖安装。

| 本仓库文件 | 原路径 | 改动 |
| --- | --- | --- |
| `modules.py` | `model/diffusion/modules.py`，`model/common/modules.py` | einops `Rearrange` 改成 unsqueeze / squeeze。每个 Conv 块单独建激活。随机平移网格和 `SpatialEmb` 的 patch/feature 转置保持原样。 |
| `vit.py` | `model/common/vit.py` | einops 改成 flatten / permute / reshape。注意力用 math SDPA，不用只开 flash 的内核。去掉未使用的 `act_layer` 和 `__main__`。编码器仍来自 DPPO 所注明的 IBRL。 |
| `unet1d.py` | `model/diffusion/unet.py` | einops 改成 transpose。不复制 `cond_mlp`。随机平移只在 `train()` 时做。两路图像按切好的时间长度拆通道。上采样仍在每一级都做，和 DPPO 一样，用来配上下采样在最后一级被跳过的长度。 |
| `diffusion.py` | `model/diffusion/diffusion.py`，`model/diffusion/sampling.py` | 余弦 β 调度、ε 的 MSE、leading DDIM。η 固定为 0。初始噪声由参数传入。η = 0 时不再乘一个标准差为 0 的额外噪声，RNG 不被碰。不复制源码里 `forward` 传给 `p_mean_var` 但不被接受的 `deterministic`。 |

ViT 在文件头注明 IBRL。余弦调度和 1D 残差块在文件头注明 Diffuser / Diffusion Policy。放进许可证文件的正文是这次复制的 DPPO 修订的 MIT。

## 策略

观测：`agentview` 与 `wrist_cam_left`，各 96×96 RGB，加上 `tcp_pose`（7）和 `gripper_opening`（1）。没有物体真值，没有关节角。历史 `n_obs_steps = 2`。episode 开头把第一帧 `get_obs()` 重复，补满 2 帧。

动作：数据集里的 8 维绝对 TCP，base 系，xyz + 四元数 wxyz + 夹爪开度。预测 16 步，执行 8 步（20 Hz，约 0.4 s）。执行第 k 步（0 起）时，`next_action` 是块内第 k+1 步；第 8 步（k = 7）预览块内第 9 步（下标 8）。`ChunkExecutor` 要求 `pred_horizon >= exec_horizon + 1`。

归一化：只在 30 条训练示范上、窗口填充之前，按维 min/max 映到 [-1, 1]。极差下限 `1e-6`，常数维映到 -1，反归一化回到下界。图像不做这套归一化；ViT 内部是 `obs / 255 - 0.5`。统计量在 `action_lo/hi`、`state_lo/hi` 缓冲里，也写在 checkpoint 的 `extra["normalizer"]`。

四元数：120 条示范的动作四元数没有 `w < 0`，相邻点积也没有负值（半球翻转 0，连续性翻转 0）。观测 `tcp_pose` 有 34 次 `w < 0`，连续性翻转 0。范数约在 0.99999994 到 1。训练前在 float32 里先把 `w < 0` 翻到正半球，再把与前一帧点积为负的帧翻转；不再单位化，避免改掉已经是单位四元数的存储位。送入环境前 `to_env_action` 在 float32 里单位化并保证 `w >= 0`，再升到 float64。位置和夹爪是 float32 值升到 float64，和示范的精度一致。q 与 −q 是同一旋转。

网络：两路相机共用一个 ViT（depth 1，embed 128，4 heads，embed2），各自一个 `SpatialEmb`（128，deepcopy 后分开训练）。条件是 2 帧 × 8 维状态，展平为 16。UNet `dim = 40`，`dim_mults = [1, 2]`，kernel 5，GroupNorm 8 组，FiLM `cond_predict_scale`。dropout 0。在线网络参数量 **1,392,848**。EMA 是同结构的一份拷贝，decay 0.995，`sample` 用 EMA，训练损失用在线网络。

噪声：训练 100 步余弦调度，预测 ε，`denoised_clip = 1`。采样默认 DDIM 8 步（配置项，建议范围 5–10），η = 0。给定初始噪声时更新是确定的。

增强：随机平移，`shift_pad = 4`，只在 `module.training` 时启用。`sample` 会把网络切到 eval，并暂时关掉 TF32、打开 cuDNN deterministic，结束时恢复。

窗口：每个时间步一个窗口。观测历史不足时重复第 0 帧。动作块超出末尾时重复最后一个动作。

## 噪声接口（给 M7）

`DiffusionPolicy.noise_shape` 是 `(pred_horizon, action_dim) = (16, 8)`，不含 batch。`sample(obs, noise=None, ddim_steps=None)`：

- `noise is None` 时内部采标准正态，形状 `(B, 16, 8)`。
- 传入时形状必须是 `(B, 16, 8)`，或单条时的 `(16, 8)`。
- 返回 `(B, 16, 8)` float32，已反归一化，四元数已单位化且 `w >= 0`。
- `eta` 固定 0。改 `ddim_steps` 不改变输出形状。

测试：CPU 与 CUDA 上，同一 `obs` 和同一 `noise` 两次 `sample` 逐位相同；换一份噪声则不同。checkpoint 存读往返后输出逐位相同。

## 训练子集

`subset_seed = 0`，`n_train_demos = 30`，`np.random.default_rng(0).choice` 后排序。`source_id`：

`0, 3, 7, 12, 31, 43, 49, 50, 74, 78, 85, 98, 102, 107, 108, 111, 115, 127, 132, 142, 155, 163, 165, 173, 176, 177, 180, 183, 196, 197`

对应 `demo_0000` 到 `demo_0118` 中的这 30 个名字，写在 `outputs/m6/smoke/train_meta.json` 和 checkpoint 里。其余 90 条只用于验证损失，归一化统计不用它们。

## 冒烟训练与评测

命令（AIControl）：

```sh
conda run -n AIControl python -u scripts/m6_train.py \
  --config configs/m6_dp.yaml --out outputs/m6/smoke \
  --epochs 6 --batch-size 64 --warmup-epochs 0 \
  --val-every 6 --n-val-episodes 1 --val-batches 2
```

数据加载约 10 s。7234 个训练窗口，22023 个验证窗口。设备 cuda。

| epoch | train loss | val loss（2 个 batch） | lr | step |
| --- | --- | --- | --- | --- |
| 0 | 0.849261 | 0.976169 | 1.00e-4 | 0.0188 s |
| 5 | 0.322425 | 0.370975 | 1.60e-5 | 0.0154 s |

去掉第一步后的实测步时 0.0149 s。冒烟把 warmup 设成 0，余弦调度按 6 个 epoch 收，所以末段学习率偏低；正式配置仍是 warmup 100、2000 epoch。

训练中验证（种子 0，超时 25 s）：成功 0/1，Wilson 95% [0, 0.793]，失败 `ik_fail`，放置 xy 0.266 m，时长 25.00 s。视频 `outputs/m6/smoke/val_epoch0005/videos/seed00.mp4`，235611 字节。动作块推理中位 0.0222 s，最大 0.0712 s（第一块）。

评测：

```sh
conda run -n AIControl python -u scripts/m6_eval.py \
  --checkpoint outputs/m6/smoke/ckpt_last.pt \
  --seeds 0-2 --videos all --out outputs/m6/smoke/eval
```

成功 0/3，Wilson 95% [0, 0.562]，失败 `{'ik_fail': 3}`。三条都超时 25 s。推理中位约 0.022 s。视频：

- `outputs/m6/smoke/eval/videos/seed00.mp4`，243133 字节
- `seed01.mp4`，245098 字节
- `seed02.mp4`，231905 字节

冒烟模型没有成功 episode，所以没有成功视频。派发允许成功率为 0。日志格式是 `aloha_pp.eval.v1`，`method` 为 `bc`，M7 可以写成 `bc_dsrl` 并沿用同一字段（成功率、Wilson、失败计数、放置误差、时长、每个动作块的 `infer_s`）。失败分类沿用 M5 的物理步顺序：success、ik_fail、not_grasped、not_lifted、dropped_after_lift、not_released、outside、not_still、timeout。阈值从 m1 / m4 / m5 配置读取，不另写一套。

`outputs/` 不进 git。冒烟文件：

| 文件 | 字节 |
| --- | --- |
| `outputs/m6/smoke/ckpt_last.pt` | 22587275 |
| `outputs/m6/smoke/ckpt_epoch0005.pt` | 22591430 |
| `outputs/m6/smoke/loss.png` | 32041 |
| `outputs/m6/smoke/train_log.jsonl` | 1061 |
| `outputs/m6/smoke/train_meta.json` | 1230 |
| `outputs/m6/smoke/val_epoch0005/eval.json` | 2984 |
| `outputs/m6/smoke/val_epoch0005/episodes.csv` | 344 |
| `outputs/m6/smoke/eval/eval.json` | 7884 |
| `outputs/m6/smoke/eval/episodes.csv` | 757 |

checkpoint 含在线权重、EMA、归一化缓冲、优化器、配置、epoch、训练 `source_id`、四元数翻转计数、损失历史、RNG、git commit 和 dirty。`--resume` 会核对 `source_id` 再继续。

## 正式训练建议

依据是 DPPO `cfg/robomimic/pretrain/can/pre_diffusion_unet_img.yaml`：2000 epoch，lr 1e-4，weight decay 1e-6，EMA 0.995，余弦 warmup，预测 ε，去噪 100 步；评测 DDIM 5 步。那边 batch 256、horizon 4、一个相机。这里 horizon 16、两个相机，batch 用 64。DDIM 用 8 步，落在要求的 5–10。

`configs/m6_dp.yaml` 已经是这组默认值。启动：

```sh
conda run -n AIControl python -u scripts/m6_train.py \
  --config configs/m6_dp.yaml --out outputs/m6/train
```

结束后：

```sh
conda run -n AIControl python -u scripts/m6_eval.py \
  --checkpoint outputs/m6/train/ckpt_last.pt
```

默认种子 0–49，超时 25 s（M5 示范最长 15.5 s）。

时长：每个 epoch 114 步（7234 / 64），步时 0.0149 s，2000 epoch 的训练约 **0.94 h**。验证每 100 epoch 跑 4 条。冒烟里一条跑满 25 s 的 episode 墙钟约 110 s。若 20 次验证、每次 4 条都超时，验证再加约 2.2 h，合计约 **3.2 h**。策略如果提前成功，验证会短于这个上界。数据加载另加约 10 s。

## 测试与回归

全部用 `conda run -n AIControl`。

- `python -m pytest tests -q --tb=line`：93 passed，9.79 s。原有 M0–M5 测试未删改。覆盖噪声调度、η = 0 确定性、DDIM 步数不改变形状、小数据上损失下降、UNet / ViT 形状与两种 batch、窗口首尾填充、归一化往返、子集可复现、第 8 步预览第 9 步、四元数、CUDA 逐位相同、checkpoint 往返逐位相同。
- `python scripts/check_env.py`：退出码 0，`ALL CHECKS PASSED`。
- `python scripts/m1_grasp_smoke.py`：`GRASP_OK`。
- `python -u scripts/m4_run.py --e1 --out outputs/m6/regression/m4_e1`：6 条 E1 全部 success，约 8 s。
- `python -u scripts/m4_analyze.py --runs outputs/m6/regression/m4_e1/runs --out outputs/m6/regression/m4_e1`。
- 与 `outputs/m4/summary.csv` 的 6 行 E1 比较：同名行的非空共有列，字符串或浮点逐项相同，mismatches 0。
- `tests/test_m6_replay.py`：`list_demos()[:2]` 用 `make_c3_env`、存储的初始状态和存储的 float32 动作 / `next_action` 重放。控制器是 C3。两条都再次成功。`tcp_pose`、`gripper_opening`、`q`、`qvel` 与存储 float32 逐位相同。

回归产物在 `outputs/m6/regression/m4_e1/`（`summary.csv` 8531 字节，6 个 npz 约 1.0–3.4 MB）。未写 `outputs/m3/`、`outputs/m4/`、`outputs/m5/`。`check_env.py` 和 `m1_grasp_smoke.py` 仍写它们自己的 `outputs/m0_wrist_cam.png` 和 `outputs/m1/`。

## 未完成或存疑

- 图像重放不是逐位相同。直接复位 `demo_0000` 时两路图像曾与存储值完全一致；pytest 里同一状态的离屏像素会有最多 1 个灰阶的差别，而且两次运行不是同一批像素。低维状态和成功判定是逐位 / 稳定的。测试因此要求像素最大差 ≤ 1，低维观测逐位相等，并要求两条示范再次成功。差超过 1 会失败。这是 EGL 重绘的 1 LSB，不是另一套控制器。
- 冒烟成功率是 0，失败全是 `ik_fail`。没有成功视频，因为没有成功的 episode。失败视频都在，且都小于 2 MB。
- `--live-viewer` 默认关闭。无显示环境，没有打开过 MuJoCo viewer。
- 正式 2000 epoch 训练没有跑，按派发留给用户。
- 冒烟的验证损失只用了 2 个 batch。配置默认 `val_batches: 0` 会走完 90 条验证示范。
