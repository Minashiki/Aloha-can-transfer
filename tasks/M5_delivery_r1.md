# M5 交付记录（r1）

工作树在 `fe540b49cb2291329422dbaac4498e1e3764b757` 上，未提交。数据集属性里的 `git_commit` 是这个 commit；生成时工作树已有本文件清单中的未提交改动。

## 1. 改动文件

新增：

- `configs/m5_retarget.yaml`、`configs/m5_noise.yaml`
- `src/aloha_pp/data/frames.py`、`retarget.py`、`dataset.py`、`README.md`
- `src/aloha_pp/planning/noise.py`
- `scripts/m5_generate.py`、`m5_validate.py`、`m5_analyze.py`、`m5_replay.py`
- `tests/test_m5_frames.py`、`test_m5_retarget.py`、`test_m5_noise.py`、`test_m5_dataset.py`、`test_m5_env.py`

修改：

- `src/aloha_pp/sim/env.py`：`reset` 增加仅关键字参数 `can_pose`、`state`。两者都缺省时仍走原来的采样和 50 步沉降。`capture_state` 供重放。
- `src/aloha_pp/data/__init__.py`：导出 `load_episode`、`validate_dataset`。

未改增益、力矩上限、速度上限、成功阈值、`random_can`、`gripper_open` / `gripper_close`，也未改禁止清单里的文件。运行时不 import robomimic / robosuite，不引用 `third_party/`。

## 2. 命令与输出摘要

环境均为 `conda run -n AIControl`。未安装任何包。

下载（见第 3 节）完成后：

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_generate.py
```

`outputs/m5/timing.json`：重定向 291.1 s，规划器噪声 13.1 s，带图像重放写入 266.3 s，合计 570.7 s（约 9.5 分钟）。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_validate.py outputs/m5/dataset/demos.hdf5
```

`OK`。长度 n=113，min 214，median 486，mean 481.9，max 1127。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_validate.py outputs/m5/dataset/planner_noise.hdf5
```

`OK`。n=8，min 120，median 135.5，max 159。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_analyze.py
```

重定向 122/200 = 0.610。原因：success 122，outside 43，ik_unreachable 32，dropped 3。主数据集 `retarget` 113 条。规划器噪声 8/8 成功，写入备用文件。图在 `outputs/m5/figures/`。

```text
conda run -n AIControl --no-capture-output python -u scripts/m5_replay.py
```

5 条全部再次成功。关节位置最大偏差 5.545e-4 rad（`replay demo_0001`，retarget id 3）。五条偏差为 6.17e-5、2.13e-4、5.55e-4、2.11e-4、2.42e-4。

```text
conda run -n AIControl --no-capture-output python -m pytest tests -q --tb=line
```

`75 passed in 6.05s`。

```text
conda run -n AIControl --no-capture-output python scripts/check_env.py
```

退出码 0，`ALL CHECKS PASSED`。该脚本会写 `outputs/m0_wrist_cam.png`，这是它原有行为。

```text
conda run -n AIControl --no-capture-output python scripts/m1_grasp_smoke.py
```

打印 `GRASP_OK`。视频写到 `outputs/m1/`，未动 `outputs/m3/`、`outputs/m4/`。

```text
conda run -n AIControl --no-capture-output python scripts/m4_run.py --e1 --out outputs/m5/regression/m4_e1
conda run -n AIControl --no-capture-output python scripts/m4_analyze.py --runs outputs/m5/regression/m4_e1/runs --out outputs/m5/regression/m4_e1
```

6 组 E1 全部成功。与 `outputs/m4/summary.csv` 的 E1 行比较 75 个共有字段，最大绝对差为 0。

## 3. 原始数据

当前文档（robomimic 0.5，<https://robomimic.github.io/docs/datasets/robomimic_v0.1.html>）把仿真数据放在 Hugging Face `robomimic/robomimic_datasets` 的 `v1.5/can/ph/low_dim_v15.hdf5`。直接地址 `https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/can/ph/low_dim_v15.hdf5` 连接超时（curl 28）。斯坦福旧地址上的 `low_dim_v15.hdf5` 返回 HTTP 404。

任务允许 v1.4.1。robomimic v0.3 登记的 Can-PH low_dim 地址可用：

- URL：`http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/low_dim_v141.hdf5`
- 版本：v1.4.1（文件名 `low_dim_v141.hdf5`），200 条，low_dim，约 20 Hz
- 大小：48293744 字节
- sha256：`6ce6e6c7f60b04e3fd180327de7c2dbe456373efe4b2682299cb2350fdc68cdd`
- 路径：`datasets/robomimic/can/ph/low_dim_v141.hdf5`
- 许可证：发布该 URL 的 robomimic 代码为 MIT（Copyright 2021 Stanford Vision and Learning Lab）。下载页没有单独的数据许可证字段。

命令：`curl -L -C - --connect-timeout 30 --max-time 240 -o datasets/robomimic/can/ph/low_dim_v141.hdf5 http://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/low_dim_v141.hdf5`，退出码 0。

文件内 `data/demo_0` … `demo_199`。观测含 `robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`、`object`（14 维，前 7 维是 Can 的位置和四元数）。不读 OSC 增量动作。夹爪开度（两指绝对位置之和）最小 0.030，p10 0.049，中位 0.058，p90 0.079，最大 0.080。每条的最小值都是 0.0417 左右；开头 3 帧是复位姿态，也停在这个宽度。

## 4. 坐标映射

数值来自阅读 `third_party/robosuite`（v1.5.1）后写入 `src/aloha_pp/data/frames.py`，运行时不引用该目录。

- `pick_place.py` 默认：起始 bin `bin1_pos=(0.1, -0.25, 0.8)`，目标 bin `bin2_pos=(0.1, 0.28, 0.8)`，`table_full_size=(0.39, 0.49, 0.82)`。
- `object_to_id["can"]=3`。`_construct_objects` 顺序是 Milk、Bread、Cereal、Can，所以 `_setup_references` 里 can 的 `bin_id=3`。该分支不加半边长，再加 `size/4`。目标格中心为 `(0.1+0.39/4, 0.28+0.49/4, 0.8) = (0.1975, 0.4025, 0.8)`，相对 bin2 是 `+X+Y`。与 `assets/scene/pick_place_can.xml` 里 “Robosuite Can target is the +x,+y cell of bin2” 以及本项目 `fixed_1` 一致。
- `panda_robot.py` 的 `base_xpos_offset["bins"]=(-0.5, -0.1, 0)`。Panda 朝 `+X`、`+Z` 向上，与 `left/base_link` 轴向相同。

两格都是各自 bin 的 `+X+Y` 象限，所以旋转取单位阵，`det(R)=+1`。平移把目标格中心送到 `fixed_1`：

```text
t = [0.1385, -0.4825, -0.76396]
p_base = p_robo + t
```

测试容差 1e-9，往返误差在该量级内。两个 bin 中心相距 0.53 m，本项目是 0.25 m，一个刚体变换不能同时送中两个 bin 中心。起始 bin 在机器人右侧、本项目起始 bin 在左侧，对调左右需要镜像，这里不用镜像。Can 的初始 xy 不用这条刚体映射，而用 bin 内分数（见下）。

## 5. 重定向

实现在 `src/aloha_pp/data/retarget.py`。

1. 分段。开度小于 `gripper_close_width` 0.055，且连续至少 `min_close_frames` 8 帧，才算抓住。0.055 是因为有 6 条示范的真实抓取开度落在 0.05 和 0.055 之间；8 帧用来跳过开头 3 帧复位。松开点是这段之后的第一帧张开；若不再张开则取最后一帧。
2. 抓取前：`tcp = can_new + (eef - can)`，Can 坐标系平移，距离不乘 0.36。
3. 搬运：去掉从抓住点到松开点的源直线平移，残差加在本项目抓住点与松开点的 smoothstep 上，因此不复制 0.53 m 的 bin 间距。中段若低于 `carry_clearance_z` 0.10 m，用正弦包络抬高，端点不动。bin 壁顶约 0.029 m。
4. 松开点。源数据在夹爪张开时 Can 仍高出目标格约 0.24 m（中位），手并不跟着降到格内；示范结束时 Can 相对格心的 xy 距离中位 4.6 cm。本项目格子半宽只有 3.24 cm / 4.14 cm，这个落点进不了格子。因此在源松开姿态之后，先在该高度平移到格子正上方，再下降到 `fixed_1` 中心加上抓取时的手–Can 偏移，z 再加 `place_gap` 0.006 m（与 M4 相同，写在 m5 配置里）。这段夹爪保持闭合。然后才张开，并上抬 `retreat_z` 0.08 m，让手指离开 Can。
5. Can 初始 xy。相对 `bin1` 中心的分数，分母是 `table_full_size/2`，再放进 `random_can` 矩形（中心就是左 bin 中心）。越界则夹紧并记 `clamped`。200 条都没有越界。z 用 `can_rest_z`。姿态在本项目里用单位四元数，Can 是竖直圆柱。
6. 工具系。`robot0_eef_pos/quat` 是 Panda `grip_site`（指尖之间，`+Z` 为接近方向）。本项目 `left/tcp` 是两指垫中点，`+X` 为接近方向。两者都是抓取中心，位置偏移直接沿用。姿态不复制：全程 `top_down_quat(choose_yaw(can_xy), approach_axis=0)`。Can 是圆柱，人的 yaw 不改变接触；M4 已确认 `choose_yaw` 的 0 或 π/2 能避开近侧壁并且 IK 可达。
7. 动作。第 t 拍是第 t+1 拍的重定向位姿 `[xyz, quat wxyz, gripper]`，最后一拍重复前一拍。与 M4 “命令提前一拍、抵消 preview 的 50 ms” 相同。执行时 `env.step(action, next_action=...)`，末拍的 `next_action` 是它自己。
8. 夹爪。开/合命令是 0.037 / 0.015。在闭合和张开处插入与 M4 相同的等待：preview 1 拍 + `settle_s` 0.25 s，线性过渡 12 拍，闭合后停留 0.40 s，张开后停留 0.20 s。这样手指不会在手臂按 preview 到达之前动作。
9. 时间。从 home TCP 先有 0.5 s 直线接近，再对整条轨迹做 IK，按关节差分峰值放大时间，直到峰值不超过 `0.5 * v_max`（`v_max` 为 waist/shoulder/elbow 2 rad/s，其余 3 rad/s）。系数是各次测量的 `peak / (0.5 v_max)` 连乘。成功示范的系数：最小 1.17，p10 1.68，中位 2.55，p90 3.17，最大 5.42。IK 位置误差超过 0.01 m 时不再放大时间（放大也不能到达），超过 `ik_skip_pos_err` 0.002 m 则记 `ik_unreachable`，不执行。
10. 闭环与失败。C3 + preview。失败分类沿用 M4 的 `classify_episode`，并增加 `no_grasp_segment`、`ik_unreachable`、`can_out_of_bin`。本次没有 `can_out_of_bin`。`outside` 是 M4 的“未进格子”。

成功示范的 IK 峰值 / 速度上限：最小 0.481，中位 0.4994，p90 0.4999，最大 0.5002（demo 182，四位小数）。有 1 条印成 0.5002、几条印成 0.5001，来自最后一次重采样后有限差分没有正好落在 0.5 以下。闭环里记录的关节速度（`qvel`）峰值约为速度上限的 1.03 倍，那是跟踪控制器的实际速度，不是 IK 参考。

成功示范的最终放置 xy 误差（相对格子中心）：最小 0.6 mm，中位 4.9 mm，p90 9.3 mm，最大 22.9 mm，都在格子半宽内。

## 6. 规划器噪声

`NoisyPickPlacePlanner` 在 M4 规划器之上。幅度全 0 时 `reset` / `_rebuild_carry` 只调用父类，无额外运算。非零时噪声加在路点和时间律上：抓取/放置 xy 各 4 mm（小于 Can 半径 22.5 mm，也小于格子半宽），抬起高度 5 mm，yaw 0.10 rad，垂直于搬运方向的经过点 8 mm（抬起点全额、悬停点一半），七段时长各自乘 `Uniform(0.85, 1.15)`。medium 的参考峰值约在 0.5 倍速度上限，加快 15% 仍低于上限。种子固定则可复现。Can 用 `random` 模式采样。

重定向成功 122 条，不少于 50，所以主数据集只用重定向。噪声示范另存 `outputs/m5/dataset/planner_noise.hdf5`，种子 1000–1007，8/8 成功。`slowdown` 记的是各段时间因子的最大值。

## 7. 数据集

格式说明在 `src/aloha_pp/data/README.md`。

- 主文件 `outputs/m5/dataset/demos.hdf5`：113 条，全部 `source=retarget`。122 次无渲染成功里有 9 条在写图像的第二次执行中不再成功（demo 5、12、50、80、127、144、160、167、177），未写入。
- 备用 `planner_noise.hdf5`：8 条 `planner_noise`。
- 只存成功示范，在成功判定成立的那一拍结束。失败只出现在 `outputs/m5/episodes.csv`。
- 每拍观测是动作之前的状态。特权信息是 Can 位姿。动作 T×8。属性含来源、原示范号或种子、放慢系数、初始 `qpos/qvel/ctrl`、成功标志和成功拍号。
- 根属性：20 Hz，相机名与 96×96，base 系绝对位姿、wxyz、提前一拍，参考模式 preview，git commit，两份 m5 配置的 JSON。

读取：`aloha_pp.data.load_episode`。校验：`scripts/m5_validate.py`（形状、类型、图像非空、四元数单位长、长度统计）。测试里构造了非单位四元数和全零图像，校验脚本能报出来。

重放从存储的初始状态出发，同样的 `next_action` 规则。抽样 5 条：备用文件的 planner seed 1000，以及主文件 retarget id 0、3、6、7。5/5 再次成功。

视频在 `outputs/m5/videos/`，agentview，480×360（编码时 ffmpeg 把高度补到 368），单条 80–243 KB，均小于 2 MB。`obs_*.png` 是中段 96×96 的 agentview 与 wrist 横向拼接（192×96）。

## 8. 输出文件（不进 git）

| 字节 | 路径 |
|---:|---|
| 48293744 | `datasets/robomimic/can/ph/low_dim_v141.hdf5` |
| 831631721 | `outputs/m5/dataset/demos.hdf5` |
| 17096748 | `outputs/m5/dataset/planner_noise.hdf5` |
| 16525 | `outputs/m5/episodes.csv` |
| 13503 | `outputs/m5/generate.log` |
| 311 | `outputs/m5/timing.json` |
| 245 | `outputs/m5/summary.csv` |
| 31248 | `outputs/m5/figures/can_xy.png` |
| 22819 | `outputs/m5/figures/retarget_reasons.png` |
| 14422 | `outputs/m5/figures/length.png` |
| 13518 | `outputs/m5/figures/place_xy.png` |
| 13163 | `outputs/m5/figures/ik_peak_ratio.png` |
| 10210 | `outputs/m5/figures/slowdown.png` |
| 80446 | `outputs/m5/videos/replay_00_planner_noise_1000.mp4` |
| 205073 | `outputs/m5/videos/replay_01_retarget_0.mp4` |
| 118543 | `outputs/m5/videos/replay_02_retarget_3.mp4` |
| 192533 | `outputs/m5/videos/replay_03_retarget_6.mp4` |
| 242666 | `outputs/m5/videos/replay_04_retarget_7.mp4` |
| 25220 | `outputs/m5/videos/obs_00_planner_noise_1000.png` |
| 24230 | `outputs/m5/videos/obs_01_retarget_0.png` |
| 25319 | `outputs/m5/videos/obs_02_retarget_3.png` |
| 25058 | `outputs/m5/videos/obs_03_retarget_6.png` |
| 24278 | `outputs/m5/videos/obs_04_retarget_7.png` |

`outputs/m5/regression/m4_e1/` 另有 6 组 E1 的 json/npz 和 `summary.csv`（8531 字节）等表。

## 9. 存疑

- v1.5 的 Hugging Face 文件没有下下来，用的是官方 v1.4.1。没有改用其他任务或其他镜像。
- 成功示范里 demo 182 的 IK 参考峰值是速度上限的 0.5002 倍（四位小数），略高于 0.5。其余成功示范不超过 0.5001。差在最后一次时间校正后的有限差分。
- 9 条无渲染时成功、写图像时不再成功，已从数据集剔除。接触在两次执行之间有发散。主文件里的 113 条是写图像那次仍然成功的。
- 刚体映射只把 Can 的目标格对齐到 `fixed_1`。两个 bin 中心不能用同一个保向刚体变换对齐，也没有做镜像。
- 数据集里的 git commit 是干净的 `fe540b4`，生成时工作树是脏的。未执行 commit / push。
