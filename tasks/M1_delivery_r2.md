# M1 交付记录（r2）

环境：`conda run -n AIControl`（Python 3.12.14，mujoco 3.17.0，numpy 2.5.4）。没有新装包，没有改 torch / mujoco / numpy 的版本。没有改 `docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`。没有 git commit / push / add。没有改 IK 的姿态误差和 Jacobian。timestep 仍是 0.002 s，每个动作仍是 25 个物理步。

## 改动文件

本轮修改：

- `aloha/aloha.xml`：在 `left/gripper_link` 下新增 site `left/tcp`。
- `assets/scene/pick_place_can.xml`：agentview 后移并抬高。
- `assets/scene/README.md`：相机位姿，以及 TCP 坐标轴约定。
- `configs/m1_pick_place.yaml`：`tcp_site: left/tcp`；删掉 `tcp_to_grasp_center` 和 `can_center_local`；`can_rest_z` 改成碰撞圆柱底面贴地；目标格按 mode 分开。
- `src/aloha_pp/robot.py`：删掉错误的 `TCP_SITE` / `TCP_SITE_CANDIDATES`，以及和 XML、配置重复的力矩、阻尼、增益、速度常量。关节名和路径还留着。
- `src/aloha_pp/sim/env.py`：TCP 观测、物体中心、成功判定、IK 返回值、运动学同步、上下文管理器。
- `src/aloha_pp/sim/success.py`：松手条件，以及 `tcp_local_offset`。
- `scripts/m1_grasp_smoke.py`、`scripts/m1_gravity_hold.py`、`scripts/m1_ik_reach.py`。
- `tests/test_m1_env.py`、`tests/test_m1_model.py`、`tests/test_m1_success.py`。

本轮新增：

- `tasks/M1_delivery_r2.md`。

本轮没有删文件。`tcp_to_grasp_center`、`can_center_local`、`_mesh_aabb_center` 是从已有文件里删掉的符号。

截图和视频在 `outputs/m1/`（该目录被 gitignore）：`agentview_reset_96.png`、`agentview_reset_256.png`、`agentview_grasp.png`、`grasp.mp4`。

## 运行命令和输出摘要

全部用 `conda run -n AIControl`。

`python -B -m pytest tests -q -p no:cacheprovider`

```
....................                                                     [100%]
20 passed in 1.67s
```

含原来的 M0 测试。退出码 0。

`python -B scripts/check_env.py`

```
ALL CHECKS PASSED
```

腕部图仍是只加载 `aloha.xml`：`min=0 max=45`。退出码 0。

`python -B scripts/m1_grasp_smoke.py`

```
video outputs/m1/grasp.mp4 bytes 56260 mb 0.056
can_z0 0.0360 hold_min_z 0.1211 hold_lift 0.0851 peak_lift 0.0953
hold_samples 1500 hold_actions 60 hold_s 3.00
max_slip_m 0.002780
ik_failures 0
GRASP_OK
```

抬起 8.51 cm（≥ 5 cm），保持 3.00 s，1500 个物理步里的局部滑移 2.78 mm（< 5 mm）。退出码 0。

`python -B scripts/m1_gravity_hold.py`

```
ik_failures 0
pose home          shoulder ratio 0.1897  elbow 0.2162  drift max 1.76e-4
pose above_left    shoulder ratio 0.2424  elbow 0.2136  drift max 1.62e-4
pose above_right   shoulder ratio 0.2424  elbow 0.2136  drift max 1.62e-4
max_drift_rad 1.758456e-04 limit 0.01
```

退出码 0。力矩上限从模型的 `jnt_actfrcrange` 读，不再用 `robot.py` 里的副本。

`python -B scripts/m1_ik_reach.py`

```
bounds left x[0.25411,0.34589] y[0.06111,0.18889] z=0.03604
bounds right_nxny x[0.25411,0.27389] y[-0.18889,-0.15111]
bounds right_nxpy x[0.25411,0.27389] y[-0.09889,-0.06111]
bounds right_pxny x[0.32611,0.34589] y[-0.18889,-0.15111]
bounds right_pxpy x[0.32611,0.34589] y[-0.09889,-0.06111]
grid left_pick:   49/49  worst_pos=9.13e-04  worst_rot=1.51e-03
grid right_nxny:  49/49  worst_pos=8.76e-04  worst_rot=1.47e-03
grid right_nxpy:  49/49  worst_pos=2.58e-04  worst_rot=4.65e-04
grid right_pxny:  49/49  worst_pos=9.83e-04  worst_rot=1.59e-03
grid right_pxpy:  49/49  worst_pos=3.74e-04  worst_rot=6.57e-04
random 200: success 200/200  pos mean 4.59e-04 max 9.71e-04
                       rot mean 7.88e-04 max 1.56e-03
ik_failures 0
```

退出码 0。IK 容差仍是配置里的 1 mm / 0.01 rad，没有改。

## 各项验收

### 1. TCP

`left/tcp` 挂在 `left/gripper_link` 上，`pos="0.143809 0.000003 -0.003550"`，没有额外旋转。坐标轴就是这个 link 的轴：+X 是接近方向。注释写在 `aloha/aloha.xml`，`assets/scene/README.md` 也指到了这里。

计算方法：两侧指垫各取 `left/*_g0`、`g1`、`g2` 三个碰撞球心的平均，再取这两个平均的中点，和 `site_xpos` 做距离。内侧开口是最近一对球心的距离减去两个半径（球半径 0.0006 m）。

| 状态 | q | 内侧开口 | 到中点的距离 |
|---|---|---|---|
| 张开 | 0.037 | 5.74 cm | 3.78e-7 m |
| 开口等于碰撞圆柱直径 4.502 cm | 0.03081 | 4.502 cm | 3.78e-7 m |

两指对称，中点不随 q 移动，所以两个距离一样，都远小于 1 mm。`tests/test_m1_model.py` 里的 `test_tcp_site_is_pad_midpoint_open_and_on_can` 用的是同一算法。抓取脚本不再加偏移，TCP 直接对准罐子中心。

### 2. agentview

相机改到 `pos="0.71 -0.18 0.416"`，`fovy="48"`，视线朝向 `(0.26, 0, 0.14)`。相机 +X 向右，+Y 向上，沿 -Z 看。

外沿顶角取每只 bin 的 y 向墙外侧面四个角。墙半厚 0.0036 m，所以外侧面在地板中心 ±0.0936 m（y）和 ±0.0756 m（x），墙顶在 body 系 z = 0.036 m。投影：点变到相机系（`cam_xmat` 的列是相机轴），深度是 `-z`，归一化坐标是 `(x/depth, y/depth) / tan(fovy/2)`。方形成像，和分辨率无关。

```
bin_left  x- y-   0.001866  -0.182663
bin_left  x- y+   0.525659  -0.064142
bin_left  x+ y-   0.226419  -0.499565
bin_left  x+ y+   0.830989  -0.327113
bin_right x- y-  -0.861054  -0.377920
bin_right x- y+  -0.195188  -0.227251
bin_right x+ y-  -0.817313  -0.797288
bin_right x+ y+  -0.006564  -0.566023
```

绝对值最大 0.861，都在 [-1, 1] 内。截图：`outputs/m1/agentview_reset_96.png`、`outputs/m1/agentview_reset_256.png`、`outputs/m1/agentview_grasp.png`（闭合停在罐子上的那一帧，256×256）。

### 3. 目标格

`fixed_1` 和 `random` 仍是右 bin 的 +X+Y 格，中心 `(0.336, -0.080, 0.03604)`，半尺寸 `(0.0324, 0.0414, 0.02)`。`fixed_2` 是对角的 −X−Y 格，中心 `(0.264, -0.170, 0.03604)`，半尺寸相同。`fixed_2` 的 `place_pos` 是 `(0.254, -0.180, 0.03604)`，在这个格子里。`test_target_cell_follows_mode` 覆盖了两个 mode。

### 4. 松手

`success.require_release` 默认 true。两个手指 body 上的全部 geom 和 `can_collision` 都没有接触，才算松开。成功判定在每个物理步、`mj_forward` 刷新接触之后评估。

`test_release_gate_in_cell_released_and_outside`：夹着并静止在格内不成功；松手并静止在格内，满 `hold_steps` 后成功；格外即使松手也不成功。`require_release: false` 时夹着也可以成功，用来确认这个开关真的关掉了条件。

### 5. IK 状态

`step` 返回 `(obs, success, info)`。`info` 有 `ik_success`、`ik_pos_err`、`ik_rot_err`。容差仍是配置里的 0.001 m / 0.01 rad。`m1_grasp_smoke.py`、`m1_ik_reach.py`、`m1_gravity_hold.py` 都打印 `ik_failures`。这次三次都是 0。

### 6. 状态同步

`mj_step` 之后，读 xpos、site、接触之前先 `mj_forward`。成功判定的时刻写在 `AlohaCanEnv.step`：每个物理步积分之后、刷新派生量之后，用这一步的罐子中心、线速度和接触来计数。reset 的 50 步沉降之后、返回观测之前也刷新。没有改 timestep，也没有改 25 步。

`test_motion_obs_matches_independent_fk`：6 个会让手臂离开 home 的动作之后，返回的 TCP 与把同一 `qpos` 拷到新的 `MjData` 再 `mj_forward` 的结果一致（位置 atol 1e-9，四元数点积的绝对值与 1 的差 < 1e-9）。特权位置等于 freejoint 位姿乘上碰撞圆柱的 `geom_pos`。

### 7. 物体中心

中心是 `can_collision` 的 `geom_xpos`，姿态用这个 geom 的 `geom_xmat`。删了 `_mesh_aabb_center` 和 `can_center_local`。`can_rest_z` 改成 0.03663，使圆柱底面（局部 z = -0.00059 - 0.03604）落在地板 z = 0。静止中心高度仍是 0.03604 m。

`test_can_center_matches_collision_geom_at_two_orientations`：单位四元数，以及绕 X 转 90°。两种朝向下，`can_center_base` 转回世界系后与直接读的 `geom_xpos` 差在 1e-12 m 以内，并且和 body 原点不是同一个点。

### 8. IK 采样边界和失败退出

边界从 geom 读出来。地板半尺寸减去墙的薄半尺寸（内壁）再减去圆柱半径 0.02251 m。右 bin 再把隔板半厚加半径从中心线两侧挖掉，剩四个矩形。高度 z = 0.03604 m，是地板顶面加圆柱半高。任一目标不收敛都先打完统计，再以退出码 1 结束。

注入没有留在仓库里。一次性做法是 `runpy.run_path("scripts/m1_ik_reach.py")`，把 `DlsIk.solve` 换成：先调用原来的求解，第 1 次返回值用 `dataclasses.replace(..., success=False)`。输出仍有全套统计，`grid left_pick: 48/49`，`ik_failures 1`，`SystemExit` 码 1。

### 9. 滑移

保持阶段每个物理步记录 `R_tcp.T @ (p_can - p_tcp)`，再和保持开始的那一次做差。环境的 `step` 有可选的 `on_substep`，在每个物理步刷新运动学之后调用。这次 1500 个样本，最大滑移 2.78 mm。

`test_rigid_tcp_motion_has_zero_slip`：物体固定在 TCP 系里，TCP 平移并改变 yaw，滑移 < 1e-12 m。把最后一个样本沿局部轴挪 6 mm，滑移超过 5 mm，度量能看出来。

### 10. 低优先级

`robot.py` 里和 XML / 配置重复的数值常量已删。`tests/test_m1_model.py` 里的力矩、阻尼、kp、kv 仍是测试自己写的期望值。`AlohaCanEnv` 支持 `with`，`__exit__` 调用 `close`。`test_context_manager_closes_renderer` 检查渲染器被关掉。

## 存疑

1. 开口余量仍是 r1 已接受的 0.62 cm（直径 4.50 cm）。这次没有改夹爪几何。
2. agentview 里两个 bin 的顶角 y 坐标在 -0.80 到 -0.06，也就是画面下半部；上半部是手臂。外沿在画面内，抓取帧里夹爪和罐子也在画面内。
3. 重力脚本里 `above_left` / `above_right` 的肩力矩比从 r1 的 0.2430 变成 0.2424。同一个笛卡尔目标现在对的是 `left/tcp`，关节解稍有不同。home 的比值和 1.76e-4 rad 的最大漂移没有变，仍低于 0.01 rad。
