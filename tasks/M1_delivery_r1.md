# M1 交付记录（r1）

环境：`conda run -n AIControl`（Python 3.12.14，mujoco 3.17.0，numpy 2.5.4）。没有新装包，没有改 torch / mujoco / numpy 的版本。没有改 `docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`。没有 git commit / push / add。

## 改动文件

新增：

- `aloha/actuators_motor.xml`、`aloha/actuators_position.xml`
- `aloha/aloha_motor.xml`、`aloha/aloha_position.xml`（顶层：本体 + 对应执行器 + 场景）
- `assets/scene/pick_place_can.xml`、`assets/scene/README.md`、`assets/scene/LICENSE`
- `assets/scene/meshes/can.msh`
- `assets/scene/textures/`：`soda.png`、`light-wood.png`、`dark-wood.png`、`light-gray-floor-tile.png`、`ceramic.png`、`steel-brushed.png`
- `configs/m1_pick_place.yaml`
- `src/aloha_pp/robot.py`
- `src/aloha_pp/kin/ik.py`
- `src/aloha_pp/sim/env.py`、`controller.py`、`gravity.py`、`success.py`
- `scripts/m1_grasp_smoke.py`、`scripts/m1_gravity_hold.py`、`scripts/m1_ik_reach.py`
- `tests/conftest.py`、`tests/test_m1_model.py`、`tests/test_m1_ik.py`、`tests/test_m1_success.py`、`tests/test_m1_env.py`

修改：

- `aloha/aloha.xml`：去掉执行器 include；六个臂关节写入阻尼和 `actuatorfrcrange`；`<option timestep="0.002" integrator="implicitfast"/>`。armature、frictionloss、关节范围、position kp 保持 Menagerie 原值。
- `src/aloha_pp/kin/__init__.py`、`src/aloha_pp/sim/__init__.py`：导出 IK 和环境。

删除：

- `aloha/joint_position_actuators.xml`。仓库里没有其它代码再引用它。M0 的 `check_env.py` 和 `tests/test_smoke.py` 仍加载 `aloha/aloha.xml`，没有改测试路径。
- `assets/scene/.gitkeep`（目录里已经有正式资产）。

顶层模型必须放在 `aloha/`。`compiler meshdir="assets"` 是相对主 XML 所在目录解析的，不跟被 include 的文件走。放在别的目录时，机器人网格会去找 `aloha/vx300s_*.stl` 而失败。

## 运行命令和输出摘要

全部用 `conda run -n AIControl`。

`python -m pytest tests -q`

```
............                                                             [100%]
12 passed in 1.26s
```

含原来的 M0 测试。退出码 0。

`python scripts/check_env.py`

```
ALL CHECKS PASSED
```

腕部图仍是 `aloha.xml` 单独渲染：`min=0 max=45`。场景灯光在 `pick_place_can.xml` 里，不进这个检查。退出码 0。

`python scripts/m1_gravity_hold.py`（motor 版，`ctrl = qfrc_bias`，2 s）

```
pose home          shoulder ratio 0.1897  elbow 0.2162  max drift 1.76e-4
pose above_left    shoulder ratio 0.2430  elbow 0.2115  max drift 1.63e-4
pose above_right   shoulder ratio 0.2430  elbow 0.2115  max drift 1.63e-4
max_drift_rad 1.758456e-04 limit 0.01
```

腰和 wrist_rotate 的重力力矩比是 0，forearm_roll 约 0.0027，wrist_angle 约 0.0163。退出码 0。

`python scripts/m1_ik_reach.py`

```
grid left_pick:  49/49  worst_pos=9.84e-04 m  worst_rot=1.56e-03 rad
grid right_place: 49/49  worst_pos=9.84e-04 m  worst_rot=1.56e-03 rad
random 200: success 200/200  pos mean 4.25e-04 max 7.94e-04
                       rot mean 7.19e-04 max 1.28e-03
```

网格是两个 bin 地板的 7×7，z = 0.05 m，从上往下。随机 200 个同样是从上往下。退出码 0。

`python scripts/m1_grasp_smoke.py`（position 版 + IK，`fixed_1`）

```
video outputs/m1/grasp.mp4 bytes 48440 mb 0.048
agentview min 16 max 255  wrist min 11 max 255
can_z0 0.0360  hold_min_z 0.1254  hold_lift 0.0894  peak_lift 0.0963
hold_steps 60  hold_s 3.00
max_slip_m 0.002402
GRASP_OK
```

抬起 8.94 cm（≥ 5 cm），保持 3.00 s，保持期间相对 TCP 的位移 2.40 mm（< 5 mm）。截图：`outputs/m1/agentview.png`、`outputs/m1/wrist_cam_left.png`。退出码 0。

## 实测数据

### 夹爪开口

方法：把 `left/left_finger` 和 `left/right_finger` 都设到同一个 q（equality 是 1:1，`mj_forward` 不会替你改 qpos），再算两侧指尖碰撞球心的距离，减去两个半径（球半径 0.0006 m）。

| 状态 | 最近的一对球 | 内侧间距 |
|---|---|---|
| 运动学 q = 0.037 | `left_g0` / `right_g0` | 5.74 cm |
| 运动学 q = 0.002 | `left_g2` / `right_g2` | 0.98 cm |
| 动力学 ctrl = 0.037 | 同上 | 5.74 cm，q 停在 0.037 |
| 动力学 ctrl = 0.002 | `left_g0` / `right_g0` | 约 0，q 停在 0.00785 |

ctrl = 0.002 到不了：指尖球在 q ≈ 0.008 时已经碰上，接触力把 position 伺服挡住了。

### Can

网格来自 robosuite，XML 里 `scale="0.9 0.9 0.9"`。缩放后包围盒：x 4.497 cm、y 4.508 cm、z 7.207 cm。直径取 4.50 cm，高 7.21 cm。打开时间距 5.74 cm，每侧余量 (5.74 − 4.50) / 2 = 0.62 cm。见下面的存疑。

碰撞用方案 B。方案 A（缩放后的 mesh 直接碰撞）在夹持力下会慢慢陷入：手指 3 s 内继续合拢约 2 mm，罐子相对 TCP 滑了 8.86 mm，超过 5 mm。外观网格保留（`contype=0`），碰撞改成圆柱，半径 0.02251 m、半高 0.03604 m，中心与网格几何中心对齐。密度 100，质量 0.0115 kg。换完之后滑移 2.40 mm。

### 夹爪 forcerange

两个执行器文件里都是 ±20 N。依据：Menagerie 的手指 kp 是 2000 N/m，闭合指令 0.015 m 小于罐子半径 0.0225 m，所以接触后伺服会顶到力上限。保持阶段手指停在 q ≈ 0.027，未饱和的力需求大约是 2000 × (0.027 − 0.015) ≈ 24 N，20 N 刚好卡在这个量级。罐子重约 0.11 N。这是小夹爪的力，不是把力矩上限放大去硬过验收。

### TCP

用 `left/gripper`，不用 `ee_center_site`。`ee_center_site` 离两指垫中点大约 13 cm。`left/gripper` 在 site 系里离指垫中点是 (0.015372, 0, 0.004252) m：沿接近轴（site +X）偏后 1.54 cm，沿 site +Z 偏 4.3 mm。抓取时把 site 放在 `罐子中心 − R @ 这个偏移`。接近轴是 site 的 +X，从上往下时 +X 朝下。

### Bin 和可达性

robosuite 的 PickPlaceCan 用的是 BinsArena，没有单独的桌子。桌子从 `table_arena.xml` 改来，桌面顶在 z = 0，底座网格最低点也在 z = 0。

Bin 线性尺寸 × 0.36。地板全尺寸 0.144 × 0.180 m，顶面 z = 0。从机器人看（base 系 +X 向前、+Y 向左）：左 bin 中心 (0.30, 0.125)，右 bin 中心 (0.30, −0.125)。放置格是右 bin 的 +X、+Y 格，中心 (0.336, −0.080)。

两个 bin 的地板 7×7 网格（z = 0.05，朝向沿方位角从上往下）都是 49/49。随机 200/200。位置误差最大 0.98 mm，姿态误差最大 0.0016 rad。

### 重力力矩比

见上面的 gravity 输出。最大比是肩关节 0.243（侧向构型，−4.86 Nm / 20 Nm）。2 s 内最大关节漂移 1.76×10⁻⁴ rad。

### 抓取保持

抬起 8.94 cm，保持 3.00 s，相对 TCP 最大位移 2.40 mm。保持开始时手臂还在慢慢跟上目标（TCP 大约再升 6 mm），罐子跟着走，相对位移没有超标。

### 建议的成功判定

已经写进 `configs/m1_pick_place.yaml`：

- 目标格半尺寸 xy 用右 bin 内格：(0.0324, 0.0414) m；z 在静止几何中心上下 2 cm。
- 线速度 < 0.01 m/s（题目给定）。
- 连续 `hold_steps: 250` 个物理步，也就是 0.5 s。0.5 s 足够避开放下瞬间的弹一下，又不会把还在动的罐子算成功。

## robosuite

tag **v1.5.1**，commit `a071383d53568ab798eb315c0e95357911be922d`（`robosuite/__init__.py` 里 `__version__ = "1.5.1"`）。浅克隆，`git describe` 因为 grafted 失败，以这个 commit 为准。只 clone 在 `third_party/robosuite`，没有 pip 安装。运行时不引用 `third_party/`。

复制的文件和改动见 `assets/scene/README.md`。许可证是原文 MIT（`assets/scene/LICENSE`）。

## 重力补偿开关

`gravcomp` 是 body 属性，`actuatorgravcomp` 是 joint 属性。默认关。`apply_gravity_compensation` 会给 `left/*` body 打开 `body_gravcomp`，并给六个臂关节打开 `jnt_actgravcomp`。打开之后、跟踪误差为零时，`qfrc_actuator` 与 `qfrc_bias` 一致。测试在 `tests/test_m1_model.py`。M3 再用，配置项 `gravity_compensation: false`。

## 底层控制

`LowLevelController`：`reset` / `apply`。M1 的实现是 `PositionIkController`（position 执行器 + DLS IK）。`AlohaCanEnv` 接受外部 controller，M3 可以换掉。每个 `step` 走 25 个物理步（20 Hz / 500 Hz）。

## 存疑

1. 开口只有 5.74 cm。直径要落在 4.5–5 cm，同时又要每侧大约 1 cm 余量，这两条一起做不到（1 cm 余量要求直径 ≤ 3.7 cm）。取了区间下沿 4.50 cm，每侧余量 0.62 cm。没有改题目里的关节参数。
2. ctrl = 0.002 在动力学里到不了，手指在 q ≈ 0.00785 被自己的指尖球挡住。运动学距离和动力学停止点都写在上面。
3. 保持阶段 position 执行器还在以大约 2 mm/s 的速度往抬起目标爬，原因是摩擦和力矩上限下的稳态下垂，不是罐子从手里滑出去。相对位移仍是 2.4 mm。
4. `check_env.py` 的腕部图仍然偏暗（max 45），因为它只加载没有场景灯的 `aloha.xml`。带场景的两路 96×96 在冒烟测试里是 agentview 16–255、腕部 11–255。
