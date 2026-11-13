# M1 派发：模型与场景

**先读这份交接单：`tasks/handover_M1_1.md`。**

项目根目录：`/home/minashiki/AIControl/FinalHomework`。背景见 `docs/PLAN.md` 第 2–4 节和第 10 节 M1（只读）。

## 硬性约束（违反任何一条即判为不合格）

1. **Python 环境**：所有安装、运行、测试只能在 conda 环境 `AIControl` 中进行（`conda run -n AIControl ...`）。禁止用系统 Python、base 环境，禁止新建环境。交付记录里的命令要能看出用的是这个环境。不得改动已安装包的版本；如果确实需要新包，先在交付记录里说明理由，并且不得升级或降级 torch / mujoco / numpy。
2. **不得修改**：`docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml` 中已有的版本号、`scripts/check_env.py` 已有的检查项。
3. **git**：不得执行 `git commit` / `push` / `config` / `add`，不得添加任何 co-author / contributor。
4. 不得启动其他 agent。不得把 robosuite / robomimic 安装为依赖包。下载的 robosuite 源码只能放在 `third_party/`（已被 .gitignore 排除），所需资产**复制**到 `assets/scene/` 后使用，运行时不得引用 `third_party/`。
5. 耗时操作只做冒烟级别：单个脚本几分钟内跑完。

## 固定参数（来自 PLAN，不得自行修改；如果实测发现不合理，写进交付记录的“存疑”一节）

| 关节 | 力矩上限 `actuatorfrcrange`（Nm） | 阻尼 damping | 速度上限（rad/s，配置项） | position 版 kv |
|---|---|---|---|---|
| waist | ±10 | 1.0 | 2.0 | 5.76 |
| shoulder | ±20 | 2.0 | 2.0 | 20.0 |
| elbow | ±20 | 2.0 | 2.0 | 18.49 |
| forearm_roll | ±10 | 0.8 | 3.0 | 6.78 |
| wrist_angle | ±10 | 0.8 | 3.0 | 6.28 |
| wrist_rotate | ±4 | 0.3 | 3.0 | 1.2 |

- armature、frictionloss、joint range 保持 Menagerie 原值；position 版 kp 保持 Menagerie 原值（43 / 265 / 227 / 78 / 37 / 10.4）。
- `timestep=0.002`，`integrator="implicitfast"`。
- 动作频率 20 Hz，物理 / 控制频率 500 Hz，即每个动作 25 个物理步。

## 要做的事（按顺序）

### 1. 执行器文件与模型修改（`aloha/`）

- 从 `aloha.xml` 中去掉对执行器文件的 include，让它只描述机器人本体；在本体里设置阻尼、`actuatorfrcrange`、`<option timestep integrator>`。
- 新建 `aloha/actuators_motor.xml`：6 个 motor，`gear=1`，`ctrlrange` 与力矩上限一致；夹爪使用 position 执行器（kp 2000，ctrlrange 0.002–0.037），并加 `forcerange`（取值见第 3 步）。
- 新建 `aloha/actuators_position.xml`：6 个 position 执行器（Menagerie kp + 上表中的 kv），夹爪同上。
- 删除旧的 `aloha/joint_position_actuators.xml`（名字和内容不符），并检查仓库里有没有其他地方引用它（M0 的 `check_env.py` / 测试加载的是 `aloha.xml`，改完后要保证仍然通过；如果需要改测试里加载的模型路径，在交付记录里说明）。
- 顶层模型：`motor` 和 `position` 两个版本，各自包含 “机器人 + 对应执行器 + 场景”。文件位置和 include 方式自定，注意 meshdir 的解析问题（见交接单第 4 节）。
- 确认 B1 用到的 `gravcomp`（body 属性）和 `actuatorgravcomp`（joint 属性）可以在 position 版上用配置开关开启。默认不开，M3 再用。

### 2. 场景（`assets/scene/`）

- 从 robosuite 源码（`git clone` 到 `third_party/robosuite`，固定一个 tag 或 commit，并在交付记录中写明）中找出 PickPlaceCan 所用的 arena（地面、桌子、bin）、Can 物体、相机定义。复制所需资产到 `assets/scene/`，附上 robosuite 的 `LICENSE`（MIT），另写一个 `assets/scene/README.md` 说明每个文件的来源路径、改动和缩放比例。
- robosuite 中有的优先用 robosuite 的；robosuite 中没有的再自己写（在 README 中注明）。
- 布置：机器人 base 放在桌面高度。从机器人视角看，左 bin 用于取，右 bin 用于放，放的目标为右 bin 中的一个目标格。bin 按 ALOHA 工作空间平移 / 缩放，使两个 bin 的**整个可用区域**内夹爪都能从上往下抓到（用第 4 步的 IK 实测，并在交付记录中给出可达性结果）。
- `agentview` 相机：取 robosuite 的定义，调整位置，使两个 bin 和机械臂都在画面内。
- 光照：保证 96×96 的两路相机图像能看清 Can（M0 的腕部图像偏暗）。

### 3. Can 与夹爪实测

- 测量夹爪的有效开口：ctrl = 0.037 和 0.002 时，两侧指尖内侧之间的实际距离（用 site 或几何体位置计算，写明方法）。
- 根据开口确定 Can 的**等比例**缩放系数（目标直径约 4.5–5 cm，打开时每侧至少留约 1 cm 余量），写明最终直径和高度。
- 碰撞几何：先用方案 A（缩放后的 mesh）；如果抓取时抖动或穿透，改用方案 B（外观不变，碰撞几何改成 primitive 圆柱），在交付记录中说明选择的理由。
- 夹爪 `forcerange`：取一个接近真实夹爪量级的值（写明依据），并保证能稳定抓住 Can。夹紧时的 ctrl 目标要小于 Can 半径，才能产生夹持力。

### 4. 逆运动学（`src/aloha_pp/kin/`）

- 基于 `mj_jacSite` 的 DLS IK：输入目标位置 + 姿态（基座坐标系），以当前 q 为初值迭代，结果限制在关节范围内。阻尼系数、迭代次数、收敛阈值都作为参数。
- TCP：在 `ee_center_site` 和 `left/gripper` 中选一个作为统一的 TCP（应位于两指尖之间的抓取中心），写明理由。
- 测试：在两个 bin 区域内随机取 ≥ 200 个从上往下的抓取位姿，统计收敛率、位置误差、姿态误差。

### 5. 环境类（`src/aloha_pp/sim/`）

- 配置文件放在 `configs/`（YAML）：执行器版本、取放位置、随机范围、成功判定阈值、相机分辨率等。
- `reset(seed, mode)`：`mode` 为 `fixed_1`、`fixed_2`（两组固定的取 / 放位置，写进配置）或 `random`（Can 在左 bin 内随机）。
- `step(action)`：动作 = 基座坐标系下的绝对末端位姿（xyz + 四元数）+ 夹爪开度。执行 25 个物理步。M1 的底层控制暂用 **position 版执行器 + IK**，M3 再换成自写控制器，所以底层控制要做成可替换的接口。
- 观测：`agentview` 和 `wrist_cam_left` 两路 96×96 RGB，加上本体状态（TCP 位姿、夹爪开度、q、q̇）。另外单独提供一个特权信息接口（物体位姿），供非对称 critic 和评测使用，不放进策略观测。
- 成功判定：Can 中心在目标格范围内，线速度 < 1 cm/s，且连续保持 N 步（N 和各阈值都是配置项，写出你建议的取值）。
- 渲染使用 `MUJOCO_GL=egl`。

### 6. 冒烟测试与验收脚本

- `scripts/m1_grasp_smoke.py`：在 `fixed_1` 下用脚本化的路点（IK + position 版）完成：接近 → 下降 → 闭合 → 抬起 10 cm → 保持 ≥ 3 s。记录 Can 相对 TCP 的位移（判断是否滑落）。输出一个小视频到 `outputs/m1/`（mp4，≤ 2 MB）和双相机截图。
- `scripts/m1_gravity_hold.py`：motor 版，ctrl = `qfrc_bias`（重力 + 科氏力补偿），从 ≥ 3 个构型开始，2 s 内关节漂移应很小（报告最大漂移）。同时报告各关节在这些构型下的静态重力力矩与力矩上限的比值。
- `tests/`：新增 pytest，覆盖两个版本模型的加载与参数（阻尼、力矩上限、kv、timestep、integrator、`neq == 1`）、IK 收敛、成功判定的正例和反例、环境 reset / step 的观测形状。原有的 M0 测试必须仍然通过。

## 交付物

1. 上面列出的代码、模型、场景、配置、脚本、测试文件。
2. 交付记录 `tasks/M1_delivery_r1.md`，必须包含：
   - 改动文件清单（新增 / 修改 / 删除）；
   - 运行过的命令及输出摘要（至少：两个冒烟脚本、IK 统计、`pytest tests -q`、`check_env.py`）；
   - 实测数据：夹爪开口、Can 缩放系数和尺寸、碰撞方案、夹爪 forcerange、TCP 的选择、bin 布置和可达性、重力力矩比、抓取保持期间 Can 的最大相对位移、建议的成功判定阈值；
   - robosuite 的版本（tag 或 commit）以及复制了哪些文件；
   - 未完成或存疑的点。

## 完成标准

- `conda run -n AIControl python -m pytest tests -q` 全部通过（包括 M0 的测试）；`check_env.py` 退出码 0。
- 抓取冒烟测试中，Can 被抬起 ≥ 5 cm，并保持 ≥ 3 s；保持期间 Can 相对 TCP 的位移 < 5 mm。
- 重力保持测试中，各构型 2 s 内关节最大漂移 < 0.01 rad。
- 做不到的项不得绕过或降低标准，在交付记录中如实写明原因并作为失败上报。
