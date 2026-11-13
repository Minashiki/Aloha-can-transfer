# M1 复核 r1（Codex）

日期：2026-11-13；基线 `4f840a9`。本轮仅写本报告，未安装包、创建环境、修改实现或执行 git 写操作。一次性验证脚本与图片、视频都放在 `/tmp/m1-recheck-codex/`。复核以 dispatch 为验收依据；不把后续里程碑的潜在需求自动升级为 M1 缺陷，也不将有限测试当成历史行为的完整证明。

## 一、逐条回应

### 1. 合规检查

| 编号 | Claude 检查项 | 判断 | 依据 |
|---|---|---|---|
| C1 | PLAN、WORKFLOW、.gitignore、environment.yml 未改 | 部分同意 | `git diff 4f840a9 -- .gitignore environment.yml scripts/check_env.py` 无输出。两份 docs 为忽略文件，本次没有派发前哈希；无法独立重证 Claude 的历史哈希结果，没有证据认定交付者改过。 |
| C2 | 没有 commit、没有改作者 | 部分同意 | `git log -1 --oneline` 是 `4f840a9 M0：conda 环境配置与 aloha_pp 仓库骨架`。支持 HEAD 未新增提交，但 HEAD 不足以证明作者配置从未变化；本轮按约束未读写 git config。 |
| C3 | 未安装 robosuite、运行时不依赖 third_party | 同意 | AIControl 的 importlib.metadata 扫描 robosuite/robomimic 分发名交集为 `[]`；运行源码的 third_party 无实际路径依赖。`git -C third_party/robosuite log -1 --format='%H'` 为 `a071383d53568ab798eb315c0e95357911be922d`；两份 LICENSE 的 SHA256 都是 `177978cbece0a4c454c2aaec5b3f145b39270814874c43109da9e829c39d9cba`。 |
| C4 | 只在 AIControl 运行 | 部分同意 | 交付记录在总说明中声明全部使用该环境，但命令展示省略了前缀。本次所有 Python 验证均用 AIControl，支持当前可复现性；仅靠交付自述不能证明所有历史运行行为。 |
| C5 | 交付记录包含规定各项内容 | 同意 | 文件清单、四类运行摘要、实测数据、版本与存疑项均存在。其“全部完成”的解释仍受 F10 与下列问题约束。 |

### 2. 复跑结果（对应原文五条）

**R1，pytest：同意。**

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -m pytest tests -q -p no:cacheprovider
```

退出码 0，`12 passed in 1.09s`，含 M0。原 `check_env.py` 的函数也复跑通过，只在内存把 `IMAGE_PATH` 改到 `/tmp/m1-recheck-codex/m0.png`，未改检查逻辑：

```text
python: 3.12.14
torch: 2.15.0+cu132
torch.version.cuda: 13.2
mujoco: 3.17.0
numpy: 2.5.4
[PASS] cuda_available: True
[PASS] sm_120: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
[PASS] gpu_matmul: shape=2048x2048 max_abs_err=2.289e-04 rtol=0.01 atol=0.01
[PASS] mujoco_load_step: neq=1 steps=20
[PASS] wrist_cam: shape=(96, 96, 3) dtype=uint8 min=0 max=45 saved=/tmp/m1-recheck-codex/m0.png
ALL CHECKS PASSED
```

**R2，抓取冒烟：部分同意。** 原脚本的成功结果与数值可复现；但“相对 TCP 位移”的测量定义与采样仍有 N4 的边界，不能等同于严格验证整个保持时段的局部坐标滑移。

调用方式是在 AIControl 的一次性脚本中：

```python
ns = runpy.run_path('scripts/m1_grasp_smoke.py')
ns['main'].__globals__['repo_root'] = lambda: Path('/tmp/m1-recheck-codex')
ns['main']()
```

仅替换抓取脚本自己的输出目录 helper，环境加载模型的 repo_root 未改变。输出退出码 0：

```text
video /tmp/m1-recheck-codex/outputs/m1/grasp.mp4 bytes 48440 mb 0.048
agentview min 16 max 255 wrist min 11 max 255
can_z0 0.0360 hold_min_z 0.1254 hold_lift 0.0894 peak_lift 0.0963
hold_steps 60 hold_s 3.00
max_slip_m 0.002402
GRASP_OK
```

**R3，IK 统计：同意数值。**

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B scripts/m1_ik_reach.py
```

```text
grid left_pick: 49/49 worst_pos=9.8387e-04 m worst_rot=1.5623e-03 rad
grid right_place: 49/49 worst_pos=9.8387e-04 m worst_rot=1.5623e-03 rad
random 200: success 200/200 pos mean 4.2484e-04 max 7.9376e-04 rot mean 7.1930e-04 max 1.2840e-03
```

但这只是所列目标的运动学收敛，未自动证明所有碰撞与所有 yaw 的抓取都可行。原网格并非 README 所述完整地板边界；Can 中心可用区域应扣掉墙厚和半径，不能直接拿地板面积断言覆盖不足。本轮另按 `0.072/0.09 - 0.0036 - 0.02251` 的外边界内缩半范围检查每 bin 9×9 个中心（该矩形尚未扣除右 bin 隔板禁区），162/162 收敛；解处张开夹爪与 bin 未发现深于 1 mm 的接触穿透。这仍不是全路径碰撞证明。

**R4，重力保持与对照：部分同意。** 已独立复跑补偿系数 1 的三个构型；Claude 的 0/0.9 系数对照本次未复跑，不能冒称独立确认全部对照数据。

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B scripts/m1_gravity_hold.py
```

退出码 0；home 的 shoulder/elbow 比为 0.1897/0.2162，左右构型为 0.2430/0.2115，`max_drift_rad 1.758456e-04 limit 0.01`。脚本实际每物理步更新 qfrc_bias；三构型、2 s 和每关节最大漂移统计符合任务，原结果有实质价值。

**R5，模型参数：同意。** `git diff 4f840a9 -- aloha/aloha.xml` 仅改要求的阻尼、力矩范围、option 与执行器 include（另删旧注释）；关节范围、armature、frictionloss 未改变。`aloha/actuators_motor.xml:3–9` 的 gear/ctrlrange 与表一致；`actuators_position.xml:3–9` 的 kp/kv 正确，夹爪 ±20 N、kp 2000、ctrlrange 正确；`tests/test_m1_model.py` 在加载后检查了这些核心值与 gravcomp 开关。纯机器人模型不含执行器仍能通过 M0，两个顶层模型各 include 一份执行器和场景。

### 3. F1–F11

**F1【中】：同意。** `src/aloha_pp/robot.py:19–21` 定义 ee_center_site，而 `configs/m1_pick_place.yaml:6` 与环境用 left/gripper。`rg -n TCP_SITE src scripts tests` 没发现运行端引用该常量，所以当前冒烟不受影响；公开常量和注释的错误是后续明确的陷阱，修正即可，无需大范围抽象。

**F2【中】：部分同意。** 指垫中心偏移确实在 YAML:7–9 和 `scripts/m1_grasp_smoke.py:28–31,43–46` 体现，TCP 统一性问题成立。不过 Claude 将“TCP 位于抓取中心”归到 PLAN 第 3 节不准确：该表只要求统一 site，具体中心要求来自 dispatch 第 4 步。dispatch 又明确在两个既有 site 中选择，因此“新增第三个 site”是改变约定的解决方案，不能当作唯一无争议修法。可统一修正所选 site 的位置或新增专用 site，并同步消费者；需记录具体位置与坐标轴，验证张开时到指垫中点 <1 mm，同时复跑抓取。偏移补偿已使当前抓取通过，不宜把它说成当前完全不能工作。

**F3【中】：反驳其缺陷定性。** `configs/m1_pick_place.yaml:52–60` 两组取位置不同、放位置也不同，相差约 1.4 cm 是事实；规格没有要求放到不同格，也没有规定位置间距。“比较没有意义”缺乏指标或实验支持，不能据此判 M1 不合格或强制新增每 mode 的目标格。若用户需要不同格对照可另行调整，但这属于实验设计扩展，不是必修缺陷。当前 place_pos 只是配置数据，不参与 reset 的目标格切换；这与配置注释明确的共用一个目标格一致。

**F4【中】：同意。** `assets/scene/pick_place_can.xml:97` 附近的 agentview 相机取景未满足 dispatch 第 2 步完整覆盖要求。本次独立用相机位姿与 fovy 投影两个 bin 外沿顶角到归一化画面，边界为 [-1,1]：

```text
bin_left  min [-0.22959719 -0.15989284] max [0.59134374 0.264173]
bin_right min [-1.10179495 -0.98889263] max [0.17386419 -0.0938843]
```

右 bin 的 x 超界约 0.10，确实被裁切；这项与分辨率无关。应修相机并验证两个 bin、机械臂及相关抓取阶段，而非只检查图片 min/max 不同。相机精确行号见 `rg -n 'camera name="agentview"' assets/scene/pick_place_can.xml`。

**F5【中】：部分同意。** `src/aloha_pp/sim/controller.py:68–73` 求解失败时回退当前 q，环境 `step` 没有通过返回值报告失败，诊断不足成立；但 `controller.last_result` 已保存结果，不能称完全没有任何可查询线索。配置的 1 mm / 0.01 rad 容差没有违反 M1 给定阈值，原 spec 只要求参数化与统计；将 M3 的误差要求直接视为当前缺陷不充分。建议明确暴露求解状态，再按跟踪精度预算决定容差，而非不经统计一律收紧。

额外核查姿态实现：`kin/ik.py:173–187` 把 mju_subQuat 的局部旋转向量乘当前 site 旋转矩阵，转换到与 mj_jacSite 一致的世界坐标；在 home 的 wrist_angle 上用 1e-6 rad 扰动，旋转差商与世界旋转 Jacobian 列差范数 `8.22e-11`，未发现原理性坐标系错误。`_target_in_world` 与 `_site_pose_base` 的位置/四元数乘法方向也相互一致。line search 使用未加权 cost、DLS 使用加权误差是算法选择，尚无失败证据，本轮不新增缺陷。

**F6【低】：部分同意。** robot.py 常量与 XML/YAML 有重复，未使用的速度常量可消除，以免漂移。但测试的“期望值”不能从同一 XML 的“实际值”读取后再与自己比较；`tests/test_m1_model.py:39–98` 的独立期望值有价值。建议只减少运行时无意义重复，并保留可对照规格的测试基准；不赞成按“唯一来源”机械删除所有测试期望值。

**F7【低】：部分同意。** 下述复现确实在进程结束打印 Renderer/GLContext.__del__ 的 EGLError，退出码仍为 0：

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -c 'from aloha_pp.sim.env import AlohaCanEnv; e=AlohaCanEnv(); e.reset(seed=0)'
```

但这是调用者未释放资源；环境已有 close，`scripts/m1_grasp_smoke.py:112–113` 和环境测试已有 finally/close。上下文管理器是便利改进，不能说当前脚本均缺 close，也不应据此阻塞 M1。

**F8【低】：反驳其当前缺陷定性。** `src/aloha_pp/sim/success.py:35–40` 的确未判断松手；夹住且静止在目标格会成功。但 PLAN 第 4 节和 dispatch 第 5 步明确列出的成功定义只有中心位置、线速度和连续 N 步，代码符合这个定义。增加无接触/开口阈值会改变评测指标，应作为后续用户选择而非要求交付者自行扩展。现有正反例测试覆盖了规定谓词与中断计数，没有违反现行成功判定规格。

**F9【信息】：同意。** 已加载模型的 `body_mass[can] = 0.011474021006973003 kg`，与密度 100 和圆柱尺寸一致；交付已披露质量。是否加入更重物体是后续对照选择，当前不要求更改。

**F10【信息】：部分同意。** 开口/直径/余量的算术成立：5.74 cm 开口装 4.50 cm 罐，每侧约 0.62 cm，无法同时达到每侧 1 cm。交付已主动记录，不能把它归因于偷改模型。合理接受哪个指标由协调者确认；在未接受偏差前不能同时声称所有原规格均已达到。“IK 误差小所以余量足够”也不等于已证明动态路径、感知误差和不同姿态全部安全。

**F11【信息】：同意。** joint 软件目标 0.002 低于动力学自碰停止位置并不表示控制失效；当前实际闭合目标 0.015，在 forcerange 内抓取已通过。指尖球几何未在本轮改动，不要求为达到 q=0.002 改旧模型。夹爪开口/缩放的细粒度几何测量采用交付结果，本轮未重新完成该整套测量。

### 4. Cursor 对话状态：部分同意

本次未访问 Cursor GUI，不能独立重证 90.8% 或对话用量；这不影响代码复核结论。新对话/交接要求属于协调工作流，本轮未启动任何 agent。

### 5. 结论：部分同意

同意模型核心参数、当前 IK 样本、两个数值冒烟结果和 12 个测试通过；同意 F1、F4 是明确应修问题，F2/F5 中的统一约定和诊断问题值得解决。反对一概把 F1–F8 都列为 M1 必修：F3、F8 缺乏现行规格依据，F6/F7 应控制改动范围。新增问题见下；当前只能称“主体功能可运行，仍有契约与验收缺口”，不能称全部规格无条件完成。

## 二、新增问题

### N1【中】step 返回的 TCP/物体派生位姿与 qpos 不在同一时刻

- 位置：`src/aloha_pp/sim/env.py:145–158,160–175,186–192`。
- 现象：mj_step 积分更新 qpos/qvel 后，派生 xpos/site_xpos 尚对应积分前状态；代码立即拿这些派生量做 success、TCP 观测与特权位姿。一次 moving action 后，取 obs 的 TCP，再对同一 qpos 执行 mj_forward，TCP 差 `0.000162761926 m`；另连续 15 个小动作最大差 `0.000102321676 m`。qpos 未变化，仅刷新派生量就改变返回位姿，证实一物理步滞后。
- 影响：策略看到的 q 与 TCP 不一致；成功计数位置与速度不同步；后续毫米级跟踪误差和物体速度判据受到偏差。
- 建议：积分后在需要读取派生量时同步必要的运动学字段，明确 success 在哪个时刻评估，避免把旧位置与新速度混合。reset 50 步后返回观测也需同步。不要为了修观测改变物理步长或控制步数。
- 验收：移动动作返回的 TCP 应与从同一返回 q 做独立 FK 一致，物体特权位姿与当前 freejoint 状态一致；加一个动态场景的环境测试，不能只断言 shape。

### N2【低】mesh 顶点 AABB 中心被误当成 body 局部坐标

- 位置：`src/aloha_pp/sim/env.py:100,186–189,238–243`。
- 现象：MuJoCo 编译后的 mesh_vert 位于网格参考系，代码忽略编译生成的 mesh/geom 平移旋转，直接当作 body local。实测 `mesh_pos=[5.301e-05,-1.203e-04,3.226e-04]`，mesh_quat 也非单位四元数。用 `geom_xmat/geom_xpos` 将所有编译后顶点转到世界再计算中心，与当前 can_center_base 差 `[-5.1392e-05,1.2059e-04,-3.2239e-04] m`，约 0.35 mm。
- 影响：特权中心、成功判定和抓取高度有固定小偏差；当前余量大所以冒烟未失败，不能由此认为坐标计算正确。配置 can_center_local 恰好与当前函数值接近，不是正确坐标转换的证据。
- 建议：通过实际 visual geom 的局部/世界变换求中心并转换到 body/base，或明确使用实际碰撞圆柱中心作为物体几何中心；不要把 mesh_vert 中心直接接到 body pose。
- 验收：以变换后的 visual 顶点中心或约定的 cylinder 中心为独立真值，在至少两个物体朝向对比特权位姿，误差处于浮点容差内。

### N3【中】IK 可达性脚本即使完全失败也返回成功

- 位置：`scripts/m1_ik_reach.py:37–78`。
- 现象：只打印计数，不检查统计是否满足验收。本次仅在 /tmp 探针里把 DlsIk.solve 的返回值用 `dataclasses.replace(..., success=False)` 替换，保留原求解误差；原 main 输出 `grid 0/49`、`random success 0/200` 后仍正常返回，无 SystemExit。原脚本默认进程状态因而仍会为 0。
- 影响：现有 pytest 对随机 200 点有断言，不能说完全没有门禁；但仅网格边缘失败不会触发对应测试或脚本退出失败，容易把“命令成功”误当成全区域验收成功。
- 建议：明确脚本所要求的收敛条件，对不满足条件返回非零，并从场景尺寸导出/核对采样边界；保留原始统计，勿隐藏失败点。
- 验收：注入一个网格点失败后进程必须非零，正常运行仍输出统计并为零；测试不得只检查输出文本存在。

### N4【低】抓取“相对 TCP 位移”没有消除 TCP 转动，且只在动作边界采样

- 位置：`scripts/m1_grasp_smoke.py:73–78,91–108`。
- 现象：使用 base 系 `can - tcp` 的变化作为 slip，没有乘 TCP 旋转逆矩阵。刚性夹持时若手腕改变姿态，该向量也会变化，可能误报滑落；相反某些组合运动会掩盖局部滑移。每 25 个物理步只采一次，无法约束中间瞬态。
- 影响：当前固定朝向冒烟的 2.40 mm 是该度量的真实结果，但不足以严格声称整个保持区间的 TCP 局部相对位移最大值小于 5 mm。没有证据表明当前运行实际滑落，本条是验收测量缺口。
- 建议：用 `R_tcp.T @ (p_can-p_tcp)` 记录相对位置，并覆盖保持开始及物理步采样或明确规定并说明采样率；与 N1 的状态同步一并处理。
- 验收：合成刚性随 TCP 平移/旋转的物体，局部 slip 应为零；注入一次超过 5 mm 的局部位移必须检出；正常抓取仍通过。

## 三、总结

计数口径：5 条合规 + 5 条复跑 + F1–F11 + 1 条 Cursor 状态 + 1 条结论，共 23 项：**同意 9、反驳 2、部分同意 12；新增 4（中 2、低 2）**。

- 同意：C3/C5，R1/R3/R5，F1/F4/F9/F11。
- 反驳：F3、F8 的当前缺陷定性；不否认所描述的实现事实。
- 部分同意：C1/C2/C4，R2/R4，F2/F5/F6/F7/F10，Cursor 状态及总评。
- 新增：N1 状态时刻不同步、N2 mesh 中心坐标错误、N3 IK 脚本不传播失败、N4 滑移测量定义与采样边界。

主要验证命令除上文所列外，探针均以 `PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B /tmp/m1-recheck-codex/probe.py` 和同目录 `additional.py` 运行。probe 包含模型中心与相机投影、微扰 Jacobian、162 点可达性、重定向后的原抓取/环境自检；additional 包含分发名扫描、失败注入、reset 重复性和动态观测同步验证。重复 `reset(seed=0)` 的 qpos 与特权位姿最大差均为 0。所有环境正常路径均显式 close；仅 F7 的最小复现故意不 close。项目内唯一新增写入为本报告。
