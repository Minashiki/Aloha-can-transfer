# M1 审查记录 r1（Claude）

- 审查对象：Cursor dispatch `ctx_4543034dbdf4`，交付记录 `tasks/M1_delivery_r1.md`
- 基线：`4f840a9`。审查范围是 `git diff` 加上 `git status --untracked-files=all` 列出的新增文件
- 方式：只读。复跑交付里的命令；一次性验证脚本放在协调者的 scratchpad 目录

## 1. 合规检查

| 项 | 结果 | 依据 |
|---|---|---|
| PLAN、WORKFLOW、`.gitignore`、`environment.yml` 没有被改动 | 通过 | 派发前记录了这 4 个文件的哈希，现在 `sha256sum -c` 四份都显示“成功” |
| 没有 commit，没有改作者 | 通过 | HEAD 仍是 `4f840a9` |
| 没有安装 robosuite，运行时不引用 `third_party/` | 通过 | `grep third_party` 只在 README 和注释里出现；robosuite 源码 clone 在 `third_party/robosuite`（已被 gitignore 排除） |
| 只在 AIControl 中运行 | 通过 | 交付记录中的命令全部是 `conda run -n AIControl` |
| 交付记录包含规定的各项内容 | 通过 | 改动清单、命令、实测数据、robosuite 版本、存疑项都有 |

## 2. 复跑结果

- `pytest tests -q`：12 passed（包括 M0 的测试）
- `scripts/m1_grasp_smoke.py`：抬起 8.94 cm，保持 3.00 s，最大滑移 2.40 mm，`GRASP_OK`
- `scripts/m1_ik_reach.py`：两个 bin 的网格都是 49/49，随机 200/200，最大位置误差 0.98 mm
- 补充对照实验（`<scratchpad>/m1_grav0.py`，motor 版，home 构型，2 s）：补偿系数为 0 时，各关节最大漂移 0.03 / 0.44 / 0.54 / 0.03 / 0.18 / 0.03 rad；系数 0.9 时 wrist_angle 漂移 0.036 rad；系数 1.0 时接近 0。说明 frictionloss 本身托不住手臂，**重力保持测试是有效的**。
- 模型参数核对：`aloha.xml` 的 diff 中，阻尼、`actuatorfrcrange`、timestep、integrator 都与 spec 的表格一致；position 版的 kp、kv 与表格一致；motor 版 `gear=1`，ctrlrange 等于力矩上限。

## 3. 问题

### F1【中】TCP 定义不一致：`robot.py` 写的是 `ee_center_site`，配置和交付写的是 `left/gripper`

- 位置：`src/aloha_pp/robot.py:21` `TCP_SITE = "ee_center_site"`，注释写“Selected in M1”；`configs/m1_pick_place.yaml` 是 `tcp_site: left/gripper`。
- 影响：目前没有代码引用 `robot.TCP_SITE`，但它是公开常量。后续里程碑（M2 / M3 / M4）的代码一旦导入它，就会悄悄用错 TCP。
- 建议：TCP 只在一个地方定义，并删除或修正错误的那一个（和 F2 一起处理）。

### F2【中】TCP 不在抓取中心，要靠一个偏移量来补

- 现象：选定的 `left/gripper` 离两指垫中点约 1.6 cm（`tcp_to_grasp_center = [0.015372, 0, 0.004252]`），抓取脚本要用 `罐子中心 − R @ 偏移` 来换算。
- 影响：PLAN 第 3 节要求 TCP 位于两指之间的抓取中心，IK、误差统计、示范数据都以它为准。现在这个偏移约定会扩散到 M4 规划器、M5 示范重定向和 M3 末端误差统计，每个地方都得记着补偿，很容易漏。
- 建议：在 `aloha/aloha.xml` 中新增一个专门的 site（例如 `left/tcp`），放在指垫中点；坐标轴约定写清楚，例如 +X 是接近方向。把它作为唯一的 TCP，删掉 `tcp_to_grasp_center`。验收：手指张开时，这个 site 到两侧指垫中点的距离 < 1 mm。

### F3【中】`fixed_2` 的放置点和 `fixed_1` 在同一个格子里，只差约 1.4 cm

- 现象：`fixed_1` 放置点 (0.336, −0.080)，`fixed_2` 放置点 (0.326, −0.090)，都在同一个目标格内；成功判定的目标格也是全局唯一的。
- 影响：PLAN 要求“至少两组取 / 放位置”，用来比较不同位置下的控制效果。现在两组的放置端几乎一样，比较没有意义。
- 建议：`fixed_2` 放到右 bin 的另一个格子（比如对角的 −X、−Y 格），目标格改成按 mode 配置；`random` 模式继续用 robosuite 的 +X、+Y 格。**这一条需要用户决定**（另一种做法是两组共用一个格子，只让取的位置不同）。

### F4【中】agentview 取景不完整

- 现象：在 `fixed_1` reset 后渲染 256×256 的 agentview（`outputs/m1/review_agent_256.png`），右 bin（放置的那个）只拍到一部分，被画面左下角切掉；home 姿态下夹爪也有一部分在画面上沿以外。
- 影响：策略观测里只有 96×96 的 agentview 和腕部相机。放置区域看不全，会直接影响 M6 / M7 的学习，也影响报告里视频的效果。
- 建议：调整 agentview，让两个 bin（完整）、目标格、Can 以及抓取时的夹爪都在画面内，并给出 96×96 和 256×256 两张截图作为证据。

### F5【中】IK 容差太松，IK 失败时手臂静默停住

- 现象：配置里 `ik.pos_tol = 1e-3`（1 mm）、`rot_tol = 1e-2`（0.57°），网格上最差的位置误差 0.98 mm，正好卡在容差边上。`PositionIkController.apply` 在 `result.success` 为 False 时，直接把当前 q 作为指令，不发出任何提示。
- 影响：M3 要统计毫米级的跟踪误差，IK 残差可达 1 mm，会混进跟踪误差；IK 失败会表现为手臂“卡住”，很难排查。
- 建议：把默认容差收紧到 `pos_tol ≤ 1e-4`、`rot_tol ≤ 1e-3`（DlsIk 类本身的默认值就是这个），重新统计收敛率和耗时；IK 失败时在 `step` 返回的信息里给出计数或标志，脚本中打印出来。

### F6【低】同一参数在多处重复定义

- `robot.py` 里的 `VELOCITY_LIMITS`、`TORQUE_LIMITS`、`POSITION_GAINS`、`DAMPING`，分别与 XML、`configs/m1_pick_place.yaml` 中的值重复。
- 建议：物理参数以 XML 为唯一来源（代码要用时从 `MjModel` 读取，测试也对 XML 做断言）；速度上限只放在配置里。

### F7【低】进程退出时 EGL 报错

- 复现：`AlohaCanEnv()` 渲染后不调用 `close()` 就退出，会打印 `EGLError ... Renderer.__del__`。
- 建议：环境实现上下文管理器（`__enter__` / `__exit__`），脚本中显式调用 `close()`。

### F8【低】成功判定没有要求“已经松手”

- 现象：成功判定只看 Can 是否在格内、是否静止，没有检查夹爪是否已经松开。夹着 Can 停在格子里也会被判为成功。
- 建议：增加一个条件，例如两个手指与 Can 都没有接触，或者夹爪开度大于某个阈值，并加入单元测试。

### F9【信息】Can 质量 11.5 g

`density=100` 与 robosuite 的 `can.xml` 一致（已核对 `third_party/robosuite/.../can.xml:10`），可以保留。报告中要写明质量；如果 M4 搬运过于轻松，可以另做一组更重的对照。

### F10【信息】夹爪余量：spec 自身有矛盾

spec 要求“直径 4.5–5 cm，每侧约 1 cm 余量”，但实测最大开口只有 5.74 cm，两条做不到同时满足。这是 spec 的问题，不算交付缺陷。建议接受现在的结果：直径 4.50 cm，每侧余量 0.62 cm（IK 误差 < 1 mm，余量足够）。**需要用户确认。**

### F11【信息】ctrl = 0.002 在动力学中到不了

指尖碰撞球在 q ≈ 0.008 时就已经接触；闭合指令实际用的是 0.015，不影响使用。不需要处理。

## 4. Cursor 对话状态

- 上下文用量：状态栏显示 **90.8%**（`Grok 4.7 256K High · 90.8%`），已经超过 65%。**审批后的修改必须开新对话，并附交接单。**

## 5. 结论

M1 主体合格：模型参数、场景、IK、环境、两个冒烟测试都达到了验收门槛。需要修改的有 F1–F5（中）、F6–F8（低）。F3、F10 需要用户决定。
