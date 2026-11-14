# M5 派发：示范数据（robomimic Can-PH 重定向 + 规划器保底）

**先读这份交接单：`tasks/handover_M5_1.md`。**

项目根目录：`/home/minashiki/AIControl/FinalHomework`。背景见 `docs/PLAN.md` 第 7、8 节和第 10 节中的 M5（只读）。M4 的接口与结论见 `tasks/M4_delivery_r2.md`、`tasks/M4_review_r3.md`。

## 硬性约束（违反任何一条即判为不合格）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`）。禁止系统 Python、base 环境，禁止新建环境，**不得安装任何包**（h5py、imageio 已有）。交付记录中的命令要写出环境前缀。
2. 不得修改：`docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、M2 计算模块（`kin/fk.py`、`kin/spatial.py`、`dyn/rnea.py`）、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`、`configs/m4_task.yaml`。
   - 可以扩展 `sim/env.py`、`planning/`、`data/`，新增 `configs/m5_*.yaml`、`scripts/m5_*.py`、`tests/test_m5_*.py`。
   - **M1–M4 已有行为必须保持不变**：默认参数下 `AlohaCanEnv` 的 reset / step、M4 规划器和参考接口的结果与 `fe540b4` 一致（见第 6 节回归）。
3. 不得修改增益、力矩上限、速度上限、成功判定阈值、`random_can` 采样范围、`gripper_open` / `gripper_close`。认为必须改时，停下来在交付记录中写明理由和数据，作为存疑项上报。
4. 不得把 robomimic / robosuite 安装为依赖包，运行时不得 import 它们，也不得引用 `third_party/`（`third_party/robosuite` 的源码可以**阅读**，用来查坐标、目标格等定义，查到的数值写进配置并注明出处）。
5. 不得执行 `git commit` / `push` / `config` / `add`，不得添加 co-author。不得启动其他 agent。
6. 不得覆盖已有的 `outputs/m3/`、`outputs/m4/`。M5 的所有输出写到 `outputs/m5/`；回归用的重跑写到 `outputs/m5/regression/` 下。
7. 单个脚本几分钟内完成。完整的数据生成总时长控制在 20 分钟以内；超过的话，改为后台运行并写日志，在交付记录中写明实际耗时。

## 要做的事

### 1. 获取原始数据

- 从 robomimic 官方渠道下载 **Can-PH 的 low_dim hdf5**（200 条人类示范，20 Hz）。先查官方文档确定下载地址和版本（v1.5 / v1.4.1 等），用 `curl` / `wget` 下载；不要用 pip 装 robomimic 来下载。
- 存放在 `datasets/robomimic/can/ph/`（`*.hdf5` 已被 `.gitignore` 排除）。交付记录写明 URL、版本、文件大小、sha256、数据集许可证（以官方页面为准）。
- 下载失败时停下来报告，不要改用其他来源或其他任务的数据。

### 2. 重定向（`src/aloha_pp/data/`，PLAN 第 7 节）

逐条处理 200 条示范：

1. **读取**：用 h5py 读 `robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`（或动作里的夹爪信号）、Can 位姿（`object` 观测的前 7 维，或由 `states` 得到，写明取法）。不读原始 OSC 增量动作。
2. **坐标映射**：robosuite PickPlaceCan 的世界系 / bin 布局与本项目 base 系不同（本项目 bin 线性尺寸是 robosuite 的 0.36 倍，左 bin 是起始 bin，右 bin 是目标 bin，见 `tasks/M1_delivery_r1.md`）。从 `third_party/robosuite` 源码查清 robosuite 的起始 bin、目标 bin 和 Can 目标格的位置，确定一个**保持手性的**变换（旋转 + 平移；不得用镜像），使 robosuite 的 Can 目标格对应本项目的 `fixed_1` / `random` 目标格（右 bin +X+Y 格）。如果找不到这样的变换，停下来报告。
3. **以物体为中心重定向**：按夹爪信号分段。抓取前的 TCP 轨迹用 Can 坐标系表示，放置后用目标格坐标系表示，搬运段在两者之间平滑过渡（方法自定，写明）。距离不按 0.36 缩放（Can 只缩了 0.9），只是起点和终点换到本项目的 Can 和目标格上；搬运高度要高过 bin 壁。
4. **Can 初始位置**：每条示范在本项目中的 Can 初始位置由原示范 Can 相对起始 bin 中心的位置映射得到（方法自定，例如按 bin 内尺寸等比例缩放），必须落在 `random_can` 范围内。
5. **工具坐标系对齐**：Panda 的 `robot0_eef_pos` / `robot0_eef_quat` 与本项目 `left/tcp`（两指垫中点，+X 为接近方向）的对应关系写明。姿态取“从上往下 + 绕竖直轴的 yaw”；yaw 怎么从原示范得到（保留、按平行夹爪的 π 对称折叠、或改用 M4 的 `choose_yaw`）由你决定并写明理由。
6. **动作重标注**：动作是 base 系绝对 TCP 位姿 `[xyz, quat(wxyz), gripper]`。按 PLAN“以实际到达的末端位姿作为绝对目标”：第 t 拍的动作取第 t+1 拍重定向后的 TCP 位姿（与 M4 规划器“命令提前一拍”的时序一致，写明）。
7. **夹爪映射**：原示范的开 / 合 → `gripper_open` 0.037 / `gripper_close` 0.015。开合过渡、以及开合前是否插入“参考滞后 + 稳定时间”的等待（M4 规划器的做法），由你决定并写明；参数写进 `configs/m5_*.yaml`。
8. **时间缩放**：按 ALOHA 的关节速度上限重采样。M4 决定“M5 示范默认 medium 档”，因此每条示范选一个放慢系数，使 IK 得到的关节参考峰值速度不超过速度上限的 **0.5 倍**（与 medium 档的峰值比例一致）。系数的选法写明，交付记录给出系数分布。
9. **闭环执行**：在 ALOHA 上用 C3 + preview 参考（`env.step(action, next_action=...)`，最后一个动作重复自身）执行重定向后的动作，判定成功（M1 成功判定，含“已松手”）。失败用 M4 的失败分类，另加重定向自己的失败原因（例如 IK 不可达、坐标映射后 Can 落在 bin 外）。

### 3. 规划器加噪声（保底）

- 在 M4 规划器基础上实现加噪声的示范生成：噪声加在**路点和时间律**上（例如抓取点 / 放置点的小偏移、经过点、yaw、各段时长在 medium 附近的缩放），而不是逐拍的动作抖动。噪声幅度写进配置并写明理由。默认参数（噪声为 0）时与 M4 规划器逐拍一致。
- Can 初始位置用 `random` 模式的采样。
- **使用规则**：重定向的成功示范 ≥ 50 条时，数据集只用重定向示范，规划器噪声示范另存一个文件备用；不足 50 条时，用规划器噪声示范补足到 50 条，每条示范标注来源。

### 4. 数据集（`outputs/m5/dataset/`）

统一 hdf5 格式（结构自定，写成文档放在 `src/aloha_pp/data/` 的模块说明或单独的 `README`），至少包含：

- 每条示范：
  - 观测，每拍一个（动作执行前）：`agentview`、`wrist_cam_left`（96×96×3 uint8），`tcp_pose`（7）、`gripper_opening`（1）、`q`、`qvel`；
  - 特权信息：Can 位姿（供 M7 的非对称 critic 使用）；
  - 动作（T×8）；
  - 属性：来源（`retarget` / `planner_noise`）、原示范编号或种子、放慢系数、初始完整状态（qpos / qvel，用于重放）、成功标志与到达成功的拍数。
- 数据集级属性：控制频率、相机名与分辨率、动作约定（base 系、wxyz、绝对位姿、提前一拍）、参考模式（preview）、生成代码的 git commit、配置快照。
- 示范在成功判定通过的那一拍结束。只存成功的示范；失败的只在统计里出现。
- 提供读取接口（供 M6 使用）和一个校验脚本：检查字段形状 / 类型、图像非空、动作四元数单位长度、长度统计。

### 5. 验证

- **重放一致性**：从数据集里抽 5 条（至少含 1 条规划器噪声示范，如果有），用存储的初始状态和动作重放（同样的 `next_action` 规则），必须再次成功，且关节轨迹与存储的观测一致（给出最大偏差）。
- **视频**：抽样回放视频至少 5 条（agentview，分辨率高于 96×96，单条 ≤ 2 MB），放在 `outputs/m5/videos/`；另存几张 96×96 双相机观测的拼图，确认训练用图像正常。
- **统计**（`scripts/m5_analyze.py`，只读日志 / 数据集）：重定向成功率与失败原因分布；放慢系数、示范长度、峰值关节速度 / 速度上限的分布；Can 初始位置散点（按成功 / 失败着色）；最终放置误差分布；数据集最终组成（来源、条数）。

### 6. 测试与回归

- 坐标映射：往返变换、手性（行列式为 +1）、robosuite 目标格映射到本项目 `fixed_1` 格中心附近（容差写明）。
- 重定向：构造一条合成轨迹，分段、坐标系表示、动作提前一拍、夹爪映射、时间缩放后峰值速度不超过 0.5·v_max。
- 规划器噪声：噪声为 0 时与 M4 规划器逐拍一致；固定种子可复现。
- 数据集：写入 / 读取往返、字段齐全、校验脚本能发现构造的错误（例如四元数非单位、图像全零）。
- env 扩展（如指定 Can 初始位姿 / 完整状态的 reset）：默认参数下与原行为一致。
- **回归**：
  - `pytest tests -q` 全部通过（M0–M4 原有测试不得删改）；`check_env.py` 退出码 0；`m1_grasp_smoke.py` 输出 `GRASP_OK`；
  - 重跑 M4 的 E1（6 组），用 `m4_run.py --e1 --out outputs/m5/regression/m4_e1`，与 `outputs/m4/summary.csv` 中 E1 行的共有指标逐项一致（差 ≤ 1e-9 或说明原因）。不带 `--out` 运行 `m4_run.py` 会写 `outputs/m4/`，禁止。

## 交付物

1. 上述代码、配置、脚本、测试；`outputs/m5/` 下的数据集、日志、统计表、图、视频（outputs 和 datasets 不进 git，交付记录中列出文件名和大小）。
2. `tasks/M5_delivery_r1.md`，包含：
   - 改动文件清单；
   - 运行过的命令及输出摘要（含下载、生成各阶段耗时与总耗时、`m5_analyze.py`、校验脚本、`pytest tests -q`、`check_env.py`、`m1_grasp_smoke.py`、M4 E1 回归）；
   - 原始数据的来源信息（URL、版本、大小、sha256、许可证）；
   - 坐标映射的推导（robosuite 源码中的出处、变换矩阵、目标格对应关系）；
   - 重定向方法：分段、坐标系表示、搬运过渡、Can 初始位置映射、工具系对齐、yaw 处理、动作提前一拍、夹爪映射与等待、时间缩放；
   - 规划器噪声的设计与幅度；
   - 重定向成功率与失败原因分布、数据集组成；
   - 数据集格式说明；
   - 重放一致性结果；
   - 未完成或存疑的点。

## 完成标准

- 数据集中 **≥ 50 条成功示范**，每条标注来源；重定向的成功率和失败原因必须报告（成功率没有硬性门槛）。
- 重放一致性：抽样 5 条全部再次成功。
- 视频和观测拼图满足第 5 节要求。
- 测试与回归全部通过。
- 做不到的项不得绕过或降低标准；如实写明，并作为失败上报。
