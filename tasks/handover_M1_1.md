# 交接单 M1-1（新 Cursor 对话）

> 原因：M0 那个 Cursor 对话（dispatch `ctx_8e8bea6a3b0e` / `ctx_147474493028`）的上下文用量已经到 58.5%，并且已完成 2 次任务。M1 工作量大，按保守规则换一个新对话。

## 1. 现在要做什么，最后要交什么

实现里程碑 M1“模型与场景”，任务说明见 `tasks/M1_dispatch.md`。最后要交的是该文件“交付物”一节列出的代码、模型 / 场景文件和交付记录 `tasks/M1_delivery_r1.md`。

## 2. 最新的要求，以及哪些旧要求已经作废

- 以 `tasks/M1_dispatch.md` 为准；背景读 `docs/PLAN.md` 第 2–4 节和第 10 节 M1，**只读，不得修改** `docs/PLAN.md`、`docs/WORKFLOW.md`。
- M0 修 F1 时临时允许修改 `.gitignore`，那次授权**已经作废**；M1 不得修改 `.gitignore`。
- 已作废的旧设想：Lift 作为独立任务（现在只在 Can 场景里做抓起冒烟测试）；保留 Menagerie 原有阻尼（现在改用低阻尼）；motor 执行器的 gear=150/200（现在统一用 `gear=1`）。

## 3. 已确认的事项和依据

- 环境：conda `AIControl`，torch 2.15.0+cu132、mujoco 3.17.0；`scripts/check_env.py` 和 `pytest tests` 全部通过。依据：`tasks/M0_review_r3.md`。
- 包结构：`src/aloha_pp/{sim,kin,dyn,control,planning,data,policy,rl,utils}`，已用 `pip install -e .` 安装。依据：提交 `4f840a9`。
- `aloha/aloha.xml` 中重复的夹爪 equality 已删，`neq == 1`。依据：提交 `3d135e0`，以及 `check_env.py` 的输出。
- 当前的 `aloha/joint_position_actuators.xml` 虽然名字叫 position，内容实际上是 motor 执行器（gear=150/200/…，没有 ctrlrange），position 版的写法都被注释掉了。依据：直接读该文件。
- `aloha.xml` 里没有地面、桌子、`agentview` 相机；光照只有一盏。M0 渲染的腕部相机图像很暗（像素最大值 45）。依据：`tasks/M0_delivery_r1.md`。
- 末端参考点（TCP）有两个候选 site：`ee_center_site`（在 `left/gripper_base` 下，pos 0.13 0 0）和 `left/gripper`（在 `left/gripper_link` 下，pos 0.13 0 -0.003）。依据：`aloha/aloha.xml`。

## 4. 还没验证的推测

- 推测：`aloha.xml` 的 `<compiler meshdir="assets">` 是相对于**主模型文件**所在目录解析的。如果顶层场景文件放在别的目录，再 include 机器人，mesh 路径可能会出错。实现时需要实测。
- 推测：robosuite 的 bins arena 自带 `agentview` 等相机定义，Can 是 mesh 物体，可以直接拿来用。需要以实际下载到的 robosuite 源码为准。
- 推测：ALOHA 左臂的 base 放在桌面高度（z = 0），bin 放在机械臂前方 0.25–0.45 m 的范围内，夹爪从上往下抓应该够得到。需要做可达性实测。

## 5. 试过不行的路和原因

- Codex 在 M0 复核时，用 urllib 访问 pytorch.org 返回 HTTP 403。用 Python 抓网页可能被拦截；用 pip / git 下载软件包是可以的（M0 时 pip 安装正常）。

## 冲突

无。

## 6. 下一步最小的一件事，以及做到什么样算合格

先完成 `tasks/M1_dispatch.md` 第 1 步“执行器文件与模型修改”。合格标准：两个顶层模型（motor 版、position 版）都能加载，`neq == 1`，6 个臂关节的 `actuatorfrcrange` 和阻尼都等于规定值，`model.opt.integrator` 为 implicitfast，`model.opt.timestep == 0.002`。
