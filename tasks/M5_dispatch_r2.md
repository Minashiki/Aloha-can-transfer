# M5 派发 r2：按审批结论修改并重新生成

**先读这份交接单：`tasks/handover_M5_2.md`。**

项目根目录：`/home/minashiki/AIControl/FinalHomework`。

## 任务

按 `tasks/M5_review_r2.md` 的“最终结论（交用户审批）”第 1–8 项修改（已经用户审批），然后整体重新生成数据集并重跑全部验证。背景与依据见 `tasks/M5_review_r1.md`、`tasks/M5_recheck_r1.md`；原始任务说明见 `tasks/M5_dispatch.md`（其中的硬性约束和完成标准继续有效）。

## 硬性约束（违反任何一条即判为不合格）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`），不得安装任何包，命令写出环境前缀。
2. 不得修改：`docs/`、`.gitignore`、`environment.yml`、`scripts/check_env.py`、`aloha/`、`assets/scene/`、M2 计算模块、`configs/m1_pick_place.yaml`、`configs/m3_track.yaml`、`configs/m4_task.yaml`。M1–M4 默认行为保持不变。
3. 不得修改增益、力矩上限、速度上限、成功阈值、`random_can`、`gripper_open` / `gripper_close`。
4. 不 import robosuite / robomimic，`third_party/` 只读。
5. 不得写 `outputs/m3/`、`outputs/m4/`；M5 输出写 `outputs/m5/`（可覆盖 r1 的结果），回归写 `outputs/m5/regression/`。
6. 不得执行 `git commit` / `push` / `config` / `add`，不得添加 co-author，不得启动其他 agent。
7. 数据生成总时长控制在 20 分钟以内；超过的话改为后台运行并写日志，交付中写明实际耗时。

## 交付物

1. 修改后的代码、配置、测试；重新生成的 `outputs/m5/dataset/demos.hdf5`、`planner_noise.hdf5` 与统计、图、视频。
2. `tasks/M5_delivery_r2.md`：改动文件清单；运行过的命令及输出摘要（生成各阶段耗时、`m5_validate`、`m5_analyze`、重放、`pytest tests -q`、`check_env.py`、`m1_grasp_smoke.py`、M4 E1 回归）；第 1–8 项逐项处理结果（含数值）；数据集新组成；未完成或存疑的点。

## 完成标准

- 第 1–8 项全部处理；参考速度峰值严格 ≤ 0.5·v_max、速度超限标记全 0；生成执行与回放执行逐位一致（float64 状态最大差 = 0）。
- 数据集 ≥ 50 条成功示范，每条有来源和逐拍段标签。
- 测试与回归全部通过。
- 做不到的项不得绕过或降低标准；如实写明，并作为失败上报。
