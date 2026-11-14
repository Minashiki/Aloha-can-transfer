# M6 审查记录 r3（Claude 验收：按审批结论修改后）

- 对象：Cursor follow-up `ctx_db23d91c655a`（同一对话 `ctx_d95eb308aab6`）；交付记录 `tasks/M6_delivery_r2.md`
- 方式：只读复核，不再经过 Codex

## 合规

- 受保护文件与 M5 数据集 `sha256sum -c` 全部成功；`outputs/m3–m5` 清单与派发前一致；HEAD 仍是 `4d4f02b`，没有 commit。
- `git status` 相对派发前新增 `sim/env.py` 修改（渲染器缓存，属于第 2 项）和 `tasks/M6_delivery_r2.md`。
- 冒烟重跑写到新目录 `outputs/m6/smoke_r2/`，r1 的 `outputs/m6/smoke/` 保留。
- 我复跑 `pytest tests -q`：97 passed。

## 逐项验收（对照 M6_review_r2 最终结论）

| 项 | 结果 | 依据 |
|---|---|---|
| 1 N1 四元数 | 通过 | `canonicalize_quaternions` 改为无状态 w ≥ 0（我读了 `quaternion.py:8–22`）；`pose_state` 被训练窗口与在线 `_cond` 共用。数据中动作 \|w\| 最小 0.500、观测 0.473，半球规则不会在 0 附近翻转。测试：`demo_0004` 中含 w < 0 的窗口，训练路径与在线路径条件向量逐项相同，换成 −q 仍相同 |
| 2 F1 渲染器 | 通过 | `env._renderers` 按 (h, w) 缓存，`close()` 全部释放，`_renderer` 属性保留给旧测试；视频按 fps 只渲染写入的拍。同一设定 5 s 录视频 episode 墙钟 **20.40 s → 1.71 s**；无视频时每拍物理 + 控制 6.5 ms、观测渲染 0.6 ms、推理 22 ms / 块 |
| 3 F2 失败分类 | 通过 | 主原因按全部闭合段的任务过程判定（`not_grasped` / `not_lifted` / `dropped_after_lift` / …），不再返回 `ik_fail`；IK 按控制拍分当前 / 预览 / 任一计数，成功 episode 也记。冒烟评测现在能看出主原因是 `not_grasped`。M4 / M5 分类函数未改 |
| 4 F3 渲染环境 | 通过 | 训练 `render.json`、评测 `eval.json` 记录 MuJoCo 3.17.0、`MUJOCO_GL=egl`、GL renderer `NVIDIA GeForce RTX 5080`、驱动 595.99.02；图像重放阈值仍为像素差 ≤ 1，测试说明写明依赖本机 GPU |
| 5 N2 时长 | 通过 | 完整验证损失一次 3.56 s，改为 `val_loss_every: 20`；正式训练估计 0.82 h 训练 + 0.10 h 验证损失 + 0.15 h 闭环 ≈ **1.1 h** |
| 6 N3 | 通过 | 评测入口沉降后调用 `reset_controller_after_settle`；测试比较 home_q、IK 种子、参考流初值与 M5 路径逐项相同。`env.reset` 默认过程未改 |
| 7 重跑与回归 | 通过 | 冒烟 train loss 0.849 → 0.322、val loss 0.976 → 0.370；训练中验证与 `m6_eval` 产出成功率、主原因、IK 计数和视频；`check_env` 通过、`GRASP_OK`、M4 E1 共有列 mismatches 0、两条 M5 示范重放再次成功 |

## 记录到后续里程碑的事项

- **正式训练**由用户启动（约 1.1 h）：`conda run -n AIControl python -u scripts/m6_train.py --config configs/m6_dp.yaml --out outputs/m6/train`；结束后 `scripts/m6_eval.py --checkpoint outputs/m6/train/ckpt_last.pt`（种子 0–49，超时 25 s）。M7 以这份 BC checkpoint 为起点，M7 spec 需在拿到 BC 成功率后再定提升目标。
- **M7 接口**：`DiffusionPolicy.noise_shape = (16, 8)`，`sample(obs, noise)` η = 0 确定性；评测日志格式 `aloha_pp.eval.v1`（`method` 字段可写 `bc_dsrl`）。
- **吞吐**：无视频时每控制拍约 7 ms 环境 + 每 8 拍一次约 22 ms 推理，即约 10 ms / 拍；一条 25 s episode 约 5 s 墙钟。M7 的 DSRL 样本预算按此估算。
- **图像确定性**只在本机 GPU 的 EGL 上成立（像素差 ≤ 1）；整条闭环 episode 不能依赖图像逐位复现。

## 上下文用量

- Cursor `ctx_d95eb308aab6`：**73%**，已完成 2 次任务。超过 65%，M7 新开 Cursor 对话。
- Codex 链：约 42.9%，已完成 3 次任务；M7 复核新开 Codex 对话。

## 结论

**M6 验收通过，可以提交。**
