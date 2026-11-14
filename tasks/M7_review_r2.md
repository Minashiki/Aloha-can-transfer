# M7 审查 r2（Claude 最终结论，交用户审批）

对象：`tasks/M7_review_r1.md`（Claude）与 `tasks/M7_recheck_r1.md`（Codex）。本文件逐条裁决 Codex 的意见，给出交 Cursor 修改的最终清单。

**结论：采纳 Codex 的全部实质意见，没有分歧项，不进入第 2 轮复核。** Codex 的 CPU bandit 实验已经证明 actor 的学习方向正确（R4 的风险降低，但测试仍要加入仓库）。Codex 补出的 N1（`videos=none` 在库函数里仍会录像）和 N2（吞吐估计漏算特征编码）都是真实问题，已由我对照代码确认，必须在正式训练前修。

## 1. Codex 只改了 recheck 文件

- 复核前后项目文件清单（929 个文件，排除 `third_party/`、`.git/`）`diff`：只多出 `tasks/M7_recheck_r1.md`。
- PLAN / WORKFLOW 哈希不变；tracked 改动仍只有 Cursor 的 3 个文件。
- Codex 对话：`gpt-6.1-sol`、effort `medium`、`service_tier` `default`（会话 jsonl 核对）；上下文用量 96050 / 475000 = **20.2%**，已完成 1 次任务。

## 2. 逐条裁决

| Codex 意见 | 裁决 | 依据 |
|---|---|---|
| 第 1 节合规各项“部分同意”（历史快照、安装记录无法反向重证） | 采纳 | 属于复核者无法独立重测的项，接受记录即可，不影响结论 |
| R1 同意；建议记录 CPU 线程 / 后端 / 版本元数据 | 采纳 | 评测日志已有 `render` 硬件信息；R2-3 的确定性回归中同时写明 torch / CUDA / 驱动 / 线程数 |
| R2 部分同意：CPU 上录像未复现轨迹差异，不能把“录像破坏复现”说成跨后端的普遍性质 | **采纳（措辞）** | 我的证据来自本机 GPU + EGL；Codex 的 CPU 结果不矛盾。r1 的结论改为“本机 GPU / EGL 下观测到”，处理方法（统计与录像隔离）双方一致 |
| R2-1 只加 CLI 不够（N1） | **采纳** | 已核对 `rl/evaluate.py`：`want = videos == "all" or index == 0 or not (have_success and have_failure)`，`index == 0` 永远录像，没有 `none` 分支 |
| R2-3 回归应确认库函数不调用录像渲染，并覆盖 `bc_dsrl` 确定性 | 采纳 | 加入修改清单 |
| R3 同意；不能事后按 0–49 结果改主 checkpoint | 采纳 | `selected.json` 是事先声明的主结果；best / last 同时报告只作诊断 |
| R4 同意；bandit 已在 CPU 上验证（w 从 0 到 0.878） | 采纳 | 测试仍要入库：固定种子和线程，断言可实现的小问题 |
| R5 部分同意：Wilson 区间重叠不是显著性检验，应给配对的不一致计数 | **采纳** | 对比改为逐种子配对：列“提升 / 退化”计数，并给精确 McNemar（二项）p 值；Wilson 区间只作描述 |
| 第 4 节 1：UTD 2.5 vs 5 是预算取舍 | 采纳 | — |
| 第 4 节 2：初期熵量级 “+180” 算错，tanh 修正后约 −86（log π），维数倍率约 4.6 | **采纳（纠正 Claude）** | 128 维标准正态 log 密度均值约 −181.6；tanh 修正项每维期望约 +0.75，合计约 +96，得约 −86，与 Codex 实测 −85.76 一致。r1 的数字作废；结论“不是缺陷”不变 |
| 第 4 节 3 | 采纳 | — |
| SAC 与 SB3 的细节差异（critic 损失无 0.5 因子、更新顺序、critic 吃 w 而非 w/1.5、初始化不同） | 采纳为“报告中写明”，不改代码 | 都不是方向错误；Codex 也没有给出会让学习失败的证据 |
| N1 `videos=none` 仍录像 | **采纳，必改** | 见上 |
| N2 吞吐 benchmark 漏算两次 `encode_actor_obs`、拷贝、写回放、checkpoint I/O、视频 | **采纳，必改** | 已核对 `trainer.benchmark_collection`：只有 `sample` + `vec.step`。正式训练约 17 h，估计偏低就可能超出预算 |
| N3 缺少长训练数值健康日志 | 采纳 | 训练日志增加 mean log π、Q 与 target Q 的均值 / 最大值、α、actor 确定性输出的均值幅度 |
| N4 `encode_actor_obs` 在缺 `tcp_pose` 时会接受任意 `state` 张量 | 采纳（低优先级） | 真实路径没有泄漏；收紧新方法的输入：只接受两路图像 + `tcp_pose` + `gripper_opening`，出现 `state` 键直接报错。M6 的 `_cond` 不动 |
| N5 续训时 checkpoint 内配置与当前 yaml 可能分叉；空回放续训不重新热身 | 采纳 | 续训时比对不可变配置（`noise_scale`、SAC 超参数、网络尺寸、`n_envs` 外的种子设置），不一致就报错退出；交付写明空回放续训的行为 |
| N6 验证中录像的那条 episode 直接计入成功率 | **采纳** | 验证的 16 条一律不录像；另跑 1 条录像 episode（同 `val_seed_start` 种子，单独环境），只出视频、不计入统计、不参与选模 |

## 3. 交给 Cursor 的修改清单（审批后派发）

> 2026-11-14 用户要求：原实现对话 `ctx_b8bb6c9c7d64` 经历过上下文压缩，不再使用。修改由新 Cursor 对话执行：交接单 `tasks/handover_M7_3.md`，任务说明 `tasks/M7_dispatch_r2.md`（内容与本节相同）。

必做：

1. **评测录像隔离**（R2、N1）：`m7_eval.py` 加 `--videos {none,auto,all}`，默认 `none`；`evaluate_latent` 在 `none` 时不调用任何录像渲染；`eval.json` 记录请求的录像模式和实际录像的种子。加测试：`none` 模式下录像渲染调用次数为 0。
2. **确定性回归**（R1、R2-3）：两个独立进程，各用 `m7_eval.py --videos none` 在正式 BC checkpoint 上对 `bc_zero`、`bc_seeded`、以及冒烟 actor 的 `bc_dsrl` 跑种子 0–2（`--out outputs/m7/regression/det_a`、`det_b`），`episodes.csv` 的成功标志、主原因、IK 拍数、`place_xy`、时长逐项相同；写明 torch / CUDA / 驱动版本和线程数。
3. **验证录像隔离**（N6）：16 条验证不录像；另跑 1 条录像 episode，只出视频，不计入成功率、不参与选模。
4. **best / last 都评测**（R3）：交付中的正式评测命令包含 `ckpt_best.pt` 和 `ckpt_last.pt` 两条 `bc_dsrl`；`m7_compare.py` 接受多个 DSRL 结果并分别标注；`selected.json` 指向的那个标为“主结果（事先声明）”。
5. **对比口径**（R5）：表中写明每种方法的噪声来源和是否录像；M6 正式评测一行标为“随机噪声（未固定）、种子 0–3 录像，仅作参考”；配对比较（`bc_seeded` vs `bc_dsrl_*`、`bc_zero` vs `bc_dsrl_*`）给出提升 / 退化计数和精确 McNemar p 值；Wilson 区间只作描述。
6. **actor 方向测试**（R4）：单步 bandit，固定种子和线程，CPU 上几秒内完成，断言确定性 w 的目标分量明显大于 0。
7. **真实吞吐复测**（N2）：用正式配置（8 环境、2048 宽、UTD 20），在 `outputs/m7/benchmark_real/` 下跑一次真实训练循环的短程计时（热身之后至少数百个 vec step，含特征编码、写回放、SAC 更新、一次验证、一次 checkpoint 保存），分项给出耗时，重新估算 2e6 步的总时长。超过约 24 h 时，给出能在 24 h 内完成的 `total_env_steps` 并写明代价；不要改 UTD 或网络宽度。
8. **数值健康日志**（N3）：训练日志增加 mean log π、Q 与 target Q 的均值 / 最大值、α、actor 确定性输出的均值绝对值。

低优先级（同轮完成）：

9. `encode_actor_obs` 收紧输入（N4），加测试。
10. 续训时比对不可变配置，不一致报错（N5）；交付写明空回放续训的行为。

报告中写明、不改代码：SAC 与 SB3 的细节差异（第 2 节）；每个转移 2.5 次更新；目标熵 0 在 128 维下初期 log π 约 −86。

修改后：`pytest tests -q` 全部通过；更新 `tasks/M7_delivery_r2.md`（改动文件、命令、结果，以及更新后的正式训练、评测、对比命令和时长）。约束不变：只在 `AIControl` 中运行，不装包，不写 `outputs/m3`–`m6`，不做 git 写操作。

## 4. 分歧项

无。
