# M7 审查 r3（Claude 验收）

对象：`tasks/M7_delivery_r2.md`；对照已审批的 `tasks/M7_review_r2.md` 第 3 节（即 `tasks/M7_dispatch_r2.md` 清单 1–10）。只读验收，一次性输出都在 Claude 的 scratchpad。命令均在 `conda run -n AIControl` 中执行。

**结论：清单 1–10 全部完成，验收通过；另发现一项残留问题（第 3 节），不阻断提交，需在报告中写明。**

## 1. 合规

| 项 | 结果 | 依据 |
|---|---|---|
| PLAN / WORKFLOW 哈希 | 通过 | 与派发前快照一致（PLAN `a90182e3…54d9`，WORKFLOW `09f4d845…803b`） |
| `outputs/m3`–`m6` | 通过 | 590 个文件的路径 / 大小 / mtime 清单与 M7 派发前一致 |
| 改动范围 | 通过 | 相对派发前的 sha256 快照，变化的正是交付列出的 11 个文件，另新增 `scripts/m7_det_compare.py`、`tasks/M7_delivery_r2.md`；`policy/rollout.py` 本轮未动 |
| `sample` 未改 | 通过 | `git diff 006a65b -- src/aloha_pp/policy/policy.py` 只有新增行（两个新方法） |
| 配置数值 | 通过 | `total_env_steps` 2e6、`n_envs` 8、`utd` 20、`layer_size` 2048、`target_entropy` 0 等与 r1 相同 |
| git | 通过 | HEAD `006a65b`，无暂存 |
| 测试 | 通过 | 本机 `pytest tests -q -p no:cacheprovider`：**117 passed**（111 + 6） |
| Cursor 上下文 | **73.5%**，已完成 1 次任务 | 超过 65%，不得再派任务给该对话（`ctx_423a62f094e0`） |

## 2. 清单逐项

| # | 判断 | 核对 |
|---|---|---|
| 1 评测录像隔离 | 通过 | `rl/evaluate.py` 的 `record_episode`：`none` 直接返回 False，`auto` 才有“第一局录像”；默认 `none`；`eval.json` 有 `videos`、`video_seeds`；测试覆盖 |
| 2 跨进程确定性回归 | 通过（附第 3 节） | Cursor 的 `det_a` / `det_b` 一致；我另起进程复跑：`bc_zero` 3 次、`bc_seeded` 1 次全部与 `det_a` 逐项相同 |
| 3 验证录像隔离 | 通过 | `trainer.py`：统计波次固定 `record=False`，出现视频即报错；录像另开 1 个环境跑 `val_seed_start`，记 `video_success`、`video_in_success: false`，不进 `n_success`、不参与选模 |
| 4 best / last 都评测 | 通过 | 交付第 6 节两条 `bc_dsrl` 命令；`m7_compare.py --dsrl LABEL=… --selected`，主结果按 `selected.json` 标注，标签对不上会退出 |
| 5 对比口径 | 通过 | 表含 `role`、`noise`、`recorded`；M6 行标“随机噪声（未固定）、种子 0–3 录像，仅作参考”；配对只做 `bc_seeded` / `bc_zero` 对各 `bc_dsrl_*`；`mcnemar_p` 为精确二项双侧 |
| 6 actor 方向测试 | 通过 | `test_actor_moves_the_rewarded_component_positive`，固定种子和线程 |
| 7 真实吞吐 | 通过 | `outputs/m7/benchmark_real/timing.json`：402 vec step、7420 次更新、1 次 16 + 1 验证、2 次保存，合计 104.6 s；外推 2e6 约 **18.5 h**（验证全部变慢一倍时约 20.1 h），未超 24 h，不改步数 |
| 8 数值健康日志 | 通过 | 冒烟与短程日志均有 `logp_mean`、`q_*`、`target_q_*`、`alpha`、`actor_abs_mean`；短程第 400 步 `logp_mean` −86.91，与 r2 纠正后的约 −86 吻合 |
| 9 `encode_actor_obs` 收紧 | 通过 | 有 `state` 或缺键即 `ValueError`；`_cond` 未改；测试覆盖 |
| 10 续训配置比对 | 通过 | `assert_resume_config`；改 `noise_scale` / `layer_size` 退出，改 `n_envs` 允许；空回放续训行为已在交付写明 |

## 3. 残留问题：`bc_dsrl` 偶发跨进程不一致

- 用冒烟 actor（`outputs/m7/smoke/train/ckpt_last.pt`，mtime 08:52，期间未变）跑 `bc_dsrl` 种子 0–2，共 8 个独立进程：Cursor 的 `det_a`、`det_b`，我的 `det_c`、`det_d`、`det_r1`–`det_r4`。**7 次逐项相同，`det_c` 一次从种子 0 起就不同**（`place_xy` 0.00603 对 0.00463，时长 13.2 对 13.3 s 等），成功标志和主原因相同。
- 同期 `bc_zero`（共 5 次进程）、`bc_seeded`（3 次）全部一致。
- 进程内：同一特征连续调用 actor 2000 次，GPU 输出逐位相同（1 种）。
- 原因未查清；推测与 CUDA 上 actor（2048 宽 MLP）的跨进程算法选择有关，没有证据。
- **影响**：正式 `bc_dsrl` 评测重跑时偶尔会得到逐位不同的轨迹，成功率可能有小幅波动；不影响统计口径，也不影响 `bc_zero` / `bc_seeded`。报告中写明“`bc_dsrl` 确定性评测在本机约 1/8 的进程中出现逐位差异，原因未定位”。
- 可选的后续缓解（需要你决定，不在已审批清单内）：评测时把 actor 推理放到 CPU（128 维输出、3 层 2048，开销很小）后再做多进程复核。

## 4. 只记录、不改代码的事项（交付第 7 节，已核对）

- SAC 与 SB3 的细节差异（critic 无 0.5 因子、更新顺序、critic 吃 w、初始化）。
- 每个转移 2.5 次更新（官方 5 次）。
- 目标熵 0、128 维下初期 log π 约 −86；短程中 Q 最大值升到约 945，由熵奖励推高（`α·|log π|` 约 4.7 / 步，`1/(1−γ)` = 100），会随 α 下降回落，正式训练靠健康日志观察。

## 5. 下一步

1. git 提交 M7（不 push）。提交范围：tracked 的 3 个改动文件 + `configs/m7_dsrl.yaml`、`scripts/m7_*.py`、`src/aloha_pp/rl/` 新文件、`tests/test_m7_dsrl.py`、`tasks/M7_*.md`（`handover_*` 被 `.gitignore` 排除）。
2. 正式训练（约 18.5 h）和四条正式评测、对比由用户启动，命令见 `tasks/M7_delivery_r2.md` 第 6 节。
3. 若后续还需要 Cursor 修改：`ctx_423a62f094e0` 已 73.5%，必须新开对话并写交接单。
