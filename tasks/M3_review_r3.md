# M3 审查记录 r3（Claude 验收：按审批结论修改后）

- 对象：Cursor follow-up dispatch `ctx_3620b2ee03a2`；交付记录 `tasks/M3_delivery_r2.md`
- 方式：只读复核，不再经过 Codex

## 合规
- 受保护文件（PLAN、WORKFLOW〔Claude 修订后重新记录了哈希〕、`.gitignore`、`environment.yml`、`check_env.py`、模型 / 场景 XML、M2 计算模块）`sha256sum -c` 没有失败项。
- 没有 commit。

## 逐项验收（对照 M3_review_r2 最终结论）
| 项 | 结果 | 依据 |
|---|---|---|
| 1 参考生成 | 通过 | 选用 lookahead 五次式，端点加速度取中心差分。我用 `m3_chk.py` 重算了 shoulder 的 q̈_d：medium 档 RMS 2.07，25 步滑动平均后为 1.91，纹波基本消失（r1 时分别是 4.44 和 1.96）；fast 档 τ_cmd 峰值从 64.6 降到 26.6 Nm。新旧对比表和 causal 方案的对比都已给出；总滞后 100 ms，交付中写明了 |
| 2 N1 速度回退 | 通过 | 复现用例逐段运行：第 1 段在 q=0.025、v=0 处制动停住，第 2 段 v=−1 开始反向，全程速度不超过上限；补充了停止和反向的测试 |
| 3 Can / 接触 | 通过 | 日志新增 `can_pos`、`can_quat`、`finger_can_contact`；38 组跟踪实验的接触步数都是 0，Can 位移 ≤ 6.3 µm |
| 4 C2 结构一致对照 | 通过 | noff 与 B1 最大差 1.10e-3 rad，略超 1e-3。同一状态下两者力矩差 1.8e-15 Nm，单步积分后差 1.06e-6 rad，差异来自 implicitfast 对 position 执行器阻尼的隐式处理，与 g(q) 无关，解释成立 |
| 5 IK 容差写入配置 | 通过 | `configs/m3_track.yaml:24` 写的是 `ik_pos_tol: 1.0e-4` |
| 6 结论补充 | 通过 | 给出了“跟得住”的成立范围（medium 及以下：峰值速度约为上限的一半，力矩余量约 40%，100 ms 滞后对应的 TCP 误差预算约 21 mm）；摩擦的措辞已改为“与摩擦平衡一致” |
| 回归 | 通过 | pytest 46 passed；check_env 通过；`m1_grasp_smoke` 输出 GRASP_OK |

## 记录到后续里程碑的事项（不阻塞 M3）
- **参考生成固定带来 100 ms 滞后**（lookahead 方案）。M6 / M7 的策略在 20 Hz 下输出动作块，这个滞后应该在策略设计和评测中说明。如果之后发现滞后影响抓取，可以再评估“对整段加速度设上界的因果方案”（交付记录中列为未做）。
- **fast 档 C3 的肩、肘仍有 0.2%–5% 的饱和**，这是物理加速度需求造成的。报告中如实说明，M4 默认用 medium 档。
- **腕部增益偏低**（原 F5），M4 抓着 Can 搬运时需要关注。

## 上下文用量
- Cursor（`ctx_be35b1b8e7c6` 系）：22%，已完成 2 次任务
- Codex：约 51.5%，已完成 4 次任务

## 结论
**M3 验收通过，可以提交。**
