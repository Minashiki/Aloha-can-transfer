# M2 审查记录 r3（Claude 验收：按审批结论修改后）

- 对象：Cursor dispatch `ctx_d3e266463be4`（交接单 `tasks/handover_M2_2.md`，任务说明 `tasks/M2_dispatch_r2.md`），交付记录 `tasks/M2_delivery_r2.md`
- 方式：只读复核，不再经过 Codex

## 合规
- 受保护文件全部未改动：PLAN、WORKFLOW、`.gitignore`、`environment.yml`、`check_env.py`、`aloha/*.xml`、场景 XML、M1 代码（`sim/*.py`、`kin/ik.py`、`robot.py`），以及 M2 的计算模块（`fk.py`、`spatial.py`、`rnea.py`）。依据：`sha256sum -c` 没有任何一项失败。
- 本轮只改了 dispatch 允许的文件：`scripts/m2_validate.py`、`scripts/m2_coupling.py`、新增的 `tests/test_m2_validate.py`。

## 复跑与验收
| 项 | 结果 | 依据 |
|---|---|---|
| N1：拒绝非有限值 | 通过 | 一次性注入：用 runpy 把 `gravity_torque` 替换成返回 NaN，输出 `g ... FAIL nonfinite`，退出码 1（脚本 `<scratchpad>/m2_nan.py`）。新增的 pytest 覆盖了 4 种非有限值的情况 |
| 正常验收 | 通过 | `m2_validate.py` 退出码 0 |
| N2：条件数标注 | 通过 | `m2_validate.py:64–67` 定义了 `分母下限 1e-16 的条件数估计` 和 `理论上为无穷大` 两个标签，并用在 CSV 和图上 |
| E1：wrist_angle 扫描和热力图 | 通过 | 生成了 `coupling_vs_wrist_angle.*`、`coupling_shoulder_wrist_map.*`。交付记录给出了数值范围，以及沿 shoulder 扫描结果不变的解释 |
| E2：快速轨迹、去重力的耦合、主动关节力矩 | 通过 | 快速档的动态耦合是 1 s 档的 4.55 倍，理论值为 (1/0.46875)² ≈ 4.55，两者一致。被动关节最大 1.40 N·m（占力矩上限的 0.070）。主动关节峰值 10.84 / 7.49 N·m，不饱和，并给出了峰值时刻的力矩分解 |
| pytest | 通过 | 29 passed |
| check_env | 通过 | ALL CHECKS PASSED |

## 上下文用量
- Cursor（`ctx_d3e266463be4`）：状态栏 56.7%，已完成 1 次任务
- Codex：约 35.7%，已完成 3 次任务

## 结论
**M2 验收通过，可以提交。**
