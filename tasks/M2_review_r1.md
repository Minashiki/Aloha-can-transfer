# M2 审查记录 r1（Claude）

- 审查对象：Cursor dispatch `ctx_faca1086f0ab`（交接单 `tasks/handover_M2_1.md`），交付记录 `tasks/M2_delivery_r1.md`
- 基线：`c6b6910`；审查范围 = `git diff` + 新增未跟踪文件
- 方式：只读

## 1. 合规

| 项 | 结果 | 依据 |
|---|---|---|
| 受保护文件未改动（PLAN、WORKFLOW、`.gitignore`、`environment.yml`、`check_env.py`、`aloha/*.xml`、场景 XML） | 通过 | 派发前记录了 11 个文件的哈希，`sha256sum -c` 全部显示“成功” |
| M1 代码未改动 | 通过 | `git status` 显示已跟踪文件中只修改了 `kin/__init__.py`、`dyn/__init__.py`（仅增加导出）|
| “自写”约束：模型代码不调用 MuJoCo 的计算函数 | 通过 | 在 `kin/fk.py`、`kin/spatial.py`、`dyn/rnea.py` 中 grep `mujoco.mj*`，只找到 `mj_id2name` / `mj_name2id`（名称查询）和 `mjtJoint` 枚举，没有 `mj_kinematics` / `mj_rne` / `mj_fullM` / `mj_jac*` |
| 没有 commit，只在 AIControl 中运行 | 通过 | HEAD 仍是 `c6b6910`；交付记录中的命令都带有环境前缀 |

## 2. 复跑

- `conda run -n AIControl python -B scripts/m2_validate.py`：退出码 0（15 项全部 PASS，最大绝对误差 ≤ 1.1e-14，Christoffel 一项为 6.3e-13）
- `pytest tests -q`：26 passed（包括 M0、M1 的测试）
- 人工查看了图表 `outputs/m2/coupling_curves.png`、`single_joint_shoulder.png`：坐标轴、图例和上限标注都齐全，与交付记录中的数值一致

## 3. 技术核对

- **mj_rne 不含 armature**：交付记录指出 MuJoCo 的 `mj_rne(flg_acc=1)` 不包含 armature，比较时需要另外加上 armature·q̈；同时用 `mj_fullM·q̈ + qfrc_bias` 做了第二路对照。两路都通过，说明这个处理是正确的。
- **耦合系数的不变性**：沿 shoulder 扫描时三对系数保持常数，沿 elbow 扫描时 elbow–wrist_angle 保持常数，这是正确的物理结果——M_ij 只依赖 i 之后的关节角，shoulder 与 elbow 的轴平行，waist 不参与。不是 bug。
- **手指自由度的推测**已经实测验证（臂关节块误差 6.7e-16，两种开度下臂关节块的差为 2.5e-4），结论可以写进报告。
- **容差依据**合理；Christoffel 符号用四阶中心差分，选择理由写得充分。

## 4. 问题

### F1【低】耦合扫描的变量选得不够好（属于 spec 自身的问题）

- 现象：spec 要求“沿 elbow、shoulder 扫描”，但从结构上就能知道，沿 shoulder 扫描三对系数都是常数（图中下面一行 3 条水平线），elbow–wrist_angle 沿 elbow 扫描也是常数。这几条曲线对报告没有信息量。
- 建议：改为沿 **elbow 和 wrist_angle**（必要时再加 forearm_roll）扫描。下面一行换成 wrist_angle 扫描，并补一张 elbow×wrist_angle 平面上 shoulder–wrist_angle 的热力图；在交付记录中用一句话说明为什么沿 shoulder 扫描是常数（可以写进报告）。

### F2【低】单关节驱动的速度太低，看不出动态耦合

- 现象：1 s 内转 0.5 rad 时，峰值速度 0.94 rad/s，其他关节的保持力矩几乎全部来自重力（惯性耦合只有 0.28 N·m，科氏 / 离心只有 0.12 N·m）。
- 影响：报告中的“关节耦合”部分只能说明“耦合很小”，体现不出速度对耦合的影响，也和 M3“比较不同运动速度”的主题脱节。
- 建议：保留现有的 1 s 情况，再增加一档快速轨迹：峰值速度接近关节速度上限 2 rad/s（例如 0.5 rad 在约 0.47 s 内完成，或 1.0 rad 在 0.94 s 内完成）。另外单独画一张**不含重力的动态耦合力矩**（M_ij q̈_j + 科氏 / 离心）图，给出峰值和与力矩上限的比值；并检查这时被驱动关节本身是否会饱和。

### F3【低】TCP 名称又被硬编码在 `kin/fk.py:29`

- 现象：`TCP_SITE = "left/tcp"` 是硬编码的，配置文件中的 `tcp_site` 也写了一遍。现在两处一致，但这和 M1 的 F1 是同一类风险。
- 建议：在 `robot.py` 中定义唯一的常量 `TCP_SITE = "left/tcp"`，`fk.py` 引用它；环境类在加载配置时检查 `cfg["tcp_site"] == TCP_SITE`，不一致就报错。M1 删掉的是错误的定义，不是“不该有常量”，两者不矛盾。

## 5. Cursor 对话状态

- 上下文用量：状态栏显示 **86.8%**，超过 65%。审批后的修改要开新对话，并附交接单。

## 6. 结论

M2 主体合格：交叉验证全部通过，误差比容差小 5 个数量级以上，“自写”约束得到遵守，报告用的参数表和图表都已生成。需要修改的只有 F1–F3 三项低优先级问题，主要是为了提高报告的质量。
