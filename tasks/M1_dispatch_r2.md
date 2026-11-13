# M1 修改派发 r2（已获用户审批）

**先读这份交接单：`tasks/handover_M1_2.md`。** 然后读 `tasks/M1_review_r2.md` 的“最终结论”一节，以及 `tasks/M1_recheck_r1.md` 中的 N1–N4。

项目根目录：`/home/minashiki/AIControl/FinalHomework`。

## 硬性约束（与 `tasks/M1_dispatch.md` 相同，违反任何一条即判为不合格）

1. 只在 conda 环境 `AIControl` 中运行（`conda run -n AIControl ...`）。禁止用系统 Python、base 环境，禁止新建环境，不得改变已安装包的版本。交付记录中的命令要写出环境前缀。
2. 不得修改：`docs/PLAN.md`、`docs/WORKFLOW.md`、`.gitignore`、`environment.yml`、`scripts/check_env.py`。
3. 不得执行 `git commit` / `push` / `config` / `add`，不得添加 co-author。
4. 不得启动其他 agent；运行时不得引用 `third_party/`。
5. 只修改下面列出的内容。不要重构无关代码，不要改动 IK 的姿态误差和 Jacobian 实现（已验证正确）。

## 要修改的内容

1. **TCP 统一**
   - 在 `aloha/aloha.xml` 中新增 site `left/tcp`，位置在夹爪张开时两侧指垫的中点，+X 为接近方向。在 XML 注释中写清坐标轴约定。
   - 把它作为唯一的 TCP（`configs` 中的 `tcp_site` 改为它）。
   - 删除 `tcp_to_grasp_center`，以及 `robot.py` 中错误的 `TCP_SITE`。仓库中所有使用 TCP 的地方同步修改。
   - 验收：张开（q = 0.037）和闭合到 Can 直径时，`left/tcp` 到两侧指垫中点的距离都 < 1 mm（写明计算方法）。
2. **agentview 取景**
   - 调整相机，使两个 bin 完整在画面内，Can 和抓取阶段的夹爪也在画面内。
   - 验收：用相机内外参投影计算两个 bin 外沿的 8 个顶角，归一化坐标都在 [-1, 1] 内（输出数值）。附上 96×96 和 256×256 的 reset 截图，以及抓取中的截图，保存在 `outputs/m1/`。
3. **按 mode 配置目标格**
   - `fixed_1` 和 `random` 用右 bin 的 +X+Y 格（不变）；`fixed_2` 用对角的 −X−Y 格。
   - 目标格的中心和半尺寸写进配置，按 mode 生效。`fixed_2` 的 `place_pos` 放在新格子内。
   - 验收：单元测试覆盖两个 mode 的目标格。
4. **成功判定增加“已松手”**
   - 新增条件：两个手指的所有 geom 与 Can 的碰撞 geom 之间都没有接触。实现为可配置项，默认开启。
   - 验收：单元测试要包含三种情况：“夹着 Can 静止在格内 → 不成功”、“松手静止在格内 → 成功”、“格外 → 不成功”。
5. **暴露 IK 状态**
   - `step` 返回 `(obs, success, info)`，`info` 中包含 `ik_success` 和 `ik_pos_err` / `ik_rot_err`。脚本中统计并打印 IK 失败次数。
   - 容差数值不改，保持可配置。同步更新调用了 `step` 的脚本和测试。
6. **状态同步（N1）**
   - `mj_step` 之后，凡是要读取派生量的地方（成功判定、观测、特权信息、reset 后返回的观测），都要先同步运动学，可以用 `mj_kinematics`，必要时用 `mj_forward`。在代码注释中说明成功判定在哪个时刻评估。不得改变 timestep 和每个动作的物理步数。
   - 验收：新增动态测试。执行若干个会让手臂运动的动作后，返回的 TCP 位姿与对同一 q 独立做 FK 的结果一致（误差 < 1e-9）；特权信息中的物体位姿与当前 freejoint 状态一致。
7. **物体中心（N2）**
   - 物体中心统一改用碰撞圆柱 geom 的世界位姿，删除 `_mesh_aabb_center` 和配置中的 `can_center_local`。如果 `can_rest_z` 等配置依赖旧算法，也一并更正。
   - 验收：在至少两个物体朝向下，与直接读取 `geom_xpos` 的结果一致。
8. **IK 可达性脚本（N3）**
   - `scripts/m1_ik_reach.py`：Can 中心的采样边界从场景尺寸推导（扣除墙厚和 Can 半径；右 bin 还要扣除隔板）。有任何一个点不收敛就退出码非 0，同时保留完整的统计输出。
   - 验收：用一次性方式注入一个失败点（不要把注入代码留在仓库里），确认退出码非 0，并在交付记录中写出注入方法。
9. **滑移度量（N4）**
   - `scripts/m1_grasp_smoke.py` 中的滑移改为 `R_tcp.T @ (p_can − p_tcp)` 相对于保持开始时刻的变化，每个物理步都采样（可以在环境外逐物理步记录，或者给环境加一个可选的逐步回调）。
   - 验收：抓取仍然达到门槛（抬起 ≥ 5 cm、保持 ≥ 3 s、滑移 < 5 mm）。另外用单元测试或一次性脚本，证明“刚体随 TCP 平移和旋转时滑移为 0”。
10. **低优先级**
   - 删除 `robot.py` 中与 XML / 配置重复的运行时常量（测试中独立写出的期望值保留）。
   - `AlohaCanEnv` 支持 `with` 语句（`__enter__` / `__exit__` 调用 `close`）。

## 交付物

- 上述修改。
- `tasks/M1_delivery_r2.md`：改动文件清单（新增 / 修改 / 删除）；运行过的命令及输出摘要，至少包括 `pytest tests -q`、`check_env.py`、`m1_grasp_smoke.py`、`m1_gravity_hold.py`、`m1_ik_reach.py` 以及上面各项的验收输出；未完成或存疑的点。

## 完成标准

- 上面每一项的验收全部满足；`pytest` 全部通过（包括 M0 的测试）；`check_env.py` 退出码 0。
- 做不到的项不得绕过或降低标准，在交付记录中如实写明并作为失败上报。
