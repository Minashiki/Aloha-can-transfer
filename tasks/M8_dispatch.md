# M8 计划：汇总与技术报告（草稿，待用户确认）

> 起草：2026-11-15，Claude。依据：`tasks/handover_claude_3.md`、`docs/PLAN.md` §10 M8，以及用户 2026-11-15 的答复。
> **本里程碑不派发给 Cursor**，报告由 Claude 撰写（用户 2026-11-15 决定）。本文件是计划，不是派发 spec。

## 0. 现状快照（2026-11-15）

- HEAD `e4ac982`，与 `origin/main` 一致，工作区干净。
- `docs/PLAN.md` sha256 `a90182e3…54d9`，`docs/WORKFLOW.md` `09f4d845…803b`（与交接单一致）。
- `outputs/` 文件清单（1236 个文件，路径 / 大小 / mtime）已存在 Claude 的 scratchpad，完成后做 diff。

## 1. 用户已确定的事项

| 项 | 决定 |
|---|---|
| 语言 / 篇幅 | 中文；篇幅不设上限，以讲清每处技术细节为准。分层写清：仿真与模型、运动学 / 动力学、底层控制、规划器、示范数据、Diffusion Policy、DSRL；适量给出公式与伪代码 |
| 结构 | 按实验、算法、效果组织技术总结，不逐条对应作业要求（作业要求在附录里列一张对照表，标出对应的章节） |
| 插图 | `report/figures/`，不进 git |
| 视频 | 报告中用到的视频复制到 `report/videos/`，不进 git |
| 补跑实验 | 不补跑 |
| 执行者 | Claude 撰写，不派发给其他 Agent |

## 2. 交付物

```
report/
  report.md          # 技术报告（入 git）
  figures/           # 插图（不入 git）：从 outputs/ 复制的已有图 + 新绘的汇总图
  videos/            # 视频（不入 git）：从 outputs/ 复制
```

- `.gitignore` 增加 `/report/figures/`、`/report/videos/`。`*.mp4` 已经全局排除，这两行是为了把 png 也排除。
- 新汇总图由一个只读 `outputs/`、只写 `report/figures/` 的脚本生成（是否入 git 见第 6 节问题 1）。
- 不写任何 `outputs/m*/`，不改代码、配置、测试。

## 3. 报告大纲（`report/report.md`）

0. **摘要**：任务、技术路线、主结果表（规划器 50/50；BC `bc_zero` 36/50、`bc_seeded` 32/50；DSRL `best` 46/50、`last` 38/50）。
1. **系统总览**：分层架构图，用 Mermaid 或 ASCII 画出“20 Hz 上层 → IK → 500 Hz 参考 → C3 → MuJoCo 2 ms”；软件环境；仓库结构；各模块与里程碑的对应关系。
2. **仿真环境与机器人模型**：
   - ALOHA 左臂 vx300s，加 Can 场景（robosuite 资产，MIT 许可）。
   - 模型修复：重复的 equality 约束、`actuatorfrcrange`、motor / position 两版执行器、阻尼重设及理由、夹爪 forcerange。
   - 力矩上限按假设值写明，未核对真实舵机。
   - TCP 定义；Can 缩放 0.9 与碰撞几何方案 B；成功判定（含“已松手”）；评测配置 `fixed_1` / `fixed_2` / random。
3. **运动学与动力学**：
   - 正运动学 FK、几何雅可比；DLS 逆运动学（公式、阻尼与初值选取、腕部共轴奇异）。
   - RNEA（伪代码）；M(q) 含 armature；C 与 g；`qfrc_bias` / 阻尼 / frictionloss 分别出现在 MuJoCo 的哪一项。
   - 与 MuJoCo 的交叉验证误差。
   - 关节耦合：M 非对角项随构型的变化、单关节驱动时的耦合力矩、动态耦合（M2 图表）。
4. **底层：参考生成与关节控制**：
   - 参考生成：lookahead 五次插值；速度上限下按速度缩放时长；等待模式滞后 100 ms，preview 模式滞后 50 ms。
   - 控制器 B0 / B1 / C1–C4 的控制律公式，C3 为主控制器。C2 实际用 `qfrc_bias`（与 PLAN 原措辞的差异要注明）。
   - 跟踪实验：2 组位置 × 3 档速度 × 6 个控制器。
   - 指标：RMSE、最大误差、力矩饱和占比、速度超限占比。
   - 结论：B0 的重力下垂、C2 与 B1 一致、fast 档肩部有少量饱和、C4 在接触时积分饱和。
5. **经典规划器与任务实验**：
   - 状态机伪代码与路点；夹爪开合前先等待参考滞后与稳定。
   - E1：2 组位置 × 3 档速度；E2：随机种子 0–49，50/50；E4：preview 对等待模式。
   - 放置误差约 4.5 mm，来源是摩擦带内稳态误差（有消融实验）。
   - 腕部误差与力矩峰值；关节运动与力矩曲线。
6. **示范数据**：
   - 来源：robomimic Can-PH v1.4.1。
   - 部分重定向 + 合成放置的流程：动作重标注、按段时间缩放、夹爪映射、闭环重执行与筛选。
   - 统计：120/200，失败原因分布。
   - 数据集格式；规划器加噪声的保底示范。
7. **Diffusion Policy（BC）**：
   - 观测编码：双相机 96×96、ViT，本体状态，2 帧，四元数取 w ≥ 0。
   - UNet1D；DDPM 训练目标与 DDIM（η = 0）采样公式。
   - 动作块：预测 16 步、执行 8 步，并预览第 9 步。
   - 训练设置：30 条示范、2000 epoch，loss 曲线。
   - 评测：种子 0–49，失败模式。
   - 代码来源：DPPO `cc7234a`，MIT 许可。
8. **DSRL：噪声空间 SAC**：
   - 噪声空间 MDP 的定义（公式）；SAC 的目标函数；tanh × 1.5 的动作变换。
   - 非对称 critic：actor 用 272 维冻结特征，critic 另加 30 维特权状态。
   - 奖励；训练 / 验证 / 评测种子分离；与官方 DSRL 超参数对照，以及与 SB3 实现的差异。
   - 训练动态：崩溃—恢复、Q 与 α 的变化；时间上对应，因果未经实验证实。
   - 选模规则：事先声明，`best` 与 `last` 两者都报告。
   - 正式评测：Wilson 区间、精确 McNemar 配对检验。
9. **综合对比与讨论**：
   - 规划器、BC、DSRL 在成功率、放置误差、用时上的对比；失败模式对比。
   - 局限：单个训练种子、30 条示范、2e6 步（官方 20e6）、跨进程偶发逐位差异、录像改变闭环轨迹、Can 质量 11.5 g、力矩上限为假设值。
10. **复现**：环境与各阶段命令（`conda run -n AIControl ...`）、产物路径。

附录：
- A. 作业要求 → 章节对照。
- B. 与原计划的差异：C2 实际用 `qfrc_bias`、O1 未做、DPPO 未做、速度档定义、M5 的重定向方式等。
- C. 视频清单。
- D. 参考文献与代码来源。

## 4. 插图计划

**复制已有的图**（来自 `outputs/`，复制时改名并加章节前缀）：
- M0 / M1：腕部相机、agentview、抓取。
- M2：耦合热图与曲线、单关节驱动、动态耦合、腕部奇异。
- M3：`b0_gravity_sag`、`c2_vs_b1`、`c4_contact`、`fixed1_fast_error_torque`、`rmse_vs_speed`。
- M4：`e1_*`、`e2_scatter`、`e4_compare`。
- M5：`can_xy`、`place_xy`、`retarget_reasons`、`slowdown`、`length`、`human_fraction` 等，以及观测样例。
- M6：`loss.png`。

**新绘的汇总图**（脚本只读 `outputs/` 下的 jsonl / json / csv）：
1. M6 训练中验证成功数随 epoch 的变化。
2. M7 训练曲线：验证成功数、训练局成功率、Q 均值、α 与 log π，标出热身期、`ckpt_best` 的位置、约 688k 步的恢复点。
3. 五种方法的成功率与 Wilson 95% 区间：规划器、`bc_zero`、`bc_seeded`、`best`、`last`，M6 BC 用灰色标为参考。
4. 各方法失败原因的堆叠条形图。
5. 成功局放置误差与用时的分布（箱线图）。
6. 视需要补一张：M3 / M4 的关节轨迹、力矩与饱和汇总。

## 5. 视频计划（共约 1.6 MB，全部复制到 `report/videos/`）

| 来源 | 内容 |
|---|---|
| `outputs/m1/grasp.mp4` | 抓起冒烟 |
| `outputs/m3/videos/B0_fixed_1_fast.mp4`、`C3_fixed_1_fast.mp4` | 控制器对比 |
| `outputs/m4/videos/E1_C3_*`（6 条）、`E2_C3_random_s00.mp4` | 2 组位置 × 3 档速度，以及随机位置 |
| `outputs/m5/videos/replay_01_retarget_0.mp4`、`replay_00_planner_noise_1001.mp4` | 示范回放 |
| `outputs/m6/train/eval/videos/seed00–03.mp4` | BC |
| `outputs/m7/video_bc_dsrl_best/videos/seed00–03.mp4` | DSRL best |

报告中要注明：BC 与 DSRL 的录像会改变闭环轨迹，因此视频只作示意，不能代替无录像的正式评测。

## 6. 待用户确认

1. **汇总脚本放哪**：Claude 写 `scripts/m8_figures.py` 并入 git，可以复现插图；但这超出了 WORKFLOW §1 规定的 Claude 写权限，需要你批准。另一个做法是放在 scratchpad、不入 git，报告里只说明插图怎么生成。**建议入 git。**
2. **是否复核**：报告写完后，要不要派 Codex 只读复核一轮，核对报告中的数值、引用和公式与 `outputs/`、`tasks/` 是否一致（写 `tasks/M8_recheck_r1.md`）？若要，按 WORKFLOW 走裁决与分歧项流程。**建议复核一轮**，因为报告引用的数值很多。
3. **其他细节**：
   - 报告文件名定为 `report/report.md`。
   - `.gitignore` 增加两行；这也是 Claude 改非 `tasks/` 文件，随第 1 点一并批准。

### 用户答复（2026-11-15）

1. 画图脚本 `scripts/m8_figures.py` 入 git。
2. 批准 Claude 写新文件（报告、插图脚本等）；**不得修改任何现有文件和代码**，需要修改时先报用户审批。因此不改根目录 `.gitignore`，改为新建 `report/.gitignore`（忽略 `figures/`、`videos/`）。
3. 报告完成后派 Codex 只读复核一次（`tasks/M8_recheck_r1.md`），Claude 逐条裁决。

## 7. 合格标准

- `report/report.md` 覆盖第 3 节的全部章节。每个数值都能追溯到 `outputs/` 或 `tasks/` 的审查结论；推测性的内容都标明“推测 / 未验证”。
- 报告引用的所有图和视频都在 `report/figures/`、`report/videos/` 中，相对链接有效，可以用脚本检查。
- 完成后 PLAN / WORKFLOW 的哈希不变，`outputs/` 文件清单不变。
- `git status` 只出现 `report/report.md`、`.gitignore`、`tasks/M8_*`，以及经批准的脚本。
- 提交一次，信息为 `M8：…`；不加 Co-Authored-By，不 push。
