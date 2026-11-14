# M6 审查记录 r1（Claude）

- 对象：Cursor dispatch `ctx_d95eb308aab6`（新对话，按 `tasks/handover_M6_1.md` 启动）；交付记录 `tasks/M6_delivery_r1.md`
- 基线：`4d4f02b`（M5）。审查以 `git diff 4d4f02b` 加未跟踪新文件为准
- 方式：只读。验证脚本与临时输出放在 Claude 的 scratchpad

## 合规

| 项 | 结果 | 依据 |
|---|---|---|
| 受保护文件与 M5 数据集 | 通过 | 派发前对 PLAN、WORKFLOW、`.gitignore`、`environment.yml`、`check_env.py`、M1 / M3 / M4 / M5 配置、`data/retarget.py`、M2 计算模块、`aloha/*.xml`、`outputs/m5/dataset/demos.hdf5` 记录 sha256，交付后全部一致 |
| `outputs/m3–m5` 未被写 | 通过 | 三个目录的“路径 + 大小 + mtime”清单与派发前一致 |
| 不装包、不依赖 DPPO | 通过 | 交付声明未安装；源码中没有 einops / torchvision / hydra / dppo 的 import；`third_party/dppo` 只读 clone（commit `cc7234a`），被 `.gitignore` 排除 |
| 第三方许可证 | 通过 | `src/aloha_pp/policy/THIRD_PARTY_LICENSES` 含 DPPO MIT 全文；复制文件的文件头注明来源 |
| git | 通过 | HEAD 仍是 `4d4f02b`，没有 commit |
| 测试 | 通过 | 我复跑 `pytest tests -q`：93 passed |

## 已核实的交付主体

- **噪声接口**：`DiffusionPolicy.noise_shape = (16, 8)`，`sample(obs, noise=None, ddim_steps=None)`，η 固定 0；CPU / CUDA 上同一 obs + 同一 noise 逐位相同、checkpoint 往返逐位相同（测试）。满足 M7 的前提。
- **策略**：双相机共享 ViT + 各自 SpatialEmb，2 帧历史，状态 8 维（无物体真值、无关节角），UNet1D（1.39 M 参数），100 步余弦调度、预测 ε、DDIM 8 步，随机平移增强，EMA 0.995。动作 min/max 归一化只用 30 条训练示范；四元数在数据中没有半球翻转，送入环境前单位化。
- **执行**：块内第 k 步预览第 k+1 步，第 8 步预览第 9 步（测试覆盖）。环境组装移到 `src/aloha_pp/sim/c3.py`，`m4_run.py` 改为调用它；M4 E1 回归逐项一致，M5 两条示范用共享环境重放再次成功、低维观测逐位一致。
- **训练子集**：种子 0 抽 30 条，`source_id` 列表写入 checkpoint；其余 90 条只用于验证损失。
- **冒烟**：6 epoch，train loss 0.849 → 0.322，val loss 0.976 → 0.371；验证与评测都产出了成功率、失败分布和视频（成功率 0，dispatch 允许）。

## 问题

### F1（中）评测 / 验证时约 75% 的时间在反复创建渲染器

交付记录：一条 25 s 的 episode 墙钟约 110 s，正式训练的验证可能占 2.2 h。我用冒烟 checkpoint 跑了一条 5 s 的评测并做 cProfile（输出写在 scratchpad）：总计 22.4 s，其中 **`AlohaCanEnv._get_renderer` 累计 18.8 s**，`mujoco.Renderer.__init__` 被调用 201 次（16.1 s），EGL `make_current` / `free` 再加约 6.8 s；而 DDIM 采样总共 0.6 s、IK 0.7 s、渲染本身（`mjr_render`）0.36 s。

原因：`policy/rollout.py:128` 每拍用 256×256 渲染视频帧，`get_obs()` 又用 96×96 渲染观测；`env._get_renderer`（`sim/env.py:441`）在分辨率变化时关闭并重建渲染器，于是每拍重建两次。视频帧还是每拍都渲染，写文件时才按 fps 抽帧。

影响：正式训练的验证和 `m6_eval`（50 条）慢约 4–5 倍；M7 的 DSRL 要跑大量 episode，环境吞吐是关键。

**要求**：
1. 渲染器按分辨率缓存（每个 (h, w) 一个，`close()` 时全部释放），默认行为与现有测试不变；或在 rollout 中为视频单独持有一个渲染器。
2. 视频帧只在需要写入的拍渲染（按 fps 抽帧之前就跳过）。
3. 给出修改前后同一条 episode 的墙钟对比，并更新正式训练的时长估计；同时给出不录视频时每个控制拍的耗时拆分（物理 + 控制、观测渲染、策略推理）。

### F2（中）失败分类把 `ik_fail` 排在最前，会掩盖学习策略的真实失败方式

`policy/failure.py` 沿用 M5 的顺序，`ik_fail` 第一：整条 episode 中任一物理步 IK 不成功就归为 `ik_fail`。对规划器和重定向，IK 失败意味着规划错误，排第一合理；对学习策略，偶尔一个动作超出可达范围时控制器保持上一解，episode 仍可能继续抓取 / 放置。冒烟评测 4 条全是 `ik_fail`，看不出它是没抓到、没提起还是放错。

**要求**：评测输出中，主失败原因按任务过程判定（not_grasped / not_lifted / dropped_after_lift / not_released / outside / not_still / timeout），IK 失败单独成列：`ik_fail_ticks`（控制拍数）与占比，以及是否出现过。汇总给两种分布。M4 / M5 的分类函数行为不变。

### F3（低，记录）离屏图像重放有 1 个灰阶的不确定性

交付如实报告：同一状态两次离屏渲染最多差 1 个灰阶，低维观测逐位一致，测试按“像素差 ≤ 1”验收。这是 EGL 渲染的特性，接受。记录：M7 中“给定噪声确定性”指同一 obs 张量下的确定性，整条 episode 的逐位复现不能依赖图像。

### F4（低）冒烟评测没有成功视频

冒烟模型成功率为 0，dispatch 允许；成功视频留到正式训练后的评测。不需要修改。

## 上下文用量

- Cursor `ctx_d95eb308aab6`：33.4%，已完成 1 次任务。修改轮继续用它（第 2 次）。
- Codex `ctx_67512818f8e5` 链：约 27%，已完成 2 次任务；M6 复核是第 3 次，之后换对话。

## 结论（待 Codex 复核）

M6 主体满足 dispatch。需要修改：F1（渲染器缓存与视频抽帧，影响评测与 M7 吞吐）、F2（失败分类）。F3、F4 记录在案。修改后由 Cursor 重跑冒烟验证与评测、测试和回归。
