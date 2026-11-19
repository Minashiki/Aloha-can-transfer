# M9 派发说明：DPPO 微调（上层扩展实验）

> 先读交接单 `tasks/handover_M9_1.md`，再读本文件。
> 起草：2026-11-19，Claude。用户已确定：本次只做 DPPO，DSRL-NA 以后再考虑；critic 用特权状态 + 冻结特征；训练预算 2×10⁶ 个转移、上限 24 h；结果补进本地报告；本月内完成。

## 0. 环境与硬性约束

- **运行环境**：所有运行、测试只能在 conda 环境 `AIControl` 中进行（`conda run -n AIControl ...`；长任务用 `conda run -n AIControl --no-capture-output python -u ...`）。**不得安装任何包。** 环境中没有 pandas、einops、hydra、gym、torchvision。
- **只新增，不修改**：
  - 本里程碑只能**新增**文件；
  - 现有的代码、配置、测试、模型、文档（包括 `src/aloha_pp/policy/`、`src/aloha_pp/rl/`、`scripts/m*_*.py`、`configs/*.yaml`、`.gitignore`、`README.md`）一律不得修改；
  - 如果确实需要改动现有文件，先用 `ask` 报告协调者，说明要改哪一处、为什么、默认行为是否不变，得到答复后再动手。
- **输出隔离**：
  - 只写 `outputs/m9/`，回归和复跑一律用 `--out` 写到 `outputs/m9/regression/`；
  - 不得写 `outputs/m0`–`m8`、`report/`、`media/`、`docs/`。
- **不得**：git commit / add / stash / checkout；运行正式训练（见第 6 节）；修改 `third_party/`（只读参考）。
- **离屏渲染**：用 `MUJOCO_GL=egl`。

## 1. 目标

在冻结前的 M6 BC（`outputs/m6/train/ckpt_last.pt`）上实现 **DPPO**（Diffusion Policy Policy Optimization，Ren et al.），即用 PPO 微调扩散策略的去噪链。要求：

1. 与 M7 DSRL 在同一环境、同一动作块接口、同一底层（C3）、同样的种子划分和评测口径下比较；
2. 提供冒烟训练、真实循环计时、正式训练命令、正式评测和对比脚本；
3. 正式训练与评测由用户启动。

## 2. 参考实现

官方仓库在本地 `third_party/dppo`（commit `cc7234a`，MIT 许可，也是 M6 复制扩散代码的那个版本）。对照以下文件：

- `cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp_img.yaml`（Can 图像微调的超参数）；
- `model/diffusion/diffusion_ppo.py`、`model/diffusion/diffusion_vpg.py`（去噪链的对数概率、PPO 损失、裁剪系数随去噪步变化、η 的处理）；
- `agent/finetune/train_ppo_diffusion_img_agent.py`（rollout、GAE、奖励缩放、更新循环、验证模式）；
- `model/common/critic.py`（critic MLP 结构）。

需要的代码复制或改写到本仓库的新文件中，文件头注明来源（文件名 + commit），并在新包里附上 DPPO 的 MIT 许可证。**不得把 `third_party/dppo` 作为依赖导入。**

本文件没有写明的细节，按官方实现；与官方不同的地方，逐条列在交付记录中。

## 3. 方法设定

### 3.1 去噪链与策略

| 项 | 设定 | 说明 |
|---|---|---|
| 基础策略 | M6 `ckpt_last.pt` 的 EMA 网络（ViT + SpatialEmb + UNet1D） | 与 M7 相同的 BC |
| 微调网络 `actor_ft` | 基础网络的完整副本，**全部参数可训练**（含 ViT 和 SpatialEmb） | 与官方一致；基础网络保持冻结，供对照和回归 |
| DDIM 步数 | 8（与 M6 采样一致），**8 步全部微调**（`ft_denoising_steps = 8`） | 官方是 5 步中 5 步全部微调；保持 M6 的 8 步，以免改变 BC 本身 |
| 训练时采样 | DDIM，η = 1（官方 `EtaFixed base_eta 1`，不学习 η），每步标准差下限 `min_sampling_denoising_std` = 0.1，噪声截断 `randn_clip_value` = 3，$\hat x_0$ 裁剪到 [−1, 1] | |
| 对数概率 | 每个去噪步为高斯分布，标准差下限 `min_logprob_denoising_std` = 0.1；按动作维度求和 | 与官方一致 |
| 验证 / 评测时采样 | η = 0（官方 `deterministic=True`）；初始噪声按种子固定为 $\mathcal N(0, I)$，与 M7 的 `bc_seeded` 相同 | 这样可以与 `bc_seeded` 逐种子配对。另提供初始噪声 $w = 0$ 的评测选项，与 `bc_zero` 配对 |
| 动作块 | 预测 16 步、执行 8 步，执行段最后一步 preview 第 9 步；一个 RL 步 = 一个动作块 | 与 M6 / M7 相同，复用现有的执行器和环境封装（只调用，不修改） |

### 3.2 PPO

| 项 | 本项目 | 官方 Can 图像配置 |
|---|---|---|
| 并行环境 | 8（M7 实测吞吐最高）；冒烟基准中对比 8 和 16 | 50 |
| 每次迭代的转移数 | 每个环境 `n_steps` = 1875 步，共 15 000 个转移 | 50 × 300 = 15 000 |
| 总转移数 | **2×10⁶**（与 M7 DSRL 相同，按动作块计），约 134 次迭代 | 151 次迭代，约 226 万 |
| γ（每个 RL 步） / GAE λ | 0.999 / 0.95 | 相同 |
| `gamma_denoising` | 0.99 | 相同 |
| 裁剪系数 | `clip_ploss_coef` 0.01，`clip_ploss_coef_base` 0.001，`clip_ploss_coef_rate` 3 | 相同 |
| `target_kl` / `update_epochs` / batch / `grad_accumulate` | 1 / 10 / 500 / 15 | 相同 |
| 学习率 | actor 5e-5（余弦退火，预热 10 次迭代），critic 1e-3 | 相同 |
| critic 预热 | 前 2 次迭代只更新 critic | 相同 |
| `vf_coef` | 0.5 | 相同 |
| 奖励缩放 | `reward_scale_running`：开 | 相同 |
| 图像增强 | 更新时开随机平移（与 M6 相同） | `augment: True` |
| 奖励 | 稀疏奖励：成功时为 1 并终止，其余为 0；25 s 超时为截断，仍然 bootstrap；分阶段奖励关闭 | 官方是 robomimic 的稀疏奖励 |

### 3.3 Critic（与官方不同，用户 2026-11-19 决定）

- 价值函数 $V(x, f)$ 的输入：
  - 30 维特权状态 $x$（与 M7 `rl/spec.py` 的 `PRIVILEGED_FIELDS` 相同）；
  - 冻结基础 BC 的 272 维特征 $f$（与 M7 `encode_actor_obs` 相同，来自**冻结的基础网络**，不是正在微调的 `actor_ft`）。
- 网络结构：MLP [256, 256, 256]，Mish 激活，残差结构，与官方 critic 的 MLP 部分相同。不含图像编码器。
- actor 不得接触特权状态。需要单元测试检查。

### 3.4 种子与选模（与 M7 相同）

- 训练种子 ≥ 10000，验证种子 20000–20015，评测种子 0–49，三者互不相交。
- 每次迭代结束后验证一次：16 局，η = 0，初始噪声按种子固定。验证录像另开一个环境（种子 20000），不计入成功数。
- 选模规则 `best_val`：验证成功数**严格大于**此前最好值才替换。同时保存 `ckpt_last`。评测种子不参与选模。

## 4. 交付内容

### 4.1 代码

- `src/aloha_pp/dppo/`（新包，名字可调整）：
  - 去噪链 PPO 策略：采样、对数概率、PPO 损失；
  - critic；
  - rollout 缓冲与 GAE；
  - 训练器；
  - checkpoint 的保存与续训；
  - `THIRD_PARTY_LICENSES`。
- `configs/m9_dppo.yaml`：第 3 节的全部超参数，每项注明是否与官方相同。
- 脚本：
  - `scripts/m9_train.py`：参数含 `--smoke`、`--benchmark`、`--resume`、`--out`、`--live-viewer`（默认关闭）；
  - `scripts/m9_eval.py`：方法 `dppo`，噪声取 `seeded` 或 `zero`，`--videos none|all` 默认 none，`--seeds`，`--out`；评测日志格式与 M7 的 `aloha_pp.eval.v1` 相同；
  - `scripts/m9_compare.py`：读取 M7 与 M9 的 `eval.json` / `episodes.csv`，输出对比表（含 Wilson 区间），以及配对的精确 McNemar 检验，至少包括 `bc_seeded`→`dppo_*`、`bc_zero`→`dppo_*`、`dsrl_best`→`dppo_*`。可以调用 `rl/compare.py` 的现有函数，但不得修改它。
- `tests/test_m9_dppo.py`，至少覆盖：
  1. **η = 0 时与 M6 一致**：`actor_ft` 未训练时，η = 0 的采样在相同初始噪声下与 M6 `sample(obs, noise)` 一致（同一进程内逐位相同；做不到逐位相同时给出原因和容差）；
  2. **对数概率正确**：单步对数概率与手算的高斯分布一致，标准差下限生效；
  3. **PPO 损失方向正确**：优势为正的去噪动作，更新后对数概率上升；
  4. **梯度隔离**：critic 的梯度不传进 `actor_ft` 和冻结的基础网络；基础网络参数在训练中保持不变；
  5. **actor 不读特权状态**；
  6. **种子集合两两不相交**；
  7. **终止与截断**：成功为终止，不 bootstrap；超时为截断，要 bootstrap。

### 4.2 运行（Cursor 只做几分钟内能完成的任务）

1. `pytest tests -q -p no:cacheprovider` 全部通过：原有 117 项不变，加上新增测试。
2. **冒烟训练**：`--smoke --out outputs/m9/smoke/train`，几分钟内完成。日志里能看到 actor 损失、critic 损失、KL、裁剪比例、η、验证成功数、计时。
3. **真实循环基准**：`--benchmark --out outputs/m9/benchmark`。
   - 用正式超参数跑 1–2 次完整迭代（如果单次迭代超过 15 分钟，就跑缩短的迭代再按比例外推），计时分项为：采样、编码、对数概率、更新、验证、I/O。
   - 外推 2×10⁶ 个转移的总时长。
   - 对比 8 个和 16 个并行环境的吞吐，选定 `n_envs`。
4. **回归**：
   - `scripts/m7_eval.py --method bc_zero` 用种子 0–2、`--videos none`，结果与 `outputs/m7/eval_bc_zero/episodes.csv` 的前 3 行逐项一致，写到 `outputs/m9/regression/`；
   - 跑 `check_env.py`；
   - 跑 `m1_grasp_smoke.py`，输出 `GRASP_OK`。
5. **冒烟评测**：`m9_eval.py` 用冒烟 checkpoint 跑种子 0–2，然后用 `m9_compare.py` 生成一份对比表，确认流程走通。

### 4.3 预算规则（事先声明）

- 基准外推不超过 24 h：正式训练用 2×10⁶ 个转移。
- 外推超过 24 h：总转移数取“24 h 内能完成的数量”，向下取整到整次迭代。只减总步数，不减 `update_epochs`、batch 和网络。交付中写明计算过程，由协调者确认后再交给用户启动。

### 4.4 交付记录 `tasks/M9_delivery_r1.md`

必须包括：

- 改动文件清单（应全部是新文件）；
- 运行过的命令（注明 conda 环境）及输出摘要；
- 与官方实现的逐条差异；
- 基准计时表与外推结果；
- 正式训练命令、四条正式评测命令（`dppo_best` 与 `dppo_last`，各取 seeded 和 zero 两种噪声）以及对比命令；
- 未完成或存疑的点；
- 上下文用量（状态栏百分比）。

## 5. 验收标准

1. 第 4.2 节的 1–5 项全部完成，测试全部通过，回归一致。
2. 新代码不修改任何现有文件。`git status` 只出现新文件，以及 `outputs/m9/` 下的产物（被忽略）。
3. 冒烟训练中所有数值有限，KL 与裁剪比例的量级合理；验证、录像与统计相互分离。
4. 交付记录完整，能直接据此启动正式训练。

## 6. 之后的流程（不属于本次派发）

1. Claude 第一轮审查，Codex 复核，用户审批；如需修改，交 Cursor 修改；Claude 验收；git 提交（不 push）。
2. 用户启动正式训练（预计 ≤ 24 h）和四条正式评测。
3. Claude 汇总对比，把 DPPO 一章补进本地 `report/report.md`。
4. M10 是否做 DSRL-NA，届时再与用户商定。
