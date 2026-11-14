# M7 第一轮审查复核（Codex）

结论：**部分同意“修改后再训练”**。未发现 SAC 公式方向错误、真实训练路径的物体真值泄漏或 M7 环境评测种子进入训练/选模；CPU bandit 已实证 actor 可以学习。R1、R3、R4 的方向合理，R2 应作为本机观测而非普适确定性保证，R5 的 Wilson 口径需避免被误读为显著性检验。审查遗漏了 `videos=none` 的库实现缺陷、吞吐估计漏算实际特征提取，以及初期熵量级估算错误；正式训练前建议补一个实际采集+更新的短 benchmark，并将验证统计和视频执行隔离。以下意见供 Claude 独立决策。

## 范围、命令与限制

读了派发、两份交接、交付、审查、`git diff 006a65b`、新增 rl/脚本/配置/测试、官方 commit `8d21b9cf55459f022c0e769bca1343edd9d30e5a` 的配置及 train/utils/env_utils，以及 M7 benchmark、smoke 日志/selected 和 M6 正式 eval。HEAD 为 `006a65bcf52e2b11011f26cc0159d7377df8bbd0`，暂存区为空；tracked diff 是三个文件 49 行新增、2 行删除。只写本报告；临时文件均在 `/tmp/m7-recheck-codex/`，未运行训练/评测入口或写 outputs。没有安装包，没有 git 写操作，没有派生 agent。

实际执行的 Python 命令（均设 `PYTHONDONTWRITEBYTECODE=1`，后面四项另设 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`）：

```bash
conda run -n AIControl python -m pytest tests/test_m7_dsrl.py -q -p no:cacheprovider --basetemp /tmp/m7-recheck-codex/pytest
conda run -n AIControl python /tmp/m7-recheck-codex/probe.py
conda run -n AIControl python -m pytest tests -q -p no:cacheprovider --basetemp /tmp/m7-recheck-codex/allpytest
conda run -n AIControl python /tmp/m7-recheck-codex/closed.py 0 # 独立执行两次，输出 closed0a/b.json
conda run -n AIControl python /tmp/m7-recheck-codex/closed.py 1 # 独立执行两次，输出 closed1a/b.json
```

只读 shell 检查还包括 `git status --short`、`git diff 006a65b`、`git diff --cached --stat`、`git rev-parse HEAD`、`git -C third_party/dsrl rev-parse HEAD`，以及 cat/rg/nl。Orca 技能读取及完成通知使用本会话指定 CLI。这些不是 Python/仿真运行。

M7 测试 **14 passed (18.80 s)**；全套 **108 passed, 3 failed (47.20 s)**，失败为 CUDA 确定性、CUDA available、M6 图像重放（agentview 最大差129）。与交接声明的无 GPU/图像重放限制吻合，不能据此判 M7 回归。CUDA 推理/EGL 硬件路径的结论**沙箱无法复现**。CPU 推理和渲染可用，但不等同于作者硬件。

为对照 SB3，使用 curl 只下载官方 master 的 `common/distributions.py` 和 `sac/sac.py` 到 `/tmp`，未安装/执行源码：[分布实现](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py)、[SAC 实现](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py)。这是当前源码对照，不宣称锁定了作者使用的 SB3 版本。

## 对第 1 节逐项意见

|条目|意见|可核对依据及边界|
|---|---|---|
|PLAN/WORKFLOW 哈希|部分同意|接受 Claude 派发前快照记录；没有历史快照可独立重证，不能把当前哈希当作“未改”的证明。|
|outputs/m3–m6 未写|部分同意|同上，接受其590文件记录；本轮没有写入这些目录。|
|tracked 范围|同意|上述 git diff；sample 函数未改，rollout 仅显式传 noise 和增加回调。|
|不装包|部分同意|接受安装历史记录，本轮未安装；无法反向证明 Cursor 操作历史。|
|官方仓库非依赖|同意|src 的 import 无 third_party/dsrl；官方目录 commit 如上。|
|git|同意|HEAD 和空暂存区独立核对。|
|111 测试通过|部分同意|接受本机结果；本沙箱108+3环境失败，M7新增14项全通过；CUDA RNG测试无GPU时直接return，并非覆盖CUDA恢复路径。|
|其余回归|部分同意|接受交付记录，未重跑。|
|Cursor上下文|部分同意|属于界面/历史状态，未独立验证；不影响代码正确性。|

## 对第 2 节逐项意见

|条目|意见|依据|
|---|---|---|
|tanh潜动作及幅度|同意|nets.py `sample/deterministic`，128维 reshape；官方 yaml action_magnitude=1.5。|
|actor不含物体真值|同意（实际路径）|policy.py:134–160 特征；chunk_env.py `stack_history` 只保留两相机/TCP/开度；trainer的feat仅来自此obs，replay.add/sample各字段一一对应；actor.sample仅收feat。14项测试中的特权键干预通过。API存在state备用分支，见遗漏N4。|
|非对称critic|同意|spec.py PRIVILEGED_FIELDS 30维、nets.py TwinCritic._cat 拼 priv/feat/action，430维。无字段错位。|
|超参数出处|部分同意|数字与官方 yaml/train 一致；但初始化、critic action坐标及更新细节并非逐项相同，见下节。|
|块步/terminal/truncated|同意|chunk_env.step成功时立即终止，超时truncated；reward.td_target仅用terminated乘mask；成功/超时、TD测试通过。|
|next_obs未覆盖|同意|trainer.train先算next_feat并replay.add，之后vec.reset再替换out；Replay.add逐行赋值拷贝而非保存引用。|
|训练评测执行一致|同意（代码）|两边ChunkExecutor、to_env_action、沉降后reset、历史更新、成功即停；M7块执行/预览测试通过。|
|阶段奖励|同意|reward StageTracker paid与disabled分支；相应测试通过。|
|种子两两分离|同意（M7）|spec.SeedStream跳过banned；trainer在验证前也检查evaluation交集；take(10020)的测试跨过20000区间通过。0网络初始化不是环境seed0；M6冻结权重已使用0–3验证的历史局限仍存在，不能宣称整个BC来源从未看过评测种子。|
|单/并行一致|同意（限定口径）|test_parallel_matches_one_env_on_the_same_seed_and_chunk本沙箱也通过；容差一致不等于所有GPU跨进程逐位一致。|
|正式≤24h|部分同意|benchmark.json原始时间和算术支持17h估计，但benchmark路径少做两次encode_actor_obs，且未计I/O/视频，不能作为验证通过的上界，见N2。|

## SAC专项对照和干预

- **tanh log概率正确，缩放口径应讲清**：nets.py:100–103 为 `Normal.log_prob(u)-log(1-tanh(u)^2+1e-6)`，求维度和，与 SB3 SquashedDiagGaussianDistribution 一致。它是归一化动作 `a=w/1.5` 的密度，不是物理w密度；后者还需减 `128 log(1.5)=51.89953`。官方collector先scale_action再写replay，actor也在归一化空间算熵，因此这里不减常数是合理的SB3口径，**不建议为了“修正Jacobian”孤立加上该项**，否则目标熵的实际目标改变。
- 当前critic直接吃w，官方吃a=w/1.5；前者数学上可重参数化，训练优化/初始化尺度不同，不能宣称逐位同算法。均值头0.01正交、log_std头正交随机权重也不同于默认SB3 Linear初始化；`log_std_init=0`只是偏置，并不保证每个观测log_std为0。需记录，未见足以判学习必失败的证据。
- **α方向正确**：loss为 `-log_alpha*(logp.detach()+target)`；logp<0时梯度下降使log_alpha下降，>0时上升。第375行起测试通过。目标熵0在128维意味着归一化动作联合微分熵目标0，即每维平均约0，不是把探索关掉；这是较28维更强的初期soft-value影响，不能仅凭冒烟loss上升断定错误。
- **双Q backup正确**：target.min_q减αlogp，`r+gamma*(1-terminated)*next_q`，actor最小化αlogp−minQ；target无梯度，Polyak正确。截断保留bootstrap、终止不bootstrap，与要求一致。
- **与SB3细节差异**：SB3 critic loss有0.5倍，本实现是两项MSE直接相加；Adam下不等于简单翻倍实际学习率，仍应记录。SB3先采当前actor动作并更新α，再critic、actor；这里先critic、再重采当前actor动作、actor、α（下次更新才用新α），不构成方向错误，但不是逐步完全复刻。critic在actor.backward时也积累梯度却不step，下次critic.zero_grad会清掉，正确但浪费计算。
- **热身正确**：noise.clip_standard_normal先标准正态再clip±1.5，官方utils.py:164–179相同，区别是官方写归一化action。并不是clip到±1后再乘1.5。
- **actor干预成功**：`probe.py` CPU单线程，固定seed0，2维零特征/1维特权、1维w，2048条均匀动作，reward=(w>0)、全部terminated；两层32宽，lr=.003，500次batch128 SAC。确定性w从 **0.0→0.8777487874**，α=0.3499750。这证明当前实现存在正确学习方向，不证明128维长任务必然收敛。

## 对第 3 节 R1–R5 和第 4 节意见

**R1：同意。** 未记录噪声序列/RNG状态的旧noise=None评测不能要求逐位回放；sample原分支未动，不能把历史数值对不上当作回归。支持替代成显式噪声回归。固定CPU线程/后端/版本并记录元数据，比“全平台逐位保证”准确。

**R2：部分同意。** 同意将录像从统计评测隔离，接受Claude对其本机干预的记录，但不能从有限种子和有限拍数归纳任意硬件/全部50种子必定逐位一致。静态96图像不变也不能唯一定位“录像影响观测”的根因，尚有物理状态/渲染副作用等路径；审查保留机制未知是正确的。M6 0.62仅作随机抽样参考、bc_zero/seeded作为新主对照合理；M6 eval.json证实种子0–3有视频。

R2-1增加CLI还**不够**，必须修 `evaluate_latent` 的want条件，见N1；R2-3应覆盖库函数确实不调用录像渲染，也最好加DSRL deterministic，不能仅检查CLI接收了none。R2-4接受验证首条录像存在噪声的方案技术上可运行，但选模可消除这项污染：先无录像验证所有种子、另跑同一种子录像，仅统计前者（满足派发“至少一条视频”）。这是建议增强，不把明确允许的现方案判硬约束失败。

**R3：同意（必要作诊断，非修复收敛的充分条件）。** trainer严格大于留旧；正式第4000/8000/12000步都仍热身，学习第12008步才开始，早期权重参与竞争；smoke selected.json独立证明step140的1.0被保留。固定16种子反复500次可能对验证集过拟合，即使彻底确定性也存在，不宜全归于录像随机性。同时报best/last并提前指定selected为主结果合理；不能事后依0–49较好者改主checkpoint。两者都失败不能由报告两者解决，应先看短训练曲线和实际更新。

**R4：同意。** 原测试只测critic/alpha/target，没有actor方向断言；新增bandit便宜且有效，本轮已证成功。建议锁定seed和线程，对固定可实现的小问题断言而非要求任意随机初始化必然学会；它不能替代128维非终止backup的数值监测。

**R5：部分同意。** 噪声来源、录像、配对种子/共同checkpoint标注必要；不把旧M6作为严格可复现配对基线合理。Wilson区间重叠可以描述，**不能用作“无提升/显著提升”的判据**；逐种子配对结果至少列二者不一致计数（提升/退化），需要统计结论时用配对方法，不能用两个独立Wilson区间代替。

第4节逐项：

1. **同意** UTD20/8=2.5 vs官方20/4=5是明确预算取舍；不因降低样本量而判实现错误。
2. **部分同意** 目标熵0不是公式缺陷，但“初期约+180”的依据错用了未tanh高斯。probe在零特征/初始化log_std=0时测得128维tanh修正后mean logp=**−85.76001**（10000样本），不是−180；28维同条件应约−18.8，维数倍率仍约4.57。真实272维观测会令log_std_head输出不同，必须实测真实batch；冒烟actor loss约−103也更接近这个量级。不能以误算+180为critic几千loss的充分解释，建议日志增加mean logp、target Q和Q/reward统计。
3. **同意** 冒烟best未经训练，没有“学到”的证据；约17h是待修正的估计，正式运行由用户启动，本轮没跑。

## 审查遗漏问题清单

**N1（R2修复必需）：`videos='none'` 当前仍录像。** evaluate.py 的 `want = videos == 'all' or index == 0 or not (have_success and have_failure)` 完全未检查none；第一条永远True。CLI加选项而原样传库函数仍失败。建议 `videos=='all' or (videos=='auto' and (...))`，加none模式零录像渲染调用的测试，eval日志记录请求模式及实际录像种子。

**N2（正式前建议核实）：17h benchmark漏算采集路径的两次特征编码。** trainer.benchmark_collection仅policy.sample+vec.step；真实train在sample前encode_actor_obs、step后再encode_actor_obs，再做CPU拷贝/写replay。_validate也要额外encode。图像编码不是零成本，benchmark仅20 vec步且更新仅10次，也没包含定期checkpoint和视频编码。建议本机用正式8环境/128维/2048网络跑数百真实vec步的独立临时短训练，分解采集/更新/验证/I/O，复核24h预算，避免跑到截止才发现估算偏低；不要求现在跑17h。

**N3（第4节需纠正）：初期熵量级估算见上，且缺真实128维训练数值健康证据。** bandit和60次terminal测试均不覆盖长期非终止熵bootstrap；短训练至少记录mean logp、Q/target、alpha、actor均值幅度和训练后验证变化。它是风险验证缺口，不是已证的SAC bug，不建议未经实验改目标熵/奖励。

**N4（低优先级文档边界）：encode_actor_obs不只接受声明的四键组合。** policy._cond:185在有state但没有tcp_pose时使用任意state张量；故其docstring“keys other than四键ignored”和spec.ACTOR_OBS_KEYS不是严格API安全边界。真实chunk_env.stack_history提供tcp_pose且不提供state，因此**没有发现真实泄漏**；可以收紧新encode方法的输入契约或修文档，保持M6原_cond不变。回放字段没有错位。

**N5（续训行为需说明/保护）：resume模型配置和当前cfg可能分叉。** checkpoint.load_training按checkpoint内config重建agent（γ/τ/lr/scale），scripts.m7_train继续用当前yaml cfg控制采样热身scale/UTD/新保存config；只核对BC哈希及特征维度。用户若续训改noise_scale或SAC配置，执行的actor/agent与记录新config可能不一致，再加载可改变确定性动作。默认原配置续训不触发，不是当前首次正式训练阻断项；建议校验不可变配置或明确恢复规则。save_replay=false恢复空池后只等到batch足够即恢复UTD20，不重新1501步热身，这一点交付应讲明。

**N6（选模污染建议）：验证录像直接计入success。** trainer._validate:124直接将record环境的terminated计入成功；一旦R2本机现象成立，best_val可能被该1/16的随机翻转影响；对最大值选模影响比“仅一条可以接受”更直接。建议统计无录像、录像另跑，并保留best/last提前声明；这是比仅标注噪声更充分的处理。

## R2 CPU 闭环实验补充结果（完成后补记）

`closed.py` 使用正式同一库的 ChunkEnv（包含两帧观测、preview、每2拍256录像渲染逻辑），加载 **M6 smoke_r2/ckpt_last.pt**、CPU单线程、torch seed123、环境seed1、显式全零w，运行直到500拍截断。分别record=False/True，各在两个独立conda进程执行；未编码/落盘视频，只触发与录像相同的渲染调用，避免视频文件干扰。每个块末对完整 `env.data.qpos.tobytes()` 做SHA256，共63个检查点（前62个8拍，最后4拍）。

|比较|不相同检查点|
|---|---|
|不录像0a vs 0b|0/63|
|录像1a vs 1b|0/63|
|不录像0a vs 录像1a|0/63|

**CPU结果未复现“录像破坏轨迹”**。支持显式噪声在此CPU样例的跨进程确定性，但不能否定GPU/EGL下作者的实验，也没有覆盖bc_seeded、正式checkpoint、全部种子或逐拍像素。故R2保持“部分同意”，支持统计与录像隔离作为控制变量，反对把录视频一定破坏复现当作已证明的跨后端性质。输出和完整可复用脚本在 `/tmp/m7-recheck-codex/closed.py`、`closed0a.json`、`closed0b.json`、`closed1a.json`、`closed1b.json`。

补充分析命令：`PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -c 'import json,pathlib; p=pathlib.Path("/tmp/m7-recheck-codex"); d={s:json.loads((p/("closed"+s+".json")).read_text()) for s in ("0a","0b","1a","1b")}; print({k:len(v) for k,v in d.items()}); print({x+" vs "+y:next((i*8+8 for i,(a,b) in enumerate(zip(d[x],d[y])) if a!=b),None) for x,y in (("0a","0b"),("1a","1b"),("0a","1a"))})'`。

最终 `git status --short` 相比本轮开始只新增本报告；已有三个tracked改动及其他未跟踪文件名单保持不变。没有声称以status反向证明所有ignored outputs历史未变；本轮执行的入口和临时路径如前可核对。
