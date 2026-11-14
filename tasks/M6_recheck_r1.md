# M6 复核 r1（Codex）

对象：`tasks/M6_review_r1.md`；另读dispatch、handover、delivery r1、`git diff 4d4f02b` 和未跟踪文件、冒烟产物与本地DPPO源码。仅写本报告；脚本/输出均在 `/tmp/m6-recheck-codex/`，无安装包、agent、git写操作，未运行会写项目outputs的main入口。

**结论：F1/F2应修改，但主体尚有审查遗漏的观测四元数预处理不一致（N1），建议正式训练前一起修复。** 扩散核心、UNet重排和块执行规则通过独立对照；F3的“≤1灰阶”仅能作为原机器记录，本次图像回放首帧差129，不能直接接受为跨环境保证。本次全测试 **90 passed、3 failed**，其中2项CUDA检测失败、1项图像重放失败。

## 可复现方法

全部命令通过AIControl环境执行，禁写pyc/pytest缓存，临时输出重定向到指定目录。主要命令：

```sh
conda run -n AIControl git diff 4d4f02b
conda run -n AIControl git -C third_party/dppo rev-parse HEAD
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 python /tmp/m6-recheck-codex/parity.py
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl python /tmp/m6-recheck-codex/probe.py
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl python /tmp/m6-recheck-codex/replay.py
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl python -m pytest tests -q -p no:cacheprovider --basetemp=/tmp/m6-recheck-codex/pytest
```

`parity.py` 从只读DPPO文件取AST定义，去除导入，在/tmp实验中以原生torch替代明确列举的einops重排，不安装或import第三方仓库。它独立比较源p_mean_var、噪声调度、UNet、SpatialEmb与随机平移；ViT仅放开源代码的math attention开关以便CPU对照。结果在parity.out；probe.out/render.prof是渲染构造次数及性能、四元数差异；replay.out是两条低维重放。没有重跑训练或正式评测，不把原日志当本次重新训练。

## 合规逐项

| 项 | 判断 | 依据与边界 |
|---|---|---|
| 受保护文件/数据集 | 部分同意 | tracked diff只有m4_run包装函数及policy/__init__；受保护tracked文件无变化。历史派发前哈希和未入git的docs不能由本次反向独立证明，接受Claude记录但不声称重测历史。 |
| outputs/m3–m5未写 | 部分同意 | 本次只读已有输出、没有调用生成main；派发前目录mtime清单无法由本次反向重证。 |
| 不装包/不依赖DPPO | 同意静态证据 | 提交代码不import dppo/einops/torchvision/hydra；本次未安装；本地DPPO HEAD确为cc7234ad7ff39a8f32de3af903606723a16f0648。历史未安装依据交付声明。 |
| 第三方许可证 | 同意 | 复制文件头给URL/commit/路径，THIRD_PARTY_LICENSES保留DPPO MIT及版权声明；未发现缺失的本次DPPO署名。 |
| git | 同意当前基线状态 | tracked diff与4d4f02b交付范围吻合；HEAD相同不能单独证明历史从未调用其他git写操作。 |
| 测试 | 部分同意 | 本次90通过/3失败，15.12s；CUDA deterministic测试和既有CUDA smoke均因CUDA不可用失败；图像重放测试agentview t=0 max delta129失败。不能复签“本环境93全过”。 |

## 交付主体逐项

1. **噪声接口：同意CPU验证，CUDA受限。** noise_shape=(16,8)，支持批量给定初始噪声；η=0，sample临时eval关闭增强、恢复模式与后端flags。原测试的CPU重复、不同步数形状、checkpoint往返通过；本次独立DDIM同噪声两次逐位一致，RNG状态不变。CUDA本次不可用，保留Claude历史结果，不声称跨CPU/CUDA输出逐位相同（要求本来也只是同设备同输入重复）。
2. **策略：部分同意。** 双相机共享ViT、分开的SpatialEmb、2帧历史、8维TCP+gripper状态、16×8输出、EMA .995和参数量1,392,848均确认。没有物体真值或关节角泄漏。归一化只拟合训练集正确；但离线/在线观测四元数处理不一致，见N1。
3. **执行：同意块规则与共享组装，保留reset细节。** ChunkExecutor.pop在k=7返回chunk[7],chunk[8]，新块每8拍替换，历史每执行一拍后更新为最近2个动作前观测，首帧重复填充；符合第8步预览第9步约定。m4包装转调c3.py逐项等价，M4/M5增益、IK、motor、重力补偿配置相同。共享构造函数不代表每条reset调用路径也逐位等价，见N3。
4. **训练子集：同意。** checkpoint中的30个source_id与交付列表相符；select_indices以seed0无放回抽取排序，其余90条验证。sequences.load_splits先取30条拟合state/action min/max，再归一化两组，未用验证集统计。正常数值维极差下限1e−6，无除零。尾部动作重复末动作、头部历史重复首观测，与执行时刻对齐，没有额外错位一拍。
5. **冒烟：同意已存日志，非本次重训。** train_log.jsonl记录.8492607→.3224253，val .9761691→.3709747，6 epoch；现存评测3条超时失败及训练验证1条失败均与审查相符。推理中位约.022s；eval首条最大.284486s而非训练验证的.0712s，两个入口不要混用峰值。

## DPPO一致性专项

- **β和训练损失：同意。** NumPy余弦公式、s=.008、clip β至[0,.999]、float32累乘与源相同，100步β逐位相同。q_sample=√ᾱ x0+√(1−ᾱ)ε，均匀随机t、ε预测MSE与源同义。
- **DDIM：同意。** leading索引是floor(100/S)倍数再反向，不是从99等距取端点；S=5时[80,60,40,20,0]，S=8时[84,72,60,48,36,24,12,0]，S=10时[90,80,70,60,50,40,30,20,10,0]。这看似未到99但与源一致，应保留。
- **公式与clip：同意当前配置等价。** x0=(xt−√(1−α)ε)/√α；先clip x0到[−1,1]再重新算ε，μ=√αprev x0+√(1−αprev)ε，末步αprev=1。本次逐步对照源p_mean_var，5/8/10步每一步最大差均0；最后输出abs max=1。省去sigma=0的附加随机抽样正确且不改变RNG。源代码可选eps_clip/final_action_clip本配置均None，当前裁剪已足够；不声称复制了未使用的所有可选功能。DPPO源码forward的非法deterministic关键字未复制是合理修正。
- **UNet/FiLM/图像重排：同意。** 将161个对应参数张量按形状复制，同输入下完整VisionUNet（上游骨干替换为相同本地ViT，以隔离卷积路径）在batch1和3上输出逐位相同，maxdiff=0。上/下采样次数、全局条件拼接顺序、scale*x+bias均正确；上采样的源is_last条件看似特殊但本实现保持了源行为，不宜自行“修正”。两路相机各自时间帧拼通道的permute正确。
- **SpatialEmb/增强：同意。** 相同权重/输入SpatialEmb逐位相同；同随机种子RandomShiftsAug逐位相同。只在training时增强是为满足sample确定性而作的必要有意变更，区别于源augment不检查training。
- **ViT：同意架构/数学等价，不能宣称与源逐位等价。** embed2卷积、patch展平、qkv布局、LayerNorm/残差/FFN与原一致。复制同权重后，允许上游math attention的CPU对照最大差8.34465e−7，非逐位相同；原源默认偏好flash/memory-efficient，本实现强制math，浮点路径变化合理且已说明。训练/采样确定性要求不等于与上游不同后端逐位相等。

## F1（中）：同意机制与修复方向，比例/加速倍数限定机器

`rollout.py:127`附近每控制拍先render256视频，随后env.step(render=True)内部get_obs渲染96×96双相机；`env.py:441`只持有一个renderer，尺寸不符就close并new。观测双相机同尺寸会共享一次创建，因此每拍确实new两次；写视频时才每隔round(20/fps)抽帧，浪费已发生。

本次直接patch计数真实Renderer构造，不改源码：初始reset(render=True)+10拍“256视频→96观测”共 **21次构造**。同状态同分辨率连续render两次像素最大差0。profile总23.54s，_get_renderer累计4.65s、构造4.49s；mjr_render累计18.14s。本机渲染开销构成与Claude的GPU测量不同，故“约75%创建/一定4–5倍加速”只能引用其机器的数据；本次仍独立确认反复创建这一性能bug。

同意按尺寸缓存并close全部实例，或rollout独持视频renderer；在渲染前按fps筛拍。无需把256视频降为96训练图像以换性能。修后比较同一环境/设备/seed/动作、相同视频采样时刻；像素在不同GL上下文可能改变，物理轨迹/统计也要核对。无视频路径本来不会跨尺寸，每拍耗时拆分有助于M7，合理。

正式总时间还遗漏每epoch验证损失遍历，见N2；仅修renderer后重新估计仍不应只计训练step和20次任务验证。

## F2（中）：同意

failure.py的success早返回后紧接ik_ok检查，rollout只累计一个全程布尔值，确会将任意一次目标/预览IK失败覆盖其他过程原因。移出主任务原因、保留IK诊断，两类分布并存合理且不改变物理成功标准，也不需要改M4/M5分类。

`env.step`的info.ik_success包含当前和未来目标IK；新增统计需注明“本拍任一个IK失败”的tick口径，不把逐物理步重复采样累计成25倍次数。最好同时保留current/preview失败计数；success episode也保留IK指标。主过程分类目前只观察第一次连续closed窗口，学习策略可能多次抓取/重试；调整顺序时应明确按哪次尝试/是否曾成功提起，不能仅删除ik_fail早返回就声称所有主原因可靠。

## F3（低，记录）：部分同意

接受同一观测张量+同噪声的确定性不等于整个闭环图像episode逐位复现；但不同机器上“图像最多差1”不是可依赖保证。本次原test_m6_replay在demo首帧agentview最大差129，明显超出1LSB；不能直接解释成同一EGL驱动特性。本次同场景连续两次同尺寸render差0，只说明本次短程重复稳定。

建议报告渲染器/驱动/设备信息并在生成数据所用GPU环境复测，将跨渲染后端像素差异与同后端重复波动区分；不要为了过本次复核而盲目把像素阈值放宽到129。本次独立无渲染两条低维回放结果见收尾；该结果可以验证控制器和物理配置，但不能代替图像一致性证明。

## F4（低）：同意

冒烟允许0成功；现存3条eval失败和1条训练中验证失败视频与说明相符。无成功episode便无成功视频，不要求伪造或改阈值；正式训练后存在成功时再按要求保存。

## N1（高）：离线训练和在线sample的观测四元数符号预处理不同

依据：`sequences.py:_pack_episode` 对tcp_pose四元数调用canonicalize_quaternions；`policy.py:148–162`直接cat原始tcp_pose与gripper并normalize，`rollout._stack_hist`也只转float32、不规范符号。

本次遍历120条低维数据，**34帧**的canonical化值与原存储值不同。例demo_0004 t=194，原始q=[−.49983937,.5005263,−.49961042,−.5000235]，训练变为其负值；将完全相同物理观测经在线_cond送入时仍用原q。checkpoint state归一化后两表示差为 **[−8.4605,3.8047,−8.4161,−3.8574]**（四元数4维），远非浮点舍入。这是可复现的训练/推理分布不一致，不能以q与−q物理等价免除网络输入差异。

复现：probe.py输出changed obs、quat example、cond raw match=True；后者确认真实在线_cond正是原始而非训练规范后的输入。

建议共享同一状态四元数规范函数，离线窗口与在线历史走同样规则；若使用跨帧连续性，必须明确episode初始符号、在线历史/重规划时的持续状态，不能一边按整episode连续一边每窗口重新起算。用同一存储观测窗口经训练路径和在线路径比较cond state逐项一致，再测等价q/−q输入。只修动作to_env_action不会修复观测问题；不需要改变环境本身的原始观测接口。

## N2（低）：正式训练总时长估计漏掉每epoch完整验证损失

`scripts/m6_train.py` 每epoch都调用_val_loss；val_every只控制闭环任务评测。默认val_batches=0，即每个epoch完整走22023个验证窗口，batch64约345批，而训练每epoch114批。冒烟只取2批验证，其实测训练step估计0.94h不能覆盖正式345批×2000 epoch的前向、数据读取、每epoch绘图与checkpoint等成本。

建议测一次默认全验证损失耗时，按“训练+val loss+落盘+周期闭环评测”更新估计；无需本次正式训练或改学习配置。否则F1改善后的总时长仍可能低估。若改为更低频验证loss，属于配置/行为选择，应写清。

## N3（低）：共享C3组装一致，但在线reset少了M5沉降后的controller.reset

M5生成器在can_pose reset后629–631行再次controller.reset，噪声路径702–704行也如此；M6 rollout_episode只调用正常env.reset，无此第二次reset。正常reset在沉降前重置controller；沉降后q与home_q的差本次seed3为 **2.175052e−12 rad**。因此“与M5完全相同”严格到controller初值的说法仍有例外；测试用state恢复路径会重置为已沉降q，无法覆盖正常random评测路径。

数值很小，本次不据此宣称影响成功率，也不把图像差129归于它。若目标是完全等价，建议仅在M6评测入口沉降后补与M5相同的controller reset，并对比controller/stream初值；不要改默认env.reset以免破坏M1–M4回归。可与修复后的M5同seed初始流程直接对照。

## 结论与复核边界

建议采纳F1/F2，F3记录应限定原硬件并解释本次像素失败，F4记录在案；正式训练前先修N1，N2/N3补足时间估计与严格环境等价性。CUDA确定性与图像回放在原GPU环境需补验，本次不把环境限制隐藏为全通过；网络数学/CPU确定性和物理低维一致性的证据分别陈述。

收尾证据：`replay.py` 对demo_0000、demo_0001均输出success=True，tcp_pose/gripper_opening/q/qvel逐控制拍float32比较最大差0。开始/结束status快照唯一差异为新增 `?? tasks/M6_recheck_r1.md`；其余项目状态行相同（status不覆盖被忽略产物内容，本次没有调用其写入入口）。
