# M5 复核 r1（Codex）

复核对象：`tasks/M5_review_r1.md`；背景为 dispatch、handover、delivery r1，基线 `fe540b49cb2291329422dbaac4498e1e3764b757`，另读未跟踪文件。假设现有输出对应该交付，不把现有文件存在当成历史执行过程的证明。

**结论：要求修改后重新生成合理，但 F1/F2/F3 的部分事实和归因必须修正。** 起步超速成立；M5 生成路径的控制器 home 初值已通过额外 reset 对齐，不能把其二次执行差异归因于 2.2e-12 的 home 差；warmstart 丢失与动作 float32 量化是本次实验直接支持的来源。outside 不能一概重命名为“未提起”。F4 的来源标签合理，但“人类拍数占比”需要明确定义，不能把合成与人类混合的拍简单解释为纯人类行为。

本次仅写本报告；验证脚本/输出均在 `/tmp/m5-recheck-codex/`，无新增包、agent、git 写操作，也未调用项目脚本的生成/分析/回放 main 入口。调用生成模块内的 retarget/rollout 只在内存执行，未写 outputs 或 datasets。

## 复现命令与证据文件

所有命令均以 AIControl 环境启动。关键复现：

```sh
conda run -n AIControl git diff fe540b4
conda run -n AIControl git rev-parse HEAD
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 python /tmp/m5-recheck-codex/probe.py
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 python /tmp/m5-recheck-codex/reproduce.py
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 python /tmp/m5-recheck-codex/extra.py
conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider --basetemp=/tmp/m5-recheck-codex/pytest
conda run -n AIControl sha256sum datasets/robomimic/can/ph/low_dim_v141.hdf5
```

脚本的输出同名 `.out`；outside 重跑逐条结果在 `outside.json`。probe 只读 hdf5 的 qvel、动作、属性，调用校验函数检查两个文件；reproduce 重建源 demo 3 的动作、从相同存储状态比较 float64/float32 执行，再重建43个 outside 的动作并无渲染执行；extra 分别恢复 time/warmstart，做单字段对照。全部环境变量禁写 pyc，pytest 缓存关闭、临时目录定向 /tmp。

## 合规逐项

| 项 | 判断 | 证据和边界 |
|---|---|---|
| 受保护文件 | 部分同意 | tracked diff 只有 `data/__init__.py` 和 `sim/env.py`；其余是新增文件。无受保护 tracked 文件差异；派发前 sha256 清单和不入 git 的 PLAN/WORKFLOW 历史无法独立重证，接受 Claude 历史记录但不冒充本次哈希验证。 |
| outputs/m3、m4 未被写 | 部分同意 | 审查报告给出历史清单哈希相同；本次无法反向重证派发前 mtime。回归产物确实在 outputs/m5/regression/m4_e1，本次没有写这些目录。 |
| git | 同意当前 HEAD | 当前 HEAD 为上述 fe540b4；无法仅从 HEAD 证明历史没有任何 git 写命令。 |
| 环境、不装包、禁止依赖 | 同意静态证据 | 所列运行有 AIControl 前缀，本次也全部如此；src/scripts/tests 无运行时 robosuite/robomimic import，frames 中只写来源说明。无证据表明安装过包；历史未安装只能依赖交付记录。 |
| 测试 | 部分同意 | 本次 **74 passed，1 failed in 12.51s**。唯一失败为 `tests/test_smoke.py:21` CUDA availability=False，不能签署当前环境75全过，也不能由此归咎M5。Claude的75 passed是其运行环境的记录。 |

## 已核实主体逐项

1. **原始数据：同意版本接受。** 文件48,293,744字节，sha256复算为 `6ce6e6c7f60b04e3fd180327de7c2dbe456373efe4b2682299cb2350fdc68cdd`，与交付一致；dispatch允许v1.4.1，无须重新下载。历史超时/404未重试；代码MIT并不自动证明数据许可证，交付对未找到独立数据许可证的表述应保留。
2. **坐标映射：同意。** `data/frames.py` 的目标格是 [.1975,.4025,.8]，R=I、t=[.1385,−.4825,−.76396]，行列式+1且映至fixed_1。坐标映射相关测试通过。两bin中心间距不同，无法用刚体变换同时对齐；实用轨迹使用物体相对偏移及合成搬运，不能声称整条人类世界轨迹仅做刚体变换。
3. **数据组成：同意113+8和校验，部分同意重放结论。** 两文件 `validate_dataset` 均返回空问题列表，主文件113条、备用8条。5/5成功是历史记录，本次针对source id3独立重放float32两次都成功且轨迹互相完全一致；其与存储轨迹存在量化动作造成的偏差，见F2。校验OK只覆盖当前校验器的结构/图像等规则，不证明物理轨迹符合0.5速度要求。
4. **M4 E1回归：同意现有产物比较。** 与outputs/m4/summary.csv按name对齐6个E1，75个共有字段的文本差异数为0；本次未重新跑回归。默认reset的tracked diff保留原沉降路径，相关测试通过。

## F1（高）：部分同意；问题成立，计数与根因描述需更正

**直接数据：** 对每条示范计算 `max(abs(obs/qvel)/[2,2,2,3,3,3])`，113条的最小/中位/最大为 **.734800、1.029525、1.029863**，峰值时刻全在观测索引1（首拍执行后）；其中112条归一化峰值在waist，1条在wrist_rotate。反例 `demo_0111`（source id197）：峰值比.988674，waist=1.771625、wrist_rotate=−2.966023 rad/s。因此“113条都等于2.06且都waist”不准确；**113条都超过.6**仍成立。备用8条的相同比值最小/中位/最大 .385705/.492513/.538560。

`demo_0025` 实为source id44，初动作相对初始TCP位移 [−.008449,.041057,−.001324] m；范数约4.2cm。obs/qvel[1]的waist=2.059727、wrist_rotate=3.003090 rad/s，与审查示例一致。

**起步问题独立重现：** source id3 的首动作距home TCP .0306315m；控制器首段preview的腰速度达到2 rad/s、waist与wrist_rotate的限幅标志均为True。不能由离线结点差分低于.5推断参考流内部峰值低于.5。

**归因修正：** `retarget.py:prepend_home` 确为线性插值，但 `retarget_demo` 是**先插home接近，再 slow_until**（约509–531行），所以接近段大部分事实上已参加时间缩放，并非整段仍固定0.5s。真正漏掉的是：prepend_home用 `(k+1)/n`，第一个样本就离开home；resample保留该首样本；`peak_joint_speed:354–357`只差分q样本间速度，不含home→首样本；lead_actions又从pos[1]开始命令，首步间隙进一步影响。位置与姿态都必须从真正home边界开始，不能只将线性改为五次却继续遗漏首边界。

**要求判断：** 同意五次接近、将整条实际动作（包含lead、末尾重复和运行时IK初值）送入preview参考流检查；严格≤.5且标志全0合理。最好同时计算未限幅参考峰值，避免用已被截断的速度误判缩放充分；规划IK种子/求解失败回退也需与实际控制器一致。闭环qvel不是参考速度，两者分开报告；.6作为需解释的诊断线合理，但并非原dispatch新增的控制器硬上限。20Hz obs/qvel只能证明采样时刻峰值，若称整段物理速度峰值须逐物理步记录，不能把控制拍采样最大值当连续全程最大值。

## F2（中）：部分同意；恢复不等价成立，所给M5根因不成立

### home差确实存在于一般reset对照，但M5已规避

正常random/seed3的controller.home_q与沉降后的q最大差 **2.175052e−12**，这个一般env行为确认。可是 `m5_generate.py:295–299` 在capture后明确又调用 `env.controller.reset(env.model,env.data)`，噪声路径341–343也一样；与恢复路径相比，M5实际起步的home_q和stream.q差均为 **0.0**。所以不能用“正常reset vs restore”的实验直接解释M5两轮生成的9条差异，或声称这些样本对1e−12扰动敏感。

### 其余不等价字段与干预结果

M5首轮额外controller.reset后 vs `_reset_from_state`：qpos/qvel/ctrl均相同，time为 .10000000000000007 vs 0；qacc最大差2.385210e−9，**qacc_warmstart最大差 .8823838**（并非相同）。qfrc_applied和xfrc_applied在此reset场景均相同为零；未声称这是任意中途状态的完整序列化。

M4 random seed3 medium固定同一实际动作及next_action序列，并模拟M5的额外controller.reset：

| 状态恢复干预 | 整条episode关节最大差(rad) |
|---|---:|
| 当前restore | 6.735579e−5 |
| 只补time | 6.735579e−5 |
| **只补qacc_warmstart** | **0.0** |
| 补warmstart+time | 0.0 |

本用例直接支持warmstart丢失导致差异；time仍应作为完整状态保存，但不能称它是本例关节偏差的原因。额外qacc差不必自行保存所有derived字段；优先保存积分/求解器必需状态与控制器状态，用完整episode逐位比较决定充分性，不能只比较reset后的qpos。

### 保存动作float32是独立且较大的不等价来源

`m5_generate.py:159` 和 `dataset.py:DatasetWriter.add_episode` 将实际执行的float64动作变float32；回放就执行不同命令。source id3重建float64动作与存储float32动作最大差 **2.877499e−8**。从**相同恢复状态**执行214拍：

| 对照 | 最大关节位置差(rad) |
|---|---:|
| float64执行 vs 存储obs/q | 5.943702e−8（float32观测量化量级） |
| float32执行 vs 存储obs/q | **5.545298e−4** |
| float64 vs float32执行 | 5.545320e−4 |
| 相同float32动作重复两次 | **0.0** |

三次均成功。该结果几乎精确解释交付的5.545e−4回放偏差，**不是1e−12 home偏移的证据**。建议从第一次闭环开始就执行最终存储精度的动作，或另存float64精确回放命令；obs/q为float32时，不要求“与存储obs逐位相等”，应将“两个float64仿真轨迹差=0”与“对float32存储观测的量化容差”分开验收。

同意一次执行记录图像或验证二次执行等价。还需统一末步next_action：首轮成功可能截断原序列，二次执行/存储末拍则重复自身；应验证终止拍的预告与首次执行一致。仅修capture_state不足以解决动作精度和截断语义。

**敏感比例要求部分反对：** 可报告9/122=7.38%的“两次执行成功翻转”，但在warmstart/动作精度等同时不等价时，不应称为“1e−12敏感比例”。若要该统计，需固定全部状态和动作，只注入明确量级扰动做受控对照；这属于新增诊断，不能由r1历史差异推定，也不是原dispatch硬验收。

## F3（中）：部分同意；需要细化分类，不能把43条全部判为未提起

CSV确实有43条outside，最终相对目标格xy误差 .152729–.240379m。这个值只说明离目标格远，不能证明未离地，也不能证明没有被提起再掉回左bin。

本次从原始源数据和当前配置重建全部43条的动作，在内存闭环执行，记录相对reset沉降后Can高度的全程及close期间最大抬升。计数与反例见文末完整复跑补充。已经确认source id1在close期间最大抬升 **32.9006mm**，最终仍分类outside；其最大相对初始xy位移90.2874mm，最终偏移51.3326mm。source id2则close期间仅抬升1.79318mm，确有未提起样本。这两类不能混称“未提起”。

`m5_generate.py:109–143` 的grasped仅需close时某次双指接触；lost用 `np.any`，确实不同于M4连续≥10个物理步。这里contacts是**每控制拍**记录，不是逐物理步：直接把阈值写成10条记录会变成0.5s，不能照搬数字10；应on_substep积累或按明确物理时长判定。close掩码还可能把开合斜坡纳入，与M4搬运状态掩码也不等价。

同意增加未提起类并放在outside前，但2cm是建议诊断阈值，应配置并明确参考高度/观察时间窗。短暂抬起后低高度掉落、成功抬起后落回起始bin，也要有独立依据，不要只看最终位置。grasp_z目前取所有双指接触时刻的高度中位数，未必等于抓取初始高度，建议一并核对，以免高度跌落分支失效。

同意补抓取几何偏移诊断（沿本项目实际指闭合方向、垂直方向、z），并与源数据的偏移及工具朝向对照；“圆柱”不使手指朝向和偏心量自动无关。不可达32条按段和位置诊断也合理，但无需保证某一个量完美分离所有成功/失败；给分布、重叠和例子即可，不应设无依据的可分性门槛。

## F4（中，说明）：部分同意；来源标签合理，人类比例须定义

源代码确认home接近、固定top-down姿态、放置下降/上撤合成；抓取前为人类相对位置、搬运为合成人工基线加人类残差。交付已说明理由，使用这种混合方法符合允许重定向的方向；增加逐拍来源标签、README和“部分重定向+合成放置”名称有助于M6解释数据，是合理的审查后新增要求，并非原dispatch明文缺字段。

建议标签至少区分人类位置偏移、carry残差、合成接近/放置/撤离、夹爪wait/ramp/dwell。现有3个phase不足：合成放置下降也标PHASE_CARRY，无法据它直接算人类比例；时间重采样混合边界、lead动作、success截断后标签还必须与最终存储动作逐拍对齐。

“来自人类的拍数”应改为明确的**含人类位置/残差信息拍数比例**，另外注明姿态和夹爪时序经过重写。carry_residual同时含人工基线，不能全算纯人类，也不应把插入的静止等待重复样本算新增人类信息。建议给出各段时长/比例分布，比一个未经定义的“人类占比”更可靠；无需为了标签重新恢复原人类姿态或放置。

## F5（低）：同意

主文件根属性git_commit确为fe540b4，缺git_dirty/源文件哈希；无法据该commit重建脏工作树生成代码。建议哈希清单包括实际依赖：除审查列举的data/noise/generate/m5配置外，至少记录此次改动的env.py，以及被import使用的m4_run.py和M1/M3/M4配置的内容哈希或快照。无需提交代码来“修正”属性；保留基线commit并增加dirty和内容指纹即可。

## F6（低）：同意

主文件长度min214、median486、mean481.9469、max1127拍，20Hz下中位24.3s、最长56.35s；备用min120、median135.5、max159拍。无需为长度本身改轨迹；补各段时长构成帮助M6设时限。

注意时间缩放发生在插入等待/夹爪斜坡**之后**，`resample_trajectory` 同时插值grip，因此重定向的等待、夹爪斜坡也随scale延长，不能按原配置固定等待时长做长度分解。要报告最终离散序列的时长，而不是仅将源人类段乘scale再加固定等待。

## N1（低）：返回的抓取/释放索引没有随prepend和重采样更新

`retarget.py` 从insert_gripper_waits取得grasp_i/release_i后，继续prepend_home、slow_until、lead_actions，最后548–549行仍原样返回两个索引。只要加home或scale不为1，它们就不是返回actions/pos序列的对应事件索引；目前生成器没用它们，尚不影响113条数据，但直接拿它们实现F3诊断、F4段标签会错位。

复现：同一输入分别开启/关闭home前缀，两个返回索引不随10拍前缀变化；scale>1时亦不重映射。建议随轨迹携带事件标签一起重采样和提前一拍映射，或明确命名为变换前索引；使用最终grip序列验证事件位置。无需重构无关模块。

## 收尾与验收建议

修复F1参考峰值判据；按实际M5路径修复状态恢复和最终动作精度；F3按可观测过程细分；F4–F6补来源/元数据/时长。然后重新生成并校验数据，在CUDA可用环境补齐全部测试及隔离输出回归。本次75全过未获复现，环境限制已单列；没有改控制器增益、速度上限或成功阈值。

### 完整复跑补充与状态核对

- F3：43条全部完成且本次仍为outside；**42条close期间最大抬升<20mm，1条为source id1、32.9006mm**。因此支持“绝大多数没有稳定提起”，反对“43条都没提起”；过20mm一次也不能自动证明持续可靠抓持，具体应结合接触/持续时长诊断。逐条数值见 `/tmp/m5-recheck-codex/outside.json`。
- N1：执行 `conda run -n AIControl env PYTHONDONTWRITEBYTECODE=1 python /tmp/m5-recheck-codex/index_probe.py`，source id3返回grasp_index=53、release_index=153、scale=1.1808673；最终grip首个闭合斜坡样本索引74，最终actions的索引73（提前一拍）。这是返回事件索引未更新的数值证据。
- 开始/结束 `git status --porcelain` 快照分别为 `/tmp/m5-recheck-codex/status.before` / `status.after`，唯一新增行为 `?? tasks/M5_recheck_r1.md`。状态列表不能证明忽略文件内容不变，但本次执行入口均不写项目输出，未运行任何生成/分析/回放main，也未写项目缓存。
