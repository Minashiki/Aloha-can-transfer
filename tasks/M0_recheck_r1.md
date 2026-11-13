# M0 复核记录 r1（Codex）

复核日期：2026-11-13。只写本报告；未安装包、创建环境、修改配置或执行 git 写操作。所有 Python 验证均在 `conda run -n AIControl` 中进行。假设：以当前文件与现有 AIControl 为复核对象，不把交付者的历史行为自述当作完整审计证据。

## 一、逐条回应

### 1. 合规检查（对应 Claude 表格六项）

| 项目 | 判断 | 独立依据与边界 |
|---|---|---|
| PLAN、WORKFLOW 未改 | 部分同意 | 当前 `git ls-files docs/PLAN.md docs/WORKFLOW.md` 无输出，两文件被忽略；git status 不能证明其未变。Claude 声称做过派发前哈希对比，但本次未获得该哈希基准，不能独立重证；也没有证据认定 Cursor 修改了它们。 |
| aloha、.gitignore 未改 | 同意 | `git diff HEAD -- .gitignore aloha` 无输出，`git status --short --untracked-files=all` 只有未跟踪交付物与审查文件。F1 的 data/ 规则也存在于 `git show HEAD:.gitignore`。 |
| 没有 commit、作者配置未变 | 部分同意 | `git log -1 --oneline` 为 `c3e6b77 .gitignore：排除本地规划文档 PLAN.md、WORKFLOW.md`，支持未新增提交。单次读取 user.email 不能证明配置“未变”，本次按约束未执行 git config，也不对历史配置作保证。 |
| 只在 AIControl 操作 | 部分同意 | 交付记录命令均注明环境，本次解释器为 `/home/minashiki/anaconda3/envs/AIControl/bin/python`，torch、mujoco 从该环境导入；历史上是否从未运行其他解释器无法由记录彻底证明。 |
| 没有安装四类禁装仓库 | 同意 | 不仅交付清单未列出，本次 importlib.metadata 扫描已安装分发名与 dppo/dsrl/robosuite/robomimic 交集也是 `[]`。这支持当前环境结论，不等同于完整历史取证。 |
| 交付记录包含规定内容 | 同意 | `tasks/M0_delivery_r1.md` 包含新增清单、安装与 editable 安装结果、自检完整输出、pytest、torch 选择理由及存疑项。 |

### 2. 复跑验收命令：同意通过的结果

为遵守本次“唯一项目写入为报告”的约束，没有直接运行会覆盖 outputs 图片的 CLI，也没有生成 pytest 缓存。运行原函数时仅在内存将 IMAGE_PATH 改到 /tmp，未改变任何检查逻辑：

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -c 'import runpy,tempfile,pathlib; ns=runpy.run_path("scripts/check_env.py"); ns["check_render"].__globals__["IMAGE_PATH"]=pathlib.Path(tempfile.mkdtemp(prefix="m0-recheck-",dir="/tmp"))/"m0_wrist_cam.png"; raise SystemExit(ns["main"]())'
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -m pytest tests -q -p no:cacheprovider
```

两条退出码均为 0。自检完整输出：

```text
python: 3.12.14
torch: 2.15.0+cu132
torch.version.cuda: 13.2
mujoco: 3.17.0
numpy: 2.5.4
[PASS] cuda_available: True
[PASS] sm_120: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
[PASS] gpu_matmul: shape=2048x2048 max_abs_err=2.289e-04 rtol=0.01 atol=0.01
[PASS] mujoco_load_step: neq=1 steps=20
[PASS] wrist_cam: shape=(96, 96, 3) dtype=uint8 min=0 max=45 saved=/tmp/m0-recheck-ah46_2l3/m0_wrist_cam.png
ALL CHECKS PASSED
```

pytest：`4 passed in 0.90s`。`tests/test_smoke.py:20–46` 确实覆盖 CUDA 可用、sm_120、模型步进与 EGL 相机，GPU 不可用时断言失败，无 skip。

骨架独立核查：`pyproject.toml:1–12` 使用 setuptools 与 src 布局，包名 aloha_pp；当前 `import aloha_pp` 指向项目 src。九个子包的 `__init__.py` 均为 0 字节，顶层文件只有包说明；configs、assets/scene 的 .gitkeep 均存在。不声明运行依赖、以 environment.yml 配环境再 editable 安装，符合当前任务，不建议额外改造。未重跑禁止的 pip install。

### 3. 补充 GPU 验证：部分同意

Claude 的 cuDNN、bf16、torch.compile 输出是有价值的兼容性证据，但本次未复跑这些超出 M0 的验证。当前 GPU GEMM 已重证可用；不能把有限测试推广为所有未来工作负载均兼容，也不能仅凭某个依赖的版本号较大认定当前环境失效。

### 4. F1【高】：同意

`git check-ignore -v src/aloha_pp/data/__init__.py outputs/m0_wrist_cam.png` 输出：

```text
.gitignore:20:data/ src/aloha_pp/data/__init__.py
.gitignore:37:outputs/ outputs/m0_wrist_cam.png
```

当前文件存在但不出现在 git 的未跟踪文件清单；正常收集提交会漏掉要求交付的 data 子包。此问题源于既有规则，不能算 Cursor 擅自改坏配置；仍是交付入库前必须解决的问题。同意根目录产物规则加 `/` 的方向，但本轮不能修改，也不应绕过禁改约束；由协调者明确后续修改范围。验收除原建议外，应确认 data/__init__.py 实际进入待提交文件清单，而非只验证本地可 import。

### 5. F2【中】：部分同意

本次 `conda run -n AIControl python -B -c 'import importlib.util; print(importlib.util.find_spec("torchvision"))'` 输出 `None`，缺包事实成立。但 dispatch 并未要求 torchvision，PLAN 第 10 节还明确 M5–M7 为可裁剪加分项，因此不能据此要求 M0 立即加包或增添 GPU 测试。

反对将其作为当前中等级别缺陷或验收前待修项。将来实际选择 torchvision 编码器时再配对安装与验证即可；版本升级风险是将来依赖管理问题，目前没有证据证明会发生。

### 6. F3【低】：反驳建议修改的依据

同意“没有完整重建环境实测”这一验证边界；反驳“defaults 中这些版本可能不存在，所以建议移到 pip”的推测性修复。本次读取 AIControl 的 conda-meta，得到以下已安装包来源：

```text
python-3.12.14-hc5f7cf0_0.conda
setuptools-83.0.0-py312h06a4308_0.conda
wheel-0.47.0-py312h06a4308_0.conda
packaging-26.3-py312h06a4308_0.conda
```

四者 URL 均为 `https://repo.anaconda.com/pkgs/main/linux-64/` 加上述文件名，channel 均为同一 defaults main 地址；与 `environment.yml:8–12` 一致。核查命令：

```sh
PYTHONDONTWRITEBYTECODE=1 conda run -n AIControl python -B -c 'import sys,json,pathlib; root=pathlib.Path(sys.prefix)/"conda-meta"; [print(p.name,json.loads(p.read_text()).get("url"),json.loads(p.read_text()).get("channel")) for name in ("python","setuptools","wheel","packaging") for p in root.glob(name+"-*.json")]'
```

本地来源记录不能保证未来远端可下载或全量依赖求解成功，但已足以反驳包版本没有来源依据。保留当前版本钉死更符合最小改动，不应为未证实的问题移动包管理归属。禁止新建环境，故完整重建仍未验证；本报告不宣称复现验收已经完成。

### 7. F4【信息】：同意保持 3.x、不处理

`environment.yml` 的 pip 段钉死 mujoco 3.17.0，运行时也为该版本；dispatch 明确要求最新 3.x，因此不能因 4.x 发布要求升级。Claude 所述“当前 PyPI 最新 4.0.1”本次不作独立背书，动态版本应以[官方包元数据](https://pypi.org/pypi/mujoco/json)为准。尝试用 AIControl 的 urllib 读取 [PyTorch 官方选择器脚本](https://pytorch.org/assets/quick-start-module.js) 得到 HTTP 403，后续同条命令的 PyPI 查询未执行；所以本次也未独立重证“最新稳定 torch / 最新 3.x”的在线版本结论，已安装版本和可运行性则已重证。

### 8. Cursor 对话状态：部分同意其证据边界

本次未访问 Cursor GUI，无法独立判断任务计数、上下文用量或配置。认可“无法读取则不编造用量”；连续三次换对话属于协调工作规则，不属于 M0 代码验收项。

### 9. 总结论：部分同意

同意当前环境功能与骨架主体通过、F1 必须在入库前解决、F4 不需改动。不同意把 F2、F3 当作当前应修缺陷：前者超出 M0 且未来里程碑可选，后者已有 defaults 安装来源证据，不支持推测性修改。应称“主体功能通过，入库完整性待 F1 解决”，不宜直接宣称交付完全就绪；最新版本在线核验、完整环境重建及历史行为仍有上述证据边界。

## 二、新增问题

无。本次没有发现独立于 F1、需要新增修复项的可证实缺陷；未重复把 F1 导致测试未发现漏交 data 子包记为另一条问题。

## 三、总结（计数口径）

以任务要求的六条合规检查、F1–F4、总评这 11 个判断单元计数：**同意 5 条、反驳 1 条、部分同意 5 条；新增 0 条**。复跑与附加验证、Cursor 状态的回应已另列，不混入该计数。

- 同意：合规第 2/5/6 项，F1、F4。
- 分歧：合规第 1/3/4 项证据不足以证明全部历史行为；F2 不应作为 M0 待修项；F3 推测性修复缺乏依据；总评只接受主体通过、F1 待解决。
- 新增：无。唯一写入的项目文件是本报告，未改变交付实现。
