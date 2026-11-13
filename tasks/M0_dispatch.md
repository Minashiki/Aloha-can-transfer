# M0 派发：环境配置与仓库骨架

项目根目录：`/home/minashiki/AIControl/FinalHomework`（git 仓库，分支 `master`）。
背景：ALOHA 左臂（vx300s）MuJoCo 仿真抓取-搬运-放置项目。本里程碑只做环境与骨架，不实现任何算法。
可以阅读 `docs/PLAN.md`（第 3 节、第 10 节 M0）了解背景，但**不得修改** `docs/PLAN.md`、`docs/WORKFLOW.md`。

## 硬性约束（违反任一条即判为不合格）

1. **Python 环境**：所有安装、运行、测试只能在 conda 环境 `AIControl`（`/home/minashiki/anaconda3/envs/AIControl`，Python 3.12）中进行。用 `conda run -n AIControl ...` 或先 `conda activate AIControl`。禁止使用系统 Python、base 环境，禁止新建其他环境。交付记录里列出的每条命令都要能看出用的是这个环境。
2. **不得修改**：`docs/PLAN.md`、`docs/WORKFLOW.md`、`aloha/` 下的任何文件、`.gitignore`。
3. **git**：不得执行 `git commit` / `git push` / `git config`，不得修改作者信息，不得添加任何 co-author / contributor。提交由协调者完成。
4. 不得在终端直接启动 `codex`、`agent`、`cursor-agent` 等其他 agent。
5. 不得安装 DPPO / DSRL / robosuite / robomimic 的官方仓库作为依赖包。

## 要做的事

### 1. 安装依赖（在 `AIControl` 中）

- **PyTorch**：必须是 CUDA 版，且为与本机显卡和驱动兼容的**最新稳定版**。本机：RTX 5080（compute capability 12.0 / sm_120），驱动 595.99.02，驱动支持的最高 CUDA 为 13.2。请查 PyTorch 官方安装页（https://pytorch.org/get-started/locally/）确定版本与 CUDA wheel（所选 CUDA 版本不得高于 13.2），不要用 CPU 版，不要固定在旧版本。
- **其他**：mujoco（最新 3.x）、numpy、scipy、h5py、matplotlib、imageio + imageio-ffmpeg、pyyaml、pytest、tqdm。如有其他确需的包，在交付记录中说明理由。
- 导出 `environment.yml` 到项目根目录，关键包（python、torch、mujoco、numpy 等）写明确切版本，可以复现。

### 2. 仓库骨架

```
pyproject.toml            # 包名 aloha_pp，src 布局，可 pip install -e .
environment.yml
src/aloha_pp/__init__.py
src/aloha_pp/{sim,kin,dyn,control,planning,data,policy,rl,utils}/__init__.py
configs/                  # 放 .gitkeep
scripts/check_env.py      # 环境自检脚本，见下
tests/test_smoke.py       # pytest 冒烟测试，见下
assets/scene/             # 放 .gitkeep（场景资产在 M1 加入）
```

- 在 `AIControl` 中执行 `pip install -e .`，确认 `import aloha_pp` 可用。
- 子包里只放空的 `__init__.py`，不要写任何业务代码。

### 3. 环境自检脚本 `scripts/check_env.py`

运行 `conda run -n AIControl python scripts/check_env.py`，逐项打印结果，任何一项失败时退出码非 0：

1. 打印 python、torch、torch.version.cuda、mujoco、numpy 版本。
2. `torch.cuda.is_available()` 为 True。
3. `torch.cuda.get_arch_list()` 包含 `sm_120`。
4. 在 GPU 上做一次矩阵乘法（例如 2048×2048），与 CPU 结果比对误差在合理范围内。
5. 用 mujoco 加载 `aloha/aloha.xml`，确认 `model.neq == 1`，并 `mj_step` 若干步不报错。
6. 在 `MUJOCO_GL=egl` 下离屏渲染相机 `wrist_cam_left`，96×96，得到非空 RGB 图像（像素不全相同）；把图片存到 `outputs/m0_wrist_cam.png`（`outputs/` 已被 .gitignore 排除）。

### 4. pytest 冒烟测试 `tests/test_smoke.py`

覆盖第 3 节中的 2、3、5、6 项（GPU 相关测试在无 CUDA 时应失败而不是跳过）。运行 `conda run -n AIControl python -m pytest tests -q` 全部通过。

## 交付物

1. 上述文件。
2. 交付记录 `tasks/M0_delivery_r1.md`，必须包含：
   - **改动文件清单**（新增 / 修改）；
   - **运行过的命令及其输出摘要**（至少包括：安装命令、`check_env.py` 完整输出、pytest 输出、`pip install -e .` 结果；带上所用环境）；
   - **最终确定的 torch 版本与 CUDA wheel，以及选择依据**（官方页面信息）；
   - **未完成或存疑的点**。

## 完成标准

- `conda run -n AIControl python scripts/check_env.py` 退出码为 0，全部检查项通过。
- `conda run -n AIControl python -m pytest tests -q` 全部通过。
- `git status` 中只出现本 spec 允许的新增文件，以及被 .gitignore 排除的产物；没有改动禁止修改的文件。
- 任何一项做不到时，不要绕过或降低标准（例如改装 CPU 版 torch、跳过 GPU 测试），在交付记录里如实写明原因，作为任务失败上报。
