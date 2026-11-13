# M0 交付记录（r1）

环境：conda `AIControl`（`/home/minashiki/anaconda3/envs/AIControl`，Python 3.12.14）。下面每条命令都通过 `conda run -n AIControl` 执行。未执行 `git commit` / `git push` / `git config`。未改 `docs/PLAN.md`、`docs/WORKFLOW.md`、`aloha/`、`.gitignore`。

## 改动文件清单

新增：

- `pyproject.toml`
- `environment.yml`
- `src/aloha_pp/__init__.py`
- `src/aloha_pp/{sim,kin,dyn,control,planning,data,policy,rl,utils}/__init__.py`（空文件）
- `configs/.gitkeep`
- `assets/scene/.gitkeep`
- `scripts/check_env.py`
- `tests/test_smoke.py`
- `tasks/M0_delivery_r1.md`（本文件）

修改：无。

未纳入 git 的运行产物（已被 `.gitignore` 排除）：`outputs/m0_wrist_cam.png`、`src/aloha_pp.egg-info/`、`__pycache__/`、`.pytest_cache/`。

开工前已存在、本任务未改动的未跟踪文件：`tasks/M0_dispatch.md`。

## 运行过的命令与输出摘要

### 1. 安装 PyTorch（CUDA 13.2 wheel）

```text
conda run -n AIControl python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
```

退出码 0。解析并安装 `torch-2.15.0+cu132`（cp312，manylinux_2_28_x86_64），同时装上其依赖（`cuda-toolkit==13.2.2`、`triton==3.9.0` 及 nvidia-* CUDA 库）。结尾：

```text
Successfully installed ... torch-2.15.0+cu132 triton-3.9.0 ...
```

### 2. 安装其余依赖

```text
conda run -n AIControl python -m pip install 'mujoco>=3,<4' numpy scipy h5py matplotlib imageio imageio-ffmpeg pyyaml pytest tqdm
```

退出码 0。PyPI 上 mujoco 最新版是 4.0.1，按「最新 3.x」约束装到 3.17.0。结尾：

```text
Successfully installed absl-py-2.5.1 contourpy-1.4.0 cycler-0.12.1 etils-1.15.0 fonttools-4.68.0 glfw-2.11.0 h5py-3.16.0 imageio-2.38.1 imageio-ffmpeg-0.8.1 iniconfig-2.3.1 kiwisolver-1.5.1 matplotlib-3.11.2 mujoco-3.17.0 numpy-2.5.4 pillow-13.0.0 pluggy-1.7.0 pygments-2.21.0 pyopengl-3.1.10 pyparsing-3.3.3 pytest-9.2.0 python-dateutil-4.0.0 pyyaml-6.0.3 scipy-1.18.1 tqdm-4.70.1 tzdata-2026.6 websockets-17.4 zipp-4.1.2
```

额外装上的包都是上述直接依赖的传递依赖：mujoco 带入 absl-py、etils、glfw、pyopengl、websockets、zipp；matplotlib 带入 contourpy、cycler、fonttools、kiwisolver、pillow、pyparsing、python-dateutil、tzdata；pytest 带入 iniconfig、pluggy、pygments。未安装 DPPO / DSRL / robosuite / robomimic，也未安装 torchvision（官方命令里有它，本里程碑不需要）。

### 3. `pip install -e .`

```text
conda run -n AIControl python -m pip install -e .
```

退出码 0。

```text
Obtaining file:///home/minashiki/AIControl/FinalHomework
...
Successfully built aloha_pp
Installing collected packages: aloha_pp
Successfully installed aloha_pp-0.0.1
```

随后 `conda run -n AIControl python -c 'import aloha_pp'` 成功，模块路径为 `src/aloha_pp/__init__.py`。`pyproject.toml` 故意不声明 torch 等运行依赖，避免 `pip install -e .` 从 PyPI 把 CUDA 版 torch 换掉。

### 4. `check_env.py` 完整输出

```text
conda run -n AIControl python scripts/check_env.py
```

退出码 0。

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
[PASS] wrist_cam: shape=(96, 96, 3) dtype=uint8 min=0 max=45 saved=/home/minashiki/AIControl/FinalHomework/outputs/m0_wrist_cam.png
ALL CHECKS PASSED
```

### 5. pytest

```text
conda run -n AIControl python -m pytest tests -q
```

退出码 0。

```text
....                                                                     [100%]
4 passed in 0.99s
```

覆盖项：`torch.cuda.is_available()`、`get_arch_list()` 含 `sm_120`、加载 `aloha/aloha.xml` 且 `neq == 1` 并 `mj_step`、`MUJOCO_GL=egl` 下 `wrist_cam_left` 96×96 非空且像素不全相同。无 CUDA 时前两项是断言失败，不是 skip。

## torch 版本与 CUDA wheel

查阅 https://pytorch.org/get-started/locally/ 的安装选择器脚本 `https://pytorch.org/assets/quick-start-module.js`（页面静态 HTML 里的默认文案是过期的 Stable 2.7.0 / CUDA 11.8，以该脚本为准）：

- `stable_version = "Stable (2.15.0)"`
- Stable 的 CUDA 选项：`cuda.x` = **13.2**（Linux 默认首选），`cuda.y` = **13.4**
- Linux + pip + Stable + CUDA 13.2 的命令是 `pip3 install torch torchvision`；对应 libtorch 文件名是 `libtorch-shared-with-deps-2.15.0+cu132.zip`
- CUDA 13.4 的命令是 `--index-url https://download.pytorch.org/whl/cu134`

本机驱动 595.99.02，`nvidia-smi` 显示最高 CUDA 13.2。13.4 高于驱动上限，因此不用。实际安装命令显式指向 cu132 索引，装到的包是 **`torch==2.15.0+cu132`**，`torch.version.cuda == 13.2`，`get_arch_list()` 含 `sm_120`，2048×2048 GPU 矩阵乘最大绝对误差 `2.289e-04`。

PyPI 当前 torch 也是 2.15.0，与官方 Stable 一致。

## 未完成或存疑

- 没有未完成项。自检与 pytest 均退出码 0。
- `torch==2.15.0+cu132` 的依赖把部分 NVIDIA pip 包装到了高于 13.2 的库版本：`nvidia-cublas==13.4.1.3`（wheel 对 cublas 的 pin 就是 `13.4.1.3.*`）、`nvidia-nvjitlink==13.4.92`（约束是 `<14,>=13.2.86`，解析到最新）、`cuda-bindings==13.4.4`、`nvidia-cuda-cccl==13.3.4.3.1`。这是 cu132 wheel 自己的依赖树，不是选择了官方页面上的 CUDA 13.4 wheel。在本机驱动上 CUDA 可用、`sm_120` 与矩阵乘都通过，因此保留解析结果，并在 `environment.yml` 里钉死这些版本。
- 腕部相机图是默认姿态下的夹爪近景，画面很暗（像素最大 45），但形状为 96×96×3，且像素不全相同，已写入 `outputs/m0_wrist_cam.png`。场景光照不在 M0 范围。
- `environment.yml` 不包含 `pip install -e .`。新建环境后需要在仓库根目录再执行一次可编辑安装。
