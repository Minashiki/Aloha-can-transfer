# M0 交付记录 r2（F1）

只改了 `.gitignore` 里审查结论点名的产物目录规则，每条加上前导 `/`。其余规则、顺序和注释未动。未执行 `git add` / `commit` / `push` / `config`。未改 `docs/PLAN.md`、`docs/WORKFLOW.md`、`aloha/`。验收命令都在 conda 环境 `AIControl` 中运行（git 命令除外）。

## 改动文件清单

- 修改：`.gitignore`
- 新增：`tasks/M0_delivery_r2.md`（本文件）

## `git diff .gitignore`

```diff
diff --git a/.gitignore b/.gitignore
index 4922647..2f2c443 100644
--- a/.gitignore
+++ b/.gitignore
@@ -3,8 +3,8 @@ __pycache__/
 *.py[cod]
 *.egg-info/
 .eggs/
-build/
-dist/
+/build/
+/dist/
 .pytest_cache/
 .mypy_cache/
 .ruff_cache/
@@ -17,36 +17,36 @@ env/
 .conda/
 
 # ---- 数据集（robomimic hdf5、重渲染示范等）----
-data/
-datasets/
+/data/
+/datasets/
 *.hdf5
 *.h5
 *.npz
 *.zarr/
 
 # ---- 训练产物：checkpoint、日志 ----
-checkpoints/
-ckpt/
+/checkpoints/
+/ckpt/
 *.pt
 *.pth
 *.ckpt
 *.safetensors
-runs/
-logs/
-wandb/
-outputs/
+/runs/
+/logs/
+/wandb/
+/outputs/
 *.log
 
 # ---- 实验结果：视频、图片、曲线 ----
-results/
-videos/
-figures/
+/results/
+/videos/
+/figures/
 *.mp4
 *.gif
 *.avi
 
 # ---- 外部下载的资产（robosuite 原始资产等，按需放这里）----
-third_party/
+/third_party/
 
 # ---- 编辑器 / 系统 ----
 .vscode/
@@ -57,8 +57,8 @@ third_party/
 Thumbs.db
 
 # ---- 临时文件 ----
-tmp/
-scratch/
+/tmp/
+/scratch/
 *.tmp
 
 # ---- 仅本地保留的规划文档（只有 Claude 可写，不入 git）----
```

## 验收命令与输出摘要

### 1. `git check-ignore -v src/aloha_pp/data/__init__.py`

无输出，退出码 **1**（没有命中任何忽略规则）。

### 2. `git status --short --untracked-files=all`

退出码 0。其中出现 `?? src/aloha_pp/data/__init__.py`。完整输出：

```text
 M .gitignore
?? assets/scene/.gitkeep
?? configs/.gitkeep
?? environment.yml
?? pyproject.toml
?? scripts/check_env.py
?? src/aloha_pp/__init__.py
?? src/aloha_pp/control/__init__.py
?? src/aloha_pp/data/__init__.py
?? src/aloha_pp/dyn/__init__.py
?? src/aloha_pp/kin/__init__.py
?? src/aloha_pp/planning/__init__.py
?? src/aloha_pp/policy/__init__.py
?? src/aloha_pp/rl/__init__.py
?? src/aloha_pp/sim/__init__.py
?? src/aloha_pp/utils/__init__.py
?? tasks/M0_delivery_r1.md
?? tasks/M0_dispatch.md
?? tasks/M0_recheck_r1.md
?? tasks/M0_review_r1.md
?? tasks/M0_review_r2.md
?? tests/test_smoke.py
```

（写完本文件后，未跟踪清单还会多出 `tasks/M0_delivery_r2.md`。上面是验收命令当时的输出。）

### 3. 仍应被忽略的路径

```text
git check-ignore -v outputs/m0_wrist_cam.png src/aloha_pp.egg-info docs/PLAN.md docs/WORKFLOW.md
```

退出码 0，四条都命中：

```text
.gitignore:37:/outputs/	outputs/m0_wrist_cam.png
.gitignore:4:*.egg-info/	src/aloha_pp.egg-info
.gitignore:65:docs/PLAN.md	docs/PLAN.md
.gitignore:66:docs/WORKFLOW.md	docs/WORKFLOW.md
```

`__pycache__` 仍被忽略（退出码 0）：

```text
git check-ignore -v src/aloha_pp/__pycache__ tests/__pycache__ src/aloha_pp/__pycache__/__init__.cpython-312.pyc
.gitignore:2:__pycache__/	src/aloha_pp/__pycache__
.gitignore:2:__pycache__/	tests/__pycache__
.gitignore:2:__pycache__/	src/aloha_pp/__pycache__/__init__.cpython-312.pyc
```

### 4. `conda run -n AIControl python scripts/check_env.py`

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

### 5. `conda run -n AIControl python -m pytest tests -q`

退出码 0。

```text
....                                                                     [100%]
4 passed in 0.82s
```

## 未完成或存疑

没有未完成项。第 1 条命令退出码为 1，这是 `git check-ignore` 在路径未被忽略时的正常结果。
