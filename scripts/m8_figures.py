"""M8: collect the report figures and videos, and draw the summary figures.

Reads only ``outputs/`` (and nothing else); writes only ``report/figures/`` and
``report/videos/``. Both directories are ignored by ``report/.gitignore``.

    conda run -n AIControl python scripts/m8_figures.py

Existing figures and videos are copied under a chapter prefix. New figures:

* ``7_m6_val_success.png``   M6 in-training closed-loop validation (4 episodes)
* ``8_m7_training.png``      M7 validation / training-episode success, Q, alpha, log pi
* ``9_success_rates.png``    success rate with Wilson 95% interval, seeds 0-49
* ``9_failure_modes.png``    failure reasons per method, seeds 0-49
* ``9_place_duration.png``   placement error and episode duration of successful episodes
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "report" / "figures"
VID = ROOT / "report" / "videos"

# Reference categorical slots (light mode), assigned in fixed order.
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 10,
    }
)

COPY_FIGURES: list[tuple[str, str]] = [
    ("m0_wrist_cam.png", "2_wrist_cam_m0.png"),
    ("m1/agentview_reset_256.png", "2_agentview_reset.png"),
    ("m1/review_agent_256.png", "2_agentview_review.png"),
    ("m1/review_wrist_256.png", "2_wrist_review.png"),
    ("m1/agentview_grasp.png", "2_agentview_grasp.png"),
    ("m1/agentview_reset_96.png", "2_agentview_96.png"),
    ("m1/wrist_cam_left.png", "2_wrist_96.png"),
    ("m2/coupling_heatmap.png", "3_coupling_heatmap.png"),
    ("m2/coupling_curves.png", "3_coupling_curves.png"),
    ("m2/coupling_shoulder_elbow_map.png", "3_coupling_shoulder_elbow_map.png"),
    ("m2/coupling_vs_wrist_angle.png", "3_coupling_vs_wrist_angle.png"),
    ("m2/coupling_shoulder_wrist_map.png", "3_coupling_shoulder_wrist_map.png"),
    ("m2/single_joint_shoulder.png", "3_single_joint_shoulder.png"),
    ("m2/single_joint_elbow.png", "3_single_joint_elbow.png"),
    ("m2/single_joint_shoulder_fast.png", "3_single_joint_shoulder_fast.png"),
    ("m2/single_joint_elbow_fast.png", "3_single_joint_elbow_fast.png"),
    ("m2/dynamic_coupling_no_gravity.png", "3_dynamic_coupling_no_gravity.png"),
    ("m2/wrist_singularity.png", "3_wrist_singularity.png"),
    ("m3/figures/b0_gravity_sag.png", "4_b0_gravity_sag.png"),
    ("m3/figures/c2_vs_b1.png", "4_c2_vs_b1.png"),
    ("m3/figures/c4_contact.png", "4_c4_contact.png"),
    ("m3/figures/fixed1_fast_error_torque.png", "4_fixed1_fast_error_torque.png"),
    ("m3/figures/rmse_vs_speed.png", "4_rmse_vs_speed.png"),
    ("m4/figures/e1_paths.png", "5_e1_paths.png"),
    ("m4/figures/e1_speed.png", "5_e1_speed.png"),
    ("m4/figures/e1_fixed1_medium.png", "5_e1_fixed1_medium.png"),
    ("m4/figures/e2_scatter.png", "5_e2_scatter.png"),
    ("m4/figures/e4_compare.png", "5_e4_compare.png"),
    ("m5/figures/retarget_reasons.png", "6_retarget_reasons.png"),
    ("m5/figures/can_xy.png", "6_can_xy.png"),
    ("m5/figures/place_xy.png", "6_place_xy.png"),
    ("m5/figures/length.png", "6_length.png"),
    ("m5/figures/segment_seconds.png", "6_segment_seconds.png"),
    ("m5/figures/slowdown.png", "6_slowdown.png"),
    ("m5/figures/human_fraction.png", "6_human_fraction.png"),
    ("m5/figures/ref_peak_ratio.png", "6_ref_peak_ratio.png"),
    ("m5/figures/exec_peak_ratio.png", "6_exec_peak_ratio.png"),
    ("m5/videos/obs_01_retarget_0.png", "6_obs_retarget_0.png"),
    ("m6/train/loss.png", "7_m6_loss.png"),
]

COPY_VIDEOS: list[tuple[str, str]] = [
    ("m1/grasp.mp4", "m1_grasp_smoke.mp4"),
    ("m3/videos/B0_fixed_1_fast.mp4", "m3_B0_fixed_1_fast.mp4"),
    ("m3/videos/C3_fixed_1_fast.mp4", "m3_C3_fixed_1_fast.mp4"),
    *[
        (f"m4/videos/E1_C3_{mode}_{speed}.mp4", f"m4_E1_C3_{mode}_{speed}.mp4")
        for mode in ("fixed_1", "fixed_2")
        for speed in ("fast", "medium", "slow")
    ],
    ("m4/videos/E2_C3_random_s00.mp4", "m4_E2_C3_random_s00.mp4"),
    ("m5/videos/replay_01_retarget_0.mp4", "m5_replay_retarget_0.mp4"),
    ("m5/videos/replay_00_planner_noise_1001.mp4", "m5_replay_planner_noise_1001.mp4"),
    *[(f"m6/train/eval/videos/seed{s:02d}.mp4", f"m6_bc_seed{s:02d}.mp4") for s in range(4)],
    *[(f"m7/video_bc_dsrl_best/videos/seed{s:02d}.mp4", f"m7_dsrl_best_seed{s:02d}.mp4") for s in range(4)],
]

# Methods compared on seeds 0-49 (random mode). Label, eval directory, family.
METHODS = [
    ("规划器 (M4 E2)", None, "planner"),
    ("BC 随机噪声 (M6, 仅参考)", "m6/train/eval", "ref"),
    ("BC w=0", "m7/eval_bc_zero", "bc"),
    ("BC 种子噪声", "m7/eval_bc_seeded", "bc"),
    ("DSRL best (主结果)", "m7/eval_bc_dsrl_best", "dsrl"),
    ("DSRL last", "m7/eval_bc_dsrl_last", "dsrl"),
]
FAMILY_COLOR = {"planner": INK2, "ref": "#c9c8c3", "bc": C1, "dsrl": C2}
FAILURES = ["not_grasped", "not_lifted", "dropped_after_lift", "not_released", "outside"]
FAILURE_ZH = {
    "not_grasped": "未抓住",
    "not_lifted": "未抬起",
    "dropped_after_lift": "抬起后掉落",
    "not_released": "未松手",
    "outside": "放在目标格外",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def episodes(method_dir: str | None) -> list[dict]:
    """Per-seed rows with success, failure, place_xy (m), duration_s."""
    if method_dir is None:
        rows = [r for r in read_csv(OUT / "m4" / "summary.csv") if r["experiment"] == "E2" and r["seed"] != ""]
        return [
            {
                "seed": int(r["seed"]),
                "success": r["success"] == "True",
                "failure": r["failure"],
                "place_xy": float(r["place_xy"]) if r["place_xy"] else float("nan"),
                "duration_s": float(r["episode_time"]),
            }
            for r in rows
        ]
    rows = read_csv(OUT / method_dir / "episodes.csv")
    return [
        {
            "seed": int(r["seed"]),
            "success": r["success"] == "True",
            "failure": r["failure"],
            "place_xy": float(r["place_xy"]) if r["place_xy"] else float("nan"),
            "duration_s": float(r["duration_s"]),
        }
        for r in rows
    ]


def copy_all() -> list[str]:
    missing = []
    FIG.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    for src, dst in COPY_FIGURES:
        path = OUT / src
        if not path.exists():
            missing.append(str(path))
            continue
        shutil.copy2(path, FIG / dst)
    for src, dst in COPY_VIDEOS:
        path = OUT / src
        if not path.exists():
            missing.append(str(path))
            continue
        shutil.copy2(path, VID / dst)
    return missing


def fig_m6_val() -> None:
    xs, ks = [], []
    for path in sorted((OUT / "m6" / "train").glob("val_epoch*/eval.json")):
        data = json.loads(path.read_text())
        xs.append(int(path.parent.name.removeprefix("val_epoch")))
        ks.append(int(data["n_success"]))
    log = [json.loads(line) for line in (OUT / "m6" / "train" / "train_log.jsonl").open()]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.4))
    ep = [r["epoch"] for r in log]
    ax0.plot(ep, [r["train_loss"] for r in log], color=C1, lw=1.5, label="训练损失")
    vx = [r["epoch"] for r in log if r.get("val_loss") is not None]
    vy = [r["val_loss"] for r in log if r.get("val_loss") is not None]
    ax0.plot(vx, vy, color=C2, lw=1.5, label="验证损失（留出示范）")
    ax0.set_yscale("log")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("ε 预测 MSE（对数）")
    ax0.set_title("M6 BC 训练 / 验证损失", color=INK)
    ax0.legend()
    ax1.bar(xs, ks, width=60, color=C1, edgecolor="white", linewidth=2)
    ax1.set_ylim(0, 4.3)
    ax1.set_yticks(range(5))
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("成功数 / 4")
    ax1.set_title("M6 训练中闭环验证（每次 4 局）", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "7_m6_val_success.png")
    plt.close(fig)


def fig_m7_training() -> dict:
    train, val = [], []
    for line in (OUT / "m7" / "train" / "train_log.jsonl").open():
        rec = json.loads(line)
        if rec["event"] == "train":
            train.append(rec)
        elif rec["event"] == "val":
            val.append(rec)
    steps = np.array([r["env_steps"] for r in train], dtype=float)
    eps = np.array([r["episodes"] for r in train], dtype=float)
    succ = np.array([r["episode_successes"] for r in train], dtype=float)
    # Training-episode success rate over 20k-transition windows (cumulative counters).
    window = 20000
    edges = np.arange(0, steps[-1] + window, window)
    idx = np.searchsorted(steps, edges, side="right") - 1
    idx[idx < 0] = 0
    win_x, win_y = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        de = eps[b] - eps[a]
        if de > 0:
            win_x.append(steps[b])
            win_y.append((succ[b] - succ[a]) / de)
    vx = np.array([r["env_steps"] for r in val], dtype=float)
    vy = np.array([r["n_success"] for r in val], dtype=float) / 16.0
    warm = 1501 * 8
    best = 1_032_000
    learn = [r for r in train if not r["warming"] and r["q_mean"] is not None]
    lx = np.array([r["env_steps"] for r in learn], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10.5), sharex=True)
    ax = axes[0]
    ax.plot(vx, vy, color=C1, lw=0, marker="o", ms=2.5, alpha=0.55, label="验证成功率（16 局，种子 20000–20015）")
    k = 10
    smooth = np.convolve(vy, np.ones(k) / k, mode="valid")
    ax.plot(vx[k - 1 :], smooth, color=C1, lw=2, label="验证成功率，10 次滑动平均")
    ax.plot(win_x, win_y, color=C2, lw=2, label="训练局成功率（每 2 万转移）")
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("成功率")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("M7 DSRL 训练过程（2×10⁶ 环境转移，8 个并行环境）", color=INK)
    ax = axes[1]
    ax.plot(lx, [r["q_mean"] for r in learn], color=C1, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("批均 Q（对数）")
    ax = axes[2]
    ax.plot(lx, [r["alpha"] for r in learn], color=C3, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("温度 α（对数）")
    ax = axes[3]
    ax.plot(lx, [r["logp_mean"] for r in learn], color=C2, lw=0.6, alpha=0.8)
    ax.set_ylabel("批均 log π(a|f)\n（a = w/1.5）")
    ax.set_xlabel("环境转移数（各环境之和）")
    for ax in axes:
        ax.axvline(warm, color=GRAY, lw=1, ls=":")
        ax.axvline(best, color=INK2, lw=1, ls="--")
    axes[0].annotate("热身结束\n12 008", (warm, 0.05), xytext=(60_000, 0.25), fontsize=8, color=INK2,
                     arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8})
    axes[0].annotate("ckpt_best\n1 032 000（首次 16/16）", (best, 1.0), xytext=(1_150_000, 0.35), fontsize=8,
                     color=INK2, arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.8})
    fig.tight_layout()
    fig.savefig(FIG / "8_m7_training.png")
    plt.close(fig)

    # Numbers quoted by the report.
    def mean_val(lo: float, hi: float) -> float:
        sel = (vx > lo) & (vx <= hi)
        return float(np.mean(vy[sel] * 16)) if np.any(sel) else float("nan")

    last = [r for r in train if r["env_steps"] >= 1_900_000]
    stats = {
        "val_mean_after_1.5M": mean_val(1_500_000, 2_000_000),
        "val_first_16": float(vx[np.argmax(vy >= 1.0)]),
        "train_rate_last_100k": (last[-1]["episode_successes"] - last[0]["episode_successes"])
        / (last[-1]["episodes"] - last[0]["episodes"]),
        "train_counts_last_100k": [
            last[-1]["episode_successes"] - last[0]["episode_successes"],
            last[-1]["episodes"] - last[0]["episodes"],
        ],
        "final_alpha": train[-1]["alpha"],
        "final_q_mean": train[-1]["q_mean"],
        "n_val": len(val),
    }
    return stats


def fig_success(rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels = [m[0] for m in METHODS]
    for i, (label, _dir, fam) in enumerate(METHODS):
        ep = rows[label]
        k, n = sum(e["success"] for e in ep), len(ep)
        lo, hi = wilson(k, n)
        y = len(METHODS) - 1 - i
        ax.barh(y, k / n, height=0.6, color=FAMILY_COLOR[fam], edgecolor="white", linewidth=2)
        ax.errorbar(k / n, y, xerr=[[k / n - lo], [hi - k / n]], color=INK, capsize=3, lw=1)
        ax.text(min(hi + 0.015, 1.02), y, f"{k}/{n}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlim(0, 1.25)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("成功率（评测种子 0–49，random 模式；误差线为 Wilson 95% 区间）")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "9_success_rates.png")
    plt.close(fig)


def fig_failures(rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = [C1, C2, C3, C4, C5]
    left = np.zeros(len(METHODS))
    counts = np.zeros((len(METHODS), len(FAILURES)))
    for i, (label, _dir, _fam) in enumerate(METHODS):
        for e in rows[label]:
            if not e["success"]:
                counts[i, FAILURES.index(e["failure"])] += 1
    ys = np.arange(len(METHODS))[::-1]
    for j, name in enumerate(FAILURES):
        ax.barh(ys, counts[:, j], left=left, height=0.6, color=colors[j], edgecolor="white", linewidth=2,
                label=FAILURE_ZH[name])
        left += counts[:, j]
    for y, total in zip(ys, left):
        ax.text(total + 0.3, y, f"{int(total)}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in METHODS])
    ax.set_xlabel("失败局数（共 50 局）")
    ax.set_xlim(0, 22)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, 1.16), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "9_failure_modes.png")
    plt.close(fig)


def fig_place_duration(rows: dict[str, list[dict]]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for i, (label, _dir, fam) in enumerate(METHODS):
        ok = [e for e in rows[label] if e["success"]]
        y = len(METHODS) - 1 - i
        place = np.array([e["place_xy"] * 1000 for e in ok])
        dur = np.array([e["duration_s"] for e in ok])
        for ax, vals in ((ax0, place), (ax1, dur)):
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(vals, y + jitter, s=12, color=FAMILY_COLOR[fam], alpha=0.75, linewidths=0)
            med = float(np.median(vals))
            ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=2)
    ax0.set_yticks(range(len(METHODS)))
    ax0.set_yticklabels([m[0] for m in METHODS][::-1])
    ax0.set_xlabel("成功局 Can 中心到目标格中心的水平距离（mm）")
    ax1.set_xlabel("成功局 episode 时长（s）")
    ax0.set_title("放置误差（竖线为中位数）", color=INK)
    ax1.set_title("完成用时（竖线为中位数）", color=INK)
    for ax in (ax0, ax1):
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "9_place_duration.png")
    plt.close(fig)


def main() -> int:
    missing = copy_all()
    fig_m6_val()
    stats = fig_m7_training()
    rows = {label: episodes(d) for label, d, _ in METHODS}
    for label, ep in rows.items():
        if len(ep) != 50 or sorted(e["seed"] for e in ep) != list(range(50)):
            raise RuntimeError(f"{label}: expected seeds 0-49")
    fig_success(rows)
    fig_failures(rows)
    fig_place_duration(rows)
    for label, ep in rows.items():
        ok = [e for e in ep if e["success"]]
        print(
            f"{label}: {len(ok)}/50 place_xy_median {np.median([e['place_xy'] for e in ok]) * 1000:.2f} mm"
            f" duration_median {np.median([e['duration_s'] for e in ok]):.2f} s"
        )
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"figures {len(list(FIG.glob('*.png')))} videos {len(list(VID.glob('*.mp4')))}")
    if missing:
        print("MISSING:\n" + "\n".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
