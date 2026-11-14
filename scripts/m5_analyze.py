"""Read-only plots and tables for the M5 logs and dataset.

Run: conda run -n AIControl python -u scripts/m5_analyze.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import h5py

from aloha_pp.data.dataset import list_demos, load_episode
from aloha_pp.data.retarget import HUMAN_INFO_PHASES, PHASE_NAMES
from aloha_pp.planning.schedule import STATE_NAMES
from aloha_pp.robot import repo_root


def m1_hz(dataset: Path) -> float:
    with h5py.File(dataset, "r") as handle:
        return float(handle.attrs["control_hz"])


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    text = row.get(key, "")
    if text == "":
        return float("nan")
    return float(text)


def main() -> None:
    root = repo_root()
    out = root / "outputs" / "m5"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(out / "episodes.csv")
    retarget = [row for row in rows if row["source"] == "retarget"]
    noise = [row for row in rows if row["source"] == "planner_noise"]
    reasons = Counter(row["reason"] if row["reason"] else "success" for row in retarget)
    n = len(retarget)
    n_ok = sum(row["success"] == "True" for row in retarget)
    rate = (n_ok / n) if n else float("nan")
    print(f"retarget {n_ok}/{n} = {rate:.3f}")
    print("reasons")
    for name, count in reasons.most_common():
        print(f"  {name} {count}")

    labels = [name for name, _ in reasons.most_common()]
    counts = [count for _, count in reasons.most_common()]
    plt.figure(figsize=(7, 3.5))
    plt.bar(labels, counts)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("demos")
    plt.title("Retarget outcomes")
    plt.tight_layout()
    plt.savefig(fig_dir / "retarget_reasons.png", dpi=120)
    plt.close()

    def hist(values: list[float], name: str, xlabel: str) -> None:
        clean = [value for value in values if np.isfinite(value)]
        plt.figure(figsize=(5, 3.2))
        if clean:
            plt.hist(clean, bins=20)
        plt.xlabel(xlabel)
        plt.ylabel("demos")
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=120)
        plt.close()

    hist([_float(row, "slowdown") for row in retarget], "slowdown.png", "slowdown")
    hist(
        [_float(row, "length") for row in retarget if row["success"] == "True"],
        "length.png",
        "control ticks",
    )
    hist([_float(row, "ref_peak_ratio") for row in retarget], "ref_peak_ratio.png", "reference peak / v_max")
    hist(
        [_float(row, "ref_unlimited_ratio") for row in retarget],
        "ref_unlimited_ratio.png",
        "unlimited reference peak / v_max",
    )
    hist([_float(row, "exec_peak_ratio") for row in retarget], "exec_peak_ratio.png", "closed-loop peak / v_max")
    hist(
        [_float(row, "place_xy") for row in rows if row["success"] == "True"],
        "place_xy.png",
        "final place xy error (m)",
    )

    plt.figure(figsize=(4.5, 4.2))
    for row, color, label in (
        ([row for row in retarget if row["success"] != "True"], "C3", "fail"),
        ([row for row in retarget if row["success"] == "True"], "C0", "success"),
    ):
        xs = [_float(item, "can_x") for item in row]
        ys = [_float(item, "can_y") for item in row]
        plt.scatter(xs, ys, s=12, c=color, label=label)
    plt.xlabel("can x (m)")
    plt.ylabel("can y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "can_xy.png", dpi=120)
    plt.close()

    dataset = out / "dataset" / "demos.hdf5"
    composition: Counter[str] = Counter()
    human = []
    phase_ticks: Counter[str] = Counter()
    if dataset.is_file():
        for name in list_demos(dataset):
            episode = load_episode(dataset, name)
            composition[episode["source"]] += 1
            if episode.get("phase_scheme") == "retarget" and "phase" in episode:
                labels = episode["phase"]
                human.append(float(np.mean(np.isin(labels, HUMAN_INFO_PHASES))))
                for value, count in zip(*np.unique(labels, return_counts=True)):
                    phase_ticks[PHASE_NAMES[int(value)]] += int(count)
            elif episode.get("phase_scheme") == "planner" and "phase" in episode:
                pass
        if human:
            hist(human, "human_fraction.png", "human position or carry-residual fraction")
            print(
                "human fraction min/median/max "
                f"{np.min(human):.3f} {np.median(human):.3f} {np.max(human):.3f}"
            )
        if phase_ticks:
            names_ph = [item for item in PHASE_NAMES if phase_ticks[item]]
            ticks = [phase_ticks[item] for item in names_ph]
            hz = float(m1_hz(dataset))
            plt.figure(figsize=(7, 3.5))
            plt.bar(names_ph, np.asarray(ticks) / hz)
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("total seconds")
            plt.tight_layout()
            plt.savefig(fig_dir / "segment_seconds.png", dpi=120)
            plt.close()
            print("segment total seconds", {name: ticks[i] / hz for i, name in enumerate(names_ph)})
            per = {name: [] for name in PHASE_NAMES}
            with h5py.File(dataset, "r") as handle:
                for demo in handle["data"].values():
                    if str(demo.attrs.get("phase_scheme", "")) != "retarget":
                        continue
                    labels = np.asarray(demo["phase"])
                    for code, name in enumerate(PHASE_NAMES):
                        per[name].append(float(np.count_nonzero(labels == code) / hz))
            print("segment seconds min/median/max")
            for name, values in per.items():
                if not values:
                    continue
                array = np.asarray(values)
                print(
                    f"  {name} {array.min():.2f} {np.median(array):.2f} {array.max():.2f}"
                )
    print("dataset", dict(composition))
    print("planner phase vocabulary", ",".join(STATE_NAMES))
    print(f"planner_noise attempts {len(noise)} successes {sum(row['success'] == 'True' for row in noise)}")

    summary = out / "summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "value"])
        writer.writerow(["retarget_n", n])
        writer.writerow(["retarget_success", n_ok])
        writer.writerow(["retarget_rate", f"{rate:.6f}"])
        for name, count in reasons.most_common():
            writer.writerow([f"reason_{name}", count])
        for source, count in sorted(composition.items()):
            writer.writerow([f"dataset_{source}", count])
        slow = np.array([_float(row, "slowdown") for row in retarget], dtype=float)
        if np.any(np.isfinite(slow)):
            writer.writerow(["slowdown_min", f"{np.nanmin(slow):.4f}"])
            writer.writerow(["slowdown_median", f"{np.nanmedian(slow):.4f}"])
            writer.writerow(["slowdown_max", f"{np.nanmax(slow):.4f}"])
        kept_rows = [row for row in retarget if row["success"] == "True"]
        print("segment scale min/median/max (success)")
        for column in (
            "scale_synth_approach",
            "scale_human_pregrasp",
            "scale_carry",
            "scale_place_slide",
            "scale_place_down",
            "scale_retreat",
        ):
            values = np.array([_float(row, column) for row in kept_rows], dtype=float)
            if not np.any(np.isfinite(values)):
                continue
            print(
                f"  {column} {np.nanmin(values):.3f} {np.nanmedian(values):.3f} {np.nanmax(values):.3f}"
            )
            writer.writerow([f"{column}_min", f"{np.nanmin(values):.4f}"])
            writer.writerow([f"{column}_median", f"{np.nanmedian(values):.4f}"])
            writer.writerow([f"{column}_max", f"{np.nanmax(values):.4f}"])
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
