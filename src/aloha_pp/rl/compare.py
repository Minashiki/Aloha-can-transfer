"""Success, failures, IK, placement, and paired seed comparisons.

Wilson intervals stay in the table as a description of each rate. They are
not the test for whether one method improved on another. That comparison is
the exact McNemar test on seeds that disagree. The M6 row is a reference
only: its noise was not fixed, and it is not a paired baseline.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def load_eval(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def success_by_seed(summary: dict) -> dict[int, bool]:
    return {int(episode["seed"]): bool(episode["success"]) for episode in summary["episodes"]}


def mcnemar_p(improve: int, regress: int) -> float:
    """Two-sided exact McNemar p-value.

    Discordant seeds are a Binomial(n, 1/2) under the null. ``improve`` is
    the number of seeds that fail on the left and succeed on the right.
    The p-value is twice the lower tail, capped at 1.
    """
    b = int(improve)
    c = int(regress)
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, (2.0 * float(tail)) / float(1 << n))


def transitions(left: dict[int, bool], right: dict[int, bool]) -> dict:
    """Seeds present in both maps. ``right_only`` succeeded on the right and failed on the left."""
    keys = sorted(set(left) & set(right))
    return {
        "n_compared": len(keys),
        "both_success": [k for k in keys if left[k] and right[k]],
        "both_fail": [k for k in keys if (not left[k]) and (not right[k])],
        "left_only": [k for k in keys if left[k] and (not right[k])],
        "right_only": [k for k in keys if right[k] and (not left[k])],
    }


def row_from_eval(name: str, summary: dict) -> dict:
    successes = [episode for episode in summary["episodes"] if episode["success"]]
    places = [float(episode["place_xy"]) for episode in successes]
    return {
        "method": name,
        "n": int(summary["n"]),
        "n_success": int(summary["n_success"]),
        "success_rate": float(summary["success_rate"]),
        "wilson_95": [float(summary["wilson_95"][0]), float(summary["wilson_95"][1])],
        "failure_counts": dict(summary.get("failure_counts") or {}),
        "ik_current_fail_ticks_sum": int(summary.get("ik_current_fail_ticks_sum", 0)),
        "ik_preview_fail_ticks_sum": int(summary.get("ik_preview_fail_ticks_sum", 0)),
        "ik_fail_ticks_sum": int(summary.get("ik_fail_ticks_sum", 0)),
        "place_xy_median_success": float(np.median(places)) if places else None,
        "place_xy_median_all": float(summary["place_xy_median"]),
        "duration_s_median": float(summary["duration_s_median"]),
    }


def _truthy(text: str) -> bool:
    return str(text).strip().lower() in {"true", "1", "yes"}


def row_from_m4(path: Path) -> tuple[dict, dict[int, bool]]:
    """Planner E2 from ``summary.csv``. Placement median is over successful seeds."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    episodes = [row for row in rows if row.get("experiment") == "E2" and row.get("name") != "E2_summary"]
    if not episodes:
        raise ValueError(f"no E2 rows in {path}")
    summary = next((row for row in rows if row.get("name") == "E2_summary"), None)
    successes = {int(row["seed"]): _truthy(row["success"]) for row in episodes if row.get("seed")}
    places = [float(row["place_xy"]) for row in episodes if _truthy(row["success"]) and row.get("place_xy")]
    durations = [float(row["episode_time"]) for row in episodes if row.get("episode_time")]
    if summary is not None and summary.get("n"):
        n = int(float(summary["n"]))
        n_success = int(float(summary["n_success"]))
        rate = float(summary["success_rate"])
        wilson = [float(summary["wilson_low"]), float(summary["wilson_high"])]
    else:
        n = len(successes)
        n_success = sum(1 for value in successes.values() if value)
        rate = float(n_success) / float(n)
        wilson = [None, None]
    row = {
        "method": "planner_e2",
        "n": n,
        "n_success": n_success,
        "success_rate": rate,
        "wilson_95": wilson,
        "failure_counts": dict(Counter(
            row["failure"] for row in episodes if row.get("failure") and not _truthy(row["success"])
        )),
        "ik_current_fail_ticks_sum": None,
        "ik_preview_fail_ticks_sum": None,
        "ik_fail_ticks_sum": None,
        "place_xy_median_success": float(np.median(places)) if places else None,
        "place_xy_median_all": float(summary["place_xy"]) if summary and summary.get("place_xy") else None,
        "duration_s_median": float(np.median(durations)) if durations else None,
    }
    return row, successes


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def recorded_label(summary: dict) -> str:
    """Whether the logged episodes include a video render."""
    mode = summary.get("videos")
    seeds = summary.get("video_seeds")
    if mode == "none" or seeds == []:
        return "不含录像"
    if isinstance(seeds, list) and seeds:
        shown = ",".join(str(seed) for seed in seeds)
        return f"含录像 {shown}"
    found = [int(episode["seed"]) for episode in summary.get("episodes", []) if episode.get("video")]
    if found:
        return "含录像 " + ",".join(str(seed) for seed in found)
    if mode is None:
        return "未记录录像字段"
    return "不含录像"


def noise_label(name: str) -> str:
    if name == "bc":
        return "随机噪声（未固定）、种子 0–3 录像，仅作参考"
    if name == "bc_zero":
        return "w = 0"
    if name == "bc_seeded":
        return "按种子固定的 N(0, I)"
    if name.startswith("bc_dsrl"):
        return "确定性 actor"
    if name == "planner_e2":
        return "无扩散噪声"
    return ""


def role_label(name: str, primary: str | None) -> str:
    if primary is not None and name == primary:
        return "主结果（事先声明）"
    if name == "bc":
        return "仅作参考"
    return ""


def render_table(rows: list[dict]) -> str:
    columns = [
        "method", "role", "noise", "recorded",
        "n", "n_success", "success_rate", "wilson_95", "failure_counts",
        "ik_fail_ticks_sum", "place_xy_median_success", "duration_s_median",
    ]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if column == "wilson_95" and isinstance(value, list):
                cells.append("[" + ", ".join(_fmt(item) for item in value) + "]")
            elif column == "failure_counts":
                cells.append(json.dumps(value, ensure_ascii=False))
            else:
                cells.append(_fmt(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _paired(left_name: str, right_name: str, maps: dict[str, dict[int, bool]]) -> dict:
    pair = transitions(maps[left_name], maps[right_name])
    improve = len(pair["right_only"])
    regress = len(pair["left_only"])
    pair["left"] = left_name
    pair["right"] = right_name
    pair["improve"] = improve
    pair["regress"] = regress
    pair["mcnemar_p"] = mcnemar_p(improve, regress)
    return pair


def build_report(
    sections: list[tuple[str, dict]],
    m4: tuple[dict, dict[int, bool]] | None = None,
    primary: str | None = None,
) -> dict:
    """Rows in ``sections`` order, then paired tests that do not use the M6 row.

    Paired tests are ``bc_seeded`` and ``bc_zero`` against every ``bc_dsrl*``
    row. ``primary`` is the DSRL label declared ahead of time, usually
    ``bc_dsrl_best`` or ``bc_dsrl_last``.
    """
    rows = []
    maps = {}
    for name, summary in sections:
        row = row_from_eval(name, summary)
        row["noise"] = noise_label(name)
        row["recorded"] = "种子 0–3 录像" if name == "bc" else recorded_label(summary)
        row["role"] = role_label(name, primary)
        rows.append(row)
        maps[name] = success_by_seed(summary)
    if m4 is not None:
        planner = dict(m4[0])
        planner["noise"] = noise_label("planner_e2")
        planner["recorded"] = "规划器轨迹"
        planner["role"] = ""
        rows.append(planner)
        maps["planner_e2"] = m4[1]
    dsrl_names = [name for name, _summary in sections if name.startswith("bc_dsrl")]
    paired = []
    transfers = {}
    for base in ("bc_seeded", "bc_zero"):
        if base not in maps:
            continue
        for name in dsrl_names:
            pair = _paired(base, name, maps)
            paired.append(pair)
            transfers[f"{base}_vs_{name}"] = pair
    return {
        "rows": rows,
        "paired": paired,
        "transfers": transfers,
        "primary": primary,
        "wilson_is_descriptive_only": True,
        "m6_is_not_a_paired_baseline": True,
    }


def write_report(report: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = render_table(report["rows"])
    lines = [
        "Wilson 区间只描述成功率，不作为有无提升的判据。M6 正式评测不作为配对基线。",
        "",
        table,
        "",
        "## 配对比较",
        "",
    ]
    if not report["paired"]:
        lines.append("没有 bc_seeded 或 bc_zero 与 bc_dsrl 的配对。")
        lines.append("")
    for pair in report["paired"]:
        lines.append(f"### {pair['left']} vs {pair['right']}")
        lines.append("")
        lines.append(
            f"比较 {pair['n_compared']} 个种子。"
            f"提升（{pair['right']} 成功且 {pair['left']} 失败）{pair['improve']}。"
            f"退化（{pair['right']} 失败且 {pair['left']} 成功）{pair['regress']}。"
        )
        lines.append("")
        lines.append(f"精确 McNemar（二项）p = {pair['mcnemar_p']:.8g}。")
        lines.append("")
        lines.append(f"- 提升种子: `{pair['right_only']}`")
        lines.append(f"- 退化种子: `{pair['left_only']}`")
        lines.append(f"- 双方成功 ({len(pair['both_success'])}): `{pair['both_success']}`")
        lines.append(f"- 双方失败 ({len(pair['both_fail'])}): `{pair['both_fail']}`")
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "compare.md").write_text(text, encoding="utf-8")
    (out_dir / "compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(text, flush=True)
