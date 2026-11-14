"""Plot and summarise M3 logs. Reads outputs/m3/runs and does not resimulate.

Run: conda run -n AIControl python scripts/m3_analyze.py
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json

from aloha_pp.control.analyze import contact_notes, load_directory, plot_all, write_tables
from aloha_pp.robot import repo_root


def main() -> None:
    root = repo_root()
    out = root / "outputs" / "m3"
    loaded = load_directory(out / "runs")
    rows = [row for _arrays, _meta, row in loaded]
    csv_path, md_path = write_tables(rows, out)
    figures = plot_all(loaded, out / "figures")
    notes = contact_notes(loaded)
    (out / "contact_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(f"runs {len(rows)}")
    print(f"csv {csv_path}")
    print(f"markdown {md_path}")
    for path in figures:
        print(f"figure {path}")
    print("contact " + json.dumps(notes))
    for row in rows:
        if row["controller"] == "C3" and row["mode"] in ("fixed_1", "fixed_2"):
            print(
                f"C3 {row['mode']} {row['speed']}: q_rmse {row['q_rmse']:.6g} "
                f"q_max {row['q_max']:.6g} sat {row['sat_ratio']:.6g} "
                f"vel {row['vel_ratio']:.6g} tau_peak {row['tau_peak_ratio']:.3f} "
                f"tcp_ref {row['tcp_ref_rmse']:.6g} tcp_fk {row['tcp_fk_rmse']:.6g}"
            )


if __name__ == "__main__":
    main()
