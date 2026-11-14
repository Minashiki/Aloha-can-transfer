"""Compare M6 BC, bc_zero, bc_seeded, one or more DSRL evals, and optionally M4.

The M6 row is labeled as an unfixed-noise reference. Paired tests are
bc_seeded and bc_zero against each DSRL label. Pass ``--selected`` so the
checkpoint named in selected.json is marked 主结果（事先声明）.

Run:
  conda run -n AIControl python -u scripts/m7_compare.py \
    --bc outputs/m6/train/eval/eval.json \
    --bc-zero outputs/m7/eval_bc_zero/eval.json \
    --bc-seeded outputs/m7/eval_bc_seeded/eval.json \
    --dsrl bc_dsrl_best=outputs/m7/eval_bc_dsrl_best/eval.json \
    --dsrl bc_dsrl_last=outputs/m7/eval_bc_dsrl_last/eval.json \
    --selected outputs/m7/train/selected.json \
    --m4 outputs/m4/summary.csv \
    --out outputs/m7/compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloha_pp.rl.compare import build_report, load_eval, row_from_m4, write_report
from aloha_pp.rl.trainer import assert_m7_output
from aloha_pp.robot import repo_root


def _path(root: Path, text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path


def _labeled(text: str) -> tuple[str, str]:
    if "=" not in text:
        return "bc_dsrl", text
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise SystemExit(f"expected NAME=PATH, got {text}")
    return name, path


def _primary(selected_name: str) -> str:
    stem = Path(str(selected_name)).stem
    if stem.startswith("ckpt_"):
        return "bc_dsrl_" + stem[len("ckpt_"):]
    return "bc_dsrl_" + stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", required=True)
    parser.add_argument("--bc-zero", required=True)
    parser.add_argument("--bc-seeded", required=True)
    parser.add_argument(
        "--dsrl",
        action="append",
        required=True,
        help="Repeat for each actor. LABEL=eval.json, for example bc_dsrl_best=outputs/m7/eval_bc_dsrl_best/eval.json",
    )
    parser.add_argument("--selected", default=None, help="selected.json. Its checkpoint is marked 主结果（事先声明）.")
    parser.add_argument("--m4", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = repo_root()
    out = _path(root, args.out)
    assert_m7_output(out, root)
    sections = [
        ("bc", load_eval(_path(root, args.bc))),
        ("bc_zero", load_eval(_path(root, args.bc_zero))),
        ("bc_seeded", load_eval(_path(root, args.bc_seeded))),
    ]
    for item in args.dsrl:
        label, path = _labeled(item)
        sections.append((label, load_eval(_path(root, path))))
    primary = None
    if args.selected:
        chosen = json.loads(_path(root, args.selected).read_text(encoding="utf-8"))["selected"]
        primary = _primary(str(chosen))
    report = build_report(sections, row_from_m4(_path(root, args.m4)) if args.m4 else None, primary=primary)
    if primary is not None and not any(row.get("role") == "主结果（事先声明）" for row in report["rows"]):
        labels = [row["method"] for row in report["rows"]]
        raise SystemExit(f"selected checkpoint maps to {primary}, which is not among {labels}")
    write_report(report, out)


if __name__ == "__main__":
    main()
