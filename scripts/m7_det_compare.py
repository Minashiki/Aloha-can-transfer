"""Compare two videos=none eval trees field by field.

Each tree is ``<root>/<method>/episodes.csv`` plus ``eval.json``. The fields
are the success flag, the failure reason, both IK tick counts and their sum,
place_xy, and duration. Values are compared as written in the csv.

Run:
  conda run -n AIControl python -u scripts/m7_det_compare.py \
    --a outputs/m7/regression/det_a \
    --b outputs/m7/regression/det_b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from aloha_pp.robot import repo_root


FIELDS = (
    "success",
    "failure",
    "ik_current_fail_ticks",
    "ik_preview_fail_ticks",
    "ik_fail_ticks",
    "place_xy",
    "duration_s",
)


def _root(root: Path, text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    return {row["seed"]: row for row in table}


def _runtime(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("runtime") or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--methods", default="bc_zero,bc_seeded,bc_dsrl")
    args = parser.parse_args()
    root = repo_root()
    left_root = _root(root, args.a)
    right_root = _root(root, args.b)
    methods = [part for part in args.methods.split(",") if part]
    mismatches = []
    print(
        "compare process torch "
        f"{torch.__version__} cuda {torch.version.cuda} "
        f"threads {torch.get_num_threads()} interop {torch.get_num_interop_threads()} "
        f"cpu_count {os.cpu_count()}",
        flush=True,
    )
    for method in methods:
        left_csv = left_root / method / "episodes.csv"
        right_csv = right_root / method / "episodes.csv"
        left = _rows(left_csv)
        right = _rows(right_csv)
        left_rt = _runtime(left_root / method / "eval.json")
        right_rt = _runtime(right_root / method / "eval.json")
        print(
            f"{method} A torch {left_rt.get('torch')} cuda {left_rt.get('cuda')} "
            f"driver {left_rt.get('driver')} threads {left_rt.get('torch_num_threads')} "
            f"gpu {left_rt.get('gpu')}",
            flush=True,
        )
        print(
            f"{method} B torch {right_rt.get('torch')} cuda {right_rt.get('cuda')} "
            f"driver {right_rt.get('driver')} threads {right_rt.get('torch_num_threads')} "
            f"gpu {right_rt.get('gpu')}",
            flush=True,
        )
        if set(left) != set(right):
            mismatches.append(f"{method} seeds {sorted(left)} vs {sorted(right)}")
            continue
        for seed in sorted(left, key=lambda item: int(item)):
            for field in FIELDS:
                if left[seed].get(field) != right[seed].get(field):
                    mismatches.append(
                        f"{method} seed {seed} {field}: {left[seed].get(field)!r} vs {right[seed].get(field)!r}"
                    )
        if not any(item.startswith(method + " ") for item in mismatches):
            print(f"{method} {len(left)} episodes match", flush=True)
    if mismatches:
        print("MISMATCH", flush=True)
        for item in mismatches:
            print(item, flush=True)
        raise SystemExit(1)
    print("DET_OK", flush=True)


if __name__ == "__main__":
    main()
