"""Check an M5 dataset. Exit 1 when a problem is found.

Run: conda run -n AIControl python -u scripts/m5_validate.py outputs/m5/dataset/demos.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aloha_pp.data.dataset import length_stats, validate_dataset
from aloha_pp.robot import repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path if args.path.is_absolute() else repo_root() / args.path
    problems = validate_dataset(path)
    stats = length_stats(path) if path.is_file() else {}
    print(f"file {path}")
    print(f"length_stats {stats}")
    if problems:
        print(f"{len(problems)} problems")
        for item in problems:
            print(f"  {item}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
