#!/usr/bin/env python3
"""Recompute APR / ARS from an existing judged.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import average_pass_rate, average_rubric_score, axis_breakdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    with args.judged.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    summary = {
        "n": len(rows),
        "APR": average_pass_rate(rows),
        "ARS": average_rubric_score(rows),
        "by_axis": axis_breakdown(rows),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
