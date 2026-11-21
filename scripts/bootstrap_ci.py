#!/usr/bin/env python3
"""Bootstrap Rank-1 / mAP CIs from a per-query ranks CSV or failure_analysis output.

Example::

    python scripts/bootstrap_ci.py --ranks results/failure_analysis/.../per_query.csv
    python scripts/bootstrap_ci.py --hits 1,0,1,1,0 --label demo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.bootstrap import bootstrap_mean, bootstrap_metric_from_ranks, format_ci  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", type=Path, default=None, help="CSV with a rank or rank1 column")
    parser.add_argument("--hits", default=None, help="Comma-separated 0/1 Rank-1 hits")
    parser.add_argument("--column", default=None, help="Column name for ranks or hits")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "bootstrap" / "ci.json")
    args = parser.parse_args()

    if args.hits:
        values = np.array([float(x) for x in args.hits.split(",")], dtype=np.float64)
        result = bootstrap_mean(values, n_boot=args.n_boot, seed=args.seed)
        result["kind"] = "hits"
    elif args.ranks and args.ranks.is_file():
        with args.ranks.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        col = args.column
        if col is None:
            for candidate in ("rank", "rank1", "true_rank", "gt_rank", "full_rank1"):
                if rows and candidate in rows[0]:
                    col = candidate
                    break
        if col is None:
            raise SystemExit(f"No rank column in {args.ranks}; pass --column")
        series = np.array([float(r[col]) for r in rows], dtype=np.float64)
        # full_rank1 style 0/1 vs 1-indexed ranks
        if set(np.unique(series)).issubset({0.0, 1.0}):
            result = bootstrap_mean(series, n_boot=args.n_boot, seed=args.seed)
            result["kind"] = "hits"
        else:
            result = bootstrap_metric_from_ranks(series, n_boot=args.n_boot, seed=args.seed)
            result["kind"] = "ranks"
    else:
        raise SystemExit("Pass --ranks CSV or --hits 0,1,1,...")

    result["label"] = args.label
    result["formatted"] = format_ci(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if args.out.is_file():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
    key = args.label or f"run_{len(payload)}"
    payload[key] = result
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{key}: {result['formatted']} (n={result['n']})")


if __name__ == "__main__":
    main()
