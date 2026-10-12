#!/usr/bin/env python3
"""Download missing Tier-A datasets (TVPReid, GroOT-MOT17, RSTPReid)."""

from __future__ import annotations

import argparse

from shawaf_vlm.retrieval.datasets import dataset_status
from shawaf_vlm.retrieval.tier_a_download import ensure_tier_a_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download missing SHAWAF Tier-A datasets.")
    parser.add_argument("--skip-tvpreid", action="store_true")
    parser.add_argument("--skip-groot", action="store_true")
    parser.add_argument("--skip-rstpreid", action="store_true")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print("Before:", dataset_status())
    results = ensure_tier_a_datasets(
        tvpreid=not args.skip_tvpreid,
        groot=not args.skip_groot,
        rstpreid=not args.skip_rstpreid,
        split=args.split,
    )
    for row in results:
        print(f"{row.name}: {row.action}" + (f" -> {row.path}" if row.path else ""))
    print("After:", dataset_status())


if __name__ == "__main__":
    main()
