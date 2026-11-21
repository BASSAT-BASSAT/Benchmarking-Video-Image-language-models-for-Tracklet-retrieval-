#!/usr/bin/env python3
"""Propose / apply human labels for Alexandria ``maybe`` semantic pairs.

Meaning-based Rank-1 only counts ``human=yes`` or automatic ``yes``. The 187
``maybe`` rows need a decision. This script:

1. Lists open maybes with the attributes the query states that the candidate
   caption omits.
2. Applies a **documented conservative policy** with ``--apply-soft``
   (see ``shawaf_vlm.semantic_review.propose``).
3. Writes the remaining queue to ``maybe_queue.csv``.

Re-run ``scripts/rescore.py --from-cache`` after editing
``datasets/alexandria/semantic_pairs.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.semantic_labels import parse_attributes, relevance  # noqa: E402
from shawaf_vlm.semantic_review import propose, unknown_attrs  # noqa: E402

FIELDS = ("query", "candidate", "auto", "score", "human", "query_caption", "candidate_caption")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=ROOT / "datasets" / "alexandria" / "semantic_pairs.csv")
    parser.add_argument("--actions", type=Path, default=ROOT / "datasets" / "alexandria" / "actions.csv")
    parser.add_argument("--apply-soft", action="store_true", help="Write soft yes/no into human column")
    parser.add_argument("--force", action="store_true", help="Overwrite existing human labels")
    parser.add_argument("--queue", type=Path, default=ROOT / "datasets" / "alexandria" / "maybe_queue.csv")
    args = parser.parse_args()

    with args.actions.open(encoding="utf-8", newline="") as handle:
        caps = {r["track_id"]: r for r in csv.DictReader(handle)}
    with args.pairs.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    decisions = Counter()
    remaining = []
    for row in rows:
        if row["auto"] != "maybe":
            continue
        if row.get("human") and not args.force:
            decisions["already_human"] += 1
            continue
        q = parse_attributes(caps[row["query"]]["caption_fixed"], caps[row["query"]]["action_class"])
        g = parse_attributes(caps[row["candidate"]]["caption_fixed"], caps[row["candidate"]]["action_class"])
        _label, score = relevance(q, g)
        unknown = unknown_attrs(q, g)
        proposal = propose(score, unknown, candidate_caption=row["candidate_caption"])
        decisions[f"propose_{proposal or 'blank'}"] += 1
        if args.apply_soft and proposal:
            row["human"] = proposal
            decisions["applied"] += 1
        else:
            remaining.append({**row, "unknown": "|".join(unknown), "propose": proposal})

    if args.apply_soft:
        with args.pairs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    with args.queue.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS) + ["unknown", "propose"])
        writer.writeheader()
        writer.writerows(remaining)

    print(dict(decisions))
    print(f"remaining for annotator: {len(remaining)} -> {args.queue}")
    if args.apply_soft:
        print("Re-score: python scripts/rescore.py --from-cache")


if __name__ == "__main__":
    main()
