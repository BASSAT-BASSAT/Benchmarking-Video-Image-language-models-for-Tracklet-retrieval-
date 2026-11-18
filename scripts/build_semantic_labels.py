#!/usr/bin/env python3
"""Build meaning-based relevance labels for Alexandria and a review page.

Reads ``datasets/alexandria/actions.csv`` (fixed captions + action classes) and
writes ``datasets/alexandria/semantic_pairs.csv``: one row per (query, other
person) pair whose automatic label is ``yes`` or ``maybe``. Pairs labelled
``no`` are not written (they are irrelevant by contradiction).

Human review: set ``human`` to ``yes`` or ``no`` (Excel / VS Code). Human
labels win over automatic ones and are kept on re-runs. ``semantic_review.html``
shows each query with its true person and every proposed match (local only:
``datasets/`` is gitignored because it shows faces).

Scoring rule used by ``scripts/rescore.py``: a person is a valid match for a
query if it is the true person, or ``human == yes``, or (``human`` empty and
``auto == yes``).
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.retrieval.datasets import load_splits  # noqa: E402
from shawaf_vlm.semantic_labels import parse_attributes, relevance  # noqa: E402

FIELDS = ("query", "candidate", "auto", "score", "human", "query_caption", "candidate_caption")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="alexandria:all")
    parser.add_argument("--actions", type=Path, default=ROOT / "datasets" / "alexandria" / "actions.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "alexandria" / "semantic_pairs.csv")
    args = parser.parse_args()

    with args.actions.open(encoding="utf-8", newline="") as handle:
        caps = {r["track_id"]: r for r in csv.DictReader(handle)}
    splits = load_splits(args.dataset, download=False)
    tracks = [g.track_id for g in splits.gallery if g.track_id in caps]
    crops = {g.track_id: list(g.crop_paths) for g in splits.gallery}
    attrs = {t: parse_attributes(caps[t]["caption_fixed"], caps[t]["action_class"]) for t in tracks}

    human: dict[tuple[str, str], str] = {}
    if args.out.is_file():
        with args.out.open(encoding="utf-8", newline="") as handle:
            human = {(r["query"], r["candidate"]): r["human"] for r in csv.DictReader(handle) if r.get("human")}

    rows = []
    for q in tracks:
        for g in tracks:
            if g == q:
                continue
            label, score = relevance(attrs[q], attrs[g])
            if label == "no" and (q, g) not in human:
                continue
            rows.append({
                "query": q, "candidate": g, "auto": label, "score": f"{score:.2f}",
                "human": human.get((q, g), ""),
                "query_caption": caps[q]["caption_fixed"], "candidate_caption": caps[g]["caption_fixed"],
            })

    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    per_query_yes = Counter(r["query"] for r in rows if r["auto"] == "yes")
    per_query_maybe = Counter(r["query"] for r in rows if r["auto"] == "maybe")
    with_extra = sum(1 for t in tracks if per_query_yes[t])
    print(f"wrote {args.out}: {len(rows)} pairs "
          f"({sum(per_query_yes.values())} auto-yes, {sum(per_query_maybe.values())} maybe, "
          f"{sum(1 for r in rows if r['human'])} with a human label)")
    print(f"queries with at least one extra valid match: {with_extra}/{len(tracks)}; "
          f"mean extra matches per query: {sum(per_query_yes.values()) / len(tracks):.2f}")

    def strip(track: str, n: int = 3) -> str:
        c = crops.get(track, [])
        picks = [c[int(i)] for i in (0, len(c) // 2, len(c) - 1)][:n] if c else []
        return "".join(f'<img src="{html.escape(p.resolve().as_uri())}" alt="" loading="lazy">' for p in picks)

    blocks = []
    for q in sorted(tracks, key=lambda t: -(per_query_yes[t] + per_query_maybe[t])):
        cands = [r for r in rows if r["query"] == q]
        if not cands:
            continue
        cards = "".join(
            f'<div class="cand {r["auto"]}"><div class="crops">{strip(r["candidate"])}</div>'
            f'<p><b>{html.escape(r["candidate"])}</b> · auto <span class="tag">{r["auto"]}</span>'
            f' · score {r["score"]}{" · human " + html.escape(r["human"]) if r["human"] else ""}</p>'
            f'<p class="cap">{html.escape(r["candidate_caption"])}</p></div>'
            for r in sorted(cands, key=lambda r: (r["auto"] != "yes", -float(r["score"])))
        )
        blocks.append(
            f'<section><h2>{html.escape(caps[q]["caption_fixed"])}</h2>'
            f'<div class="true"><div class="crops">{strip(q)}</div><p><b>{html.escape(q)}</b> · the labelled person</p></div>'
            f'<div class="grid">{cards}</div></section>'
        )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Alexandria meaning-based labels</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px auto; max-width: 1240px; padding: 0 16px; background: #f6f7f8; color: #12181f; }}
section {{ background: #fff; border: 1px solid #d9dee3; border-radius: 10px; padding: 14px; margin: 16px 0; }}
h2 {{ font-size: 16px; margin: 0 0 10px; }}
.crops {{ display: flex; gap: 3px; }} .crops img {{ width: 64px; height: 132px; object-fit: contain; background: #111; border-radius: 3px; }}
.true {{ display: flex; gap: 12px; align-items: center; padding-bottom: 10px; border-bottom: 1px solid #e4e7ea; margin-bottom: 10px; }}
.grid {{ display: grid; gap: 10px; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }}
.cand {{ border: 1px solid #d9dee3; border-radius: 8px; padding: 8px; }} .cand.yes {{ border-color: #0f7a6c; }}
.cand p {{ margin: 4px 0 0; }} .cap {{ color: #47515d; font-size: 13px; }}
.tag {{ font-size: 12px; padding: 1px 6px; border-radius: 99px; background: #eef1f3; }} .yes .tag {{ background: #e6f4f1; color: #0f7a6c; }}
</style>
<h1>Meaning-based labels: which other people also fit each caption?</h1>
<p>Automatic labels from caption attributes. <b>yes</b>: has everything the query asks for. <b>maybe</b>: nothing contradicts it, but the caption does not mention something the query asks for. Confirm or reject in <code>datasets/alexandria/semantic_pairs.csv</code> (column <code>human</code>: yes / no), then re-run <code>python scripts/build_semantic_labels.py</code>.</p>
{''.join(blocks)}</html>"""
    page_path = args.out.with_name("semantic_review.html")
    page_path.write_text(page, encoding="utf-8")
    print(f"review page: {page_path}")


if __name__ == "__main__":
    main()
