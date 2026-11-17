#!/usr/bin/env python3
"""Why does Rank-1 fail? Per-query error analysis for text-to-tracklet retrieval.

Encodes the gallery once (cached as .npz), scores ``query_max`` similarity,
and for every query records the rank of the true tracklet plus what ranked
first. Each miss is put in exactly one bucket, checked in this order
(thresholds fixed in advance, not tuned on the results):

1. ``duplicate_caption``  top-1 has the same normalized caption as the query
2. ``near_duplicate``     caption text cosine >= 0.90 under the same encoder
3. ``same_clip``          top-1 is another person from the same clip
4. ``other_clip``         top-1 is from a different clip

Each miss also gets ``visual_sim`` (cosine between the mean embeddings of the
true and top-1 tracklets). A high value marks a look-alike, or possibly the
same actor filmed in another clip; the contact sheets let you check by eye.

Outputs in ``--out``: ``per_query.csv``, ``summary.json`` and
``contact_sheet_XX.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.eval_loop import _group_clip_features, _prepare_window_gallery  # noqa: E402
from shawaf_vlm.pooling import query_max_similarity  # noqa: E402
from shawaf_vlm.query_text import normalize_query  # noqa: E402
from shawaf_vlm.retrieval.datasets import load_splits  # noqa: E402

NEAR_DUP_COS = 0.90


def clip_of(track_id: str) -> str:
    return track_id.split("/", 1)[0]


def encode_gallery(args, splits, cache: Path):
    if cache.is_file() and not args.recompute:
        data = np.load(cache, allow_pickle=True)
        return list(data["grouped"]), data["gallery_pids"]
    from shawaf_vlm.models.registry import build_encoder

    encoder = build_encoder(args.model, device=args.device, image_size=args.image_size, fit=args.fit)
    groups, gallery_pids, _ = _prepare_window_gallery(
        splits.gallery, window=8, stride=4, sample_fps=8.0, max_frames=args.max_frames
    )
    flat, owners = [], []
    for index, windows in enumerate(groups):
        for window in windows:
            flat.append(window)
            owners.append(index)
    feats = encoder.encode_videos(flat, batch_size=8)
    grouped = _group_clip_features(feats, owners, len(groups))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, grouped=np.array(grouped, dtype=object), gallery_pids=gallery_pids)
    return grouped, gallery_pids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="pe_core_l14")
    parser.add_argument("--dataset", default="alexandria:all")
    parser.add_argument("--fit", default="letterbox")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "failure_analysis")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    tag = f"{args.model}_{args.fit}_n{args.max_frames}_{args.dataset.replace(':', '_')}"
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.dataset, download=False)
    grouped, gallery_pids = encode_gallery(args, splits, out / "gallery_cache.npz")

    from shawaf_vlm.models.registry import build_encoder

    encoder = build_encoder(args.model, device=args.device, image_size=args.image_size, fit=args.fit)
    texts = [q.text for q in splits.query]
    text_feats = np.asarray(encoder.encode_texts(texts), dtype=np.float32)
    sim = query_max_similarity(text_feats, grouped)  # (Q, G)

    # Captions of each gallery tracklet, to compare with the query that beat it.
    pid_to_index = {int(pid): i for i, pid in enumerate(gallery_pids)}
    captions_by_gallery: dict[int, list[str]] = {}
    for q in splits.query:
        captions_by_gallery.setdefault(pid_to_index[int(q.person_id)], []).append(q.text)
    gallery = splits.gallery
    means = np.stack([g.mean(axis=0) / np.linalg.norm(g.mean(axis=0)) for g in grouped])
    text_norm = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)
    caption_feat = {i: text_norm[k] for k, q in enumerate(splits.query) for i in [pid_to_index[int(q.person_id)]]}

    rows = []
    for qi, q in enumerate(splits.query):
        gt = pid_to_index[int(q.person_id)]
        order = np.argsort(-sim[qi], kind="stable")
        rank = int(np.where(order == gt)[0][0]) + 1
        top1 = int(order[0])
        row = {
            "query": q.text,
            "gt_track": gallery[gt].track_id,
            "rank": rank,
            "gt_score": round(float(sim[qi, gt]), 4),
            "top1_track": gallery[top1].track_id,
            "top1_score": round(float(sim[qi, top1]), 4),
            "margin": round(float(sim[qi, top1] - sim[qi, gt]), 4),
            "top1_caption": " | ".join(captions_by_gallery.get(top1, [])),
            "bucket": "hit",
            "caption_cos": "",
            "visual_sim": "",
        }
        if rank > 1:
            top_caps = captions_by_gallery.get(top1, [])
            same_text = any(normalize_query(c) == normalize_query(q.text) for c in top_caps)
            cap_cos = float(text_norm[qi] @ caption_feat[top1]) if top1 in caption_feat else 0.0
            if same_text:
                bucket = "duplicate_caption"
            elif cap_cos >= NEAR_DUP_COS:
                bucket = "near_duplicate"
            elif clip_of(gallery[top1].track_id) == clip_of(gallery[gt].track_id):
                bucket = "same_clip"
            else:
                bucket = "other_clip"
            row.update(
                bucket=bucket,
                caption_cos=round(cap_cos, 3),
                visual_sim=round(float(means[gt] @ means[top1]), 3),
            )
        rows.append(row)

    with (out / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    misses = [r for r in rows if r["rank"] > 1]
    n = len(rows)
    buckets = Counter(r["bucket"] for r in misses)
    vis = np.array([r["visual_sim"] for r in misses], dtype=float) if misses else np.zeros(0)
    summary = {
        "config": tag,
        "queries": n,
        "rank1": round(100 * (n - len(misses)) / n, 2),
        "rank5": round(100 * sum(r["rank"] <= 5 for r in rows) / n, 2),
        "misses": len(misses),
        "miss_buckets": dict(buckets),
        "miss_buckets_pct_of_queries": {k: round(100 * v / n, 1) for k, v in buckets.items()},
        "misses_rank_2_to_5": sum(2 <= r["rank"] <= 5 for r in misses),
        "misses_rank_gt_5": sum(r["rank"] > 5 for r in misses),
        "close_calls_margin_lt_0.01": sum(r["margin"] < 0.01 for r in misses),
        "visual_sim_of_misses": {
            "median": round(float(np.median(vis)), 3) if vis.size else None,
            "share_ge_0.9": round(float((vis >= 0.9).mean()), 3) if vis.size else None,
        },
        "ceiling_if_caption_dups_unfixable": round(
            100 * (n - buckets.get("duplicate_caption", 0) - buckets.get("near_duplicate", 0)) / n, 1
        ),
        "thresholds": {"near_duplicate_caption_cos": NEAR_DUP_COS},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not args.no_sheets:
        write_contact_sheets(misses, {g.track_id: g for g in gallery}, out)


def write_contact_sheets(misses, by_track, out: Path, per_page: int = 10) -> None:
    from PIL import Image, ImageDraw

    thumb_w, thumb_h, text_w = 90, 180, 560
    misses = sorted(misses, key=lambda r: (r["bucket"], -float(r["visual_sim"] or 0)))
    for page in range(0, len(misses), per_page):
        chunk = misses[page : page + per_page]
        sheet = Image.new("RGB", (text_w + 2 * (3 * thumb_w + 10), thumb_h * len(chunk)), (252, 252, 251))
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            y = i * thumb_h
            for col, track in enumerate((r["gt_track"], r["top1_track"])):
                crops = by_track[track].crop_paths
                picks = [crops[int(k)] for k in np.linspace(0, len(crops) - 1, 3)]
                for j, path in enumerate(picks):
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((thumb_w, thumb_h))
                        sheet.paste(im, (text_w + col * (3 * thumb_w + 10) + j * thumb_w, y))
            lines = [
                f"[{r['bucket']}] rank {r['rank']}  margin {r['margin']}  vis {r['visual_sim']}",
                f"QUERY : {r['query'][:80]}",
                f"GT    : {r['gt_track']}",
                f"TOP-1 : {r['top1_track']}",
                f"its caption: {r['top1_caption'][:70]}",
                "left 3 = true person, right 3 = what ranked first",
            ]
            for k, line in enumerate(lines):
                draw.text((6, y + 6 + 16 * k), line, fill=(11, 11, 11))
            draw.line([(0, y + thumb_h - 1), (sheet.width, y + thumb_h - 1)], fill=(200, 200, 200))
        sheet.save(out / f"contact_sheet_{page // per_page + 1:02d}.png")


if __name__ == "__main__":
    main()
