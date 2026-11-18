#!/usr/bin/env python3
"""Action gap: does the model use the action words in a query?

Reads ``datasets/alexandria/actions.csv`` (``scripts/split_alexandria_actions.py``)
and, for one encoder config, encodes the gallery once and scores three query sets
against the same embeddings:

- ``full``        the original caption
- ``appearance``  the caption with the action removed
- ``action``      the action alone, as "a person <action>"

Identity metrics (Rank-1/5, mAP) use ``query_max`` and ``global_local`` pooling.
For action-only queries we also report **action accuracy**: the share of top-1
(and top-5) results whose own caption has the same action class, against the
chance rate of that class in the gallery. Per-query, we count how often adding
the action to the appearance moved the true person up or down.

Results: ``results/action_gap/<tag>.json`` and one row per tag in
``results/action_gap/summary.csv``.
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

from shawaf_vlm.action_split import action_query  # noqa: E402
from shawaf_vlm.eval_loop import _group_clip_features, _prepare_window_gallery  # noqa: E402
from shawaf_vlm.metrics import evaluate_from_distmat  # noqa: E402
from shawaf_vlm.pooling import hybrid_global_local_similarity, query_max_similarity  # noqa: E402
from shawaf_vlm.retrieval.datasets import load_splits  # noqa: E402

POOLS = ("query_max", "global_local")
SETS = ("full", "appearance", "action")


def encode_gallery(encoder, splits, max_frames: int, select: str):
    groups, gallery_pids, gallery_camids = _prepare_window_gallery(
        splits.gallery, window=8, stride=4, sample_fps=8.0, max_frames=max_frames, select=select
    )
    flat, owners = [], []
    for index, windows in enumerate(groups):
        for window in windows:
            flat.append(window)
            owners.append(index)
    feats = encoder.encode_videos(flat, batch_size=8)
    return _group_clip_features(feats, owners, len(groups)), gallery_pids, gallery_camids


def similarity(text: np.ndarray, grouped, pool: str) -> np.ndarray:
    if pool == "query_max":
        return query_max_similarity(text, grouped)
    return hybrid_global_local_similarity(text, grouped, alpha=0.5)


def ranks_of_truth(sim: np.ndarray, qpids: np.ndarray, gpids: np.ndarray) -> np.ndarray:
    out = []
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i], kind="stable")
        out.append(int(np.where(gpids[order] == qpids[i])[0][0]) + 1)
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fit", default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--select", default="uniform")
    parser.add_argument("--dataset", default="alexandria:all")
    parser.add_argument("--actions", type=Path, default=ROOT / "datasets" / "alexandria" / "actions.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "action_gap")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tag = f"{args.model}:{args.fit or 'native'}/n{args.max_frames}" + (f"+{args.select}" if args.select != "uniform" else "")
    safe = tag.replace(":", "_").replace("/", "_").replace("+", "_")
    args.out.mkdir(parents=True, exist_ok=True)

    with args.actions.open(encoding="utf-8", newline="") as handle:
        split_rows = {r["track_id"]: r for r in csv.DictReader(handle)}

    splits = load_splits(args.dataset, download=False)
    from shawaf_vlm.models.registry import build_encoder

    encoder = build_encoder(args.model, device=args.device, fit=args.fit)
    grouped, gpids, gcamids = encode_gallery(encoder, splits, args.max_frames, args.select)
    gpids = np.asarray(gpids)
    gcamids = np.asarray(gcamids)
    pid_to_track = {int(p): g.track_id for p, g in zip(gpids, splits.gallery)}
    gallery_class = np.array([split_rows.get(pid_to_track[int(p)], {}).get("action_class", "none") for p in gpids])

    queries = [q for q in splits.query if q.track_id in split_rows]
    texts = {
        # Fixed captions (typos and odd phrasing corrected) are used for all three sets.
        "full": [split_rows[q.track_id].get("caption_fixed") or q.text for q in queries],
        "appearance": [
            split_rows[q.track_id]["appearance"] or split_rows[q.track_id].get("caption_fixed") or q.text
            for q in queries
        ],
        "action": [action_query(split_rows[q.track_id]["action"]) for q in queries],
    }
    has_action = np.array([bool(split_rows[q.track_id]["action"]) for q in queries])
    qpids = np.array([int(q.person_id) for q in queries])
    qcamids = np.array([int(q.camera_id) for q in queries])
    qclass = np.array([split_rows[q.track_id]["action_class"] for q in queries])

    result: dict[str, object] = {"tag": tag, "model": args.model, "fit": args.fit or "native",
                                 "max_frames": args.max_frames, "select": args.select,
                                 "queries": int(len(queries)), "with_action": int(has_action.sum())}
    sims: dict[tuple[str, str], np.ndarray] = {}
    for name in SETS:
        mask = has_action if name == "action" else np.ones(len(queries), dtype=bool)
        idx = np.where(mask)[0]
        feats = np.asarray(encoder.encode_texts([texts[name][i] for i in idx]), dtype=np.float32)
        for pool in POOLS:
            sim = similarity(feats, grouped, pool)
            sims[(name, pool)] = (idx, sim)
            m = evaluate_from_distmat(
                distmat=(1.0 - sim).astype(np.float32),
                query_pids=qpids[idx], gallery_pids=gpids,
                query_camids=qcamids[idx], gallery_camids=gcamids, junk_same_camera=False,
            )
            result[f"{name}_{pool}"] = {k: round(float(m[k]), 2) for k in ("Rank-1", "Rank-5", "Rank-10", "mAP")}

    # Action accuracy for action-only queries: is the top result doing the same action?
    chance = Counter(gallery_class.tolist())
    for pool in POOLS:
        idx, sim = sims[("action", pool)]
        top1, top5, base = [], [], []
        for row, qi in enumerate(idx):
            order = np.argsort(-sim[row], kind="stable")
            top1.append(gallery_class[order[0]] == qclass[qi])
            top5.append(np.mean(gallery_class[order[:5]] == qclass[qi]))
            base.append(chance[qclass[qi]] / len(gallery_class))
        result[f"action_class_{pool}"] = {
            "top1_acc": round(100 * float(np.mean(top1)), 1),
            "top5_precision": round(100 * float(np.mean(top5)), 1),
            "chance": round(100 * float(np.mean(base)), 1),
        }

    # Does adding the action words help the true person rank higher?
    for pool in POOLS:
        _, s_full = sims[("full", pool)]
        _, s_app = sims[("appearance", pool)]
        sel = np.where(has_action)[0]
        r_full = ranks_of_truth(s_full[sel], qpids[sel], gpids)
        r_app = ranks_of_truth(s_app[sel], qpids[sel], gpids)
        result[f"action_effect_{pool}"] = {
            "helped": int((r_full < r_app).sum()),
            "hurt": int((r_full > r_app).sum()),
            "same": int((r_full == r_app).sum()),
            "median_rank_full": float(np.median(r_full)),
            "median_rank_appearance": float(np.median(r_app)),
        }

    (args.out / f"{safe}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    row = {
        "tag": tag,
        "r1_full": result["full_query_max"]["Rank-1"],
        "r1_appearance": result["appearance_query_max"]["Rank-1"],
        "r1_action_only": result["action_query_max"]["Rank-1"],
        "map_full": result["full_query_max"]["mAP"],
        "map_appearance": result["appearance_query_max"]["mAP"],
        "action_top1_acc": result["action_class_query_max"]["top1_acc"],
        "action_top5_prec": result["action_class_query_max"]["top5_precision"],
        "action_chance": result["action_class_query_max"]["chance"],
        "helped": result["action_effect_query_max"]["helped"],
        "hurt": result["action_effect_query_max"]["hurt"],
        "same": result["action_effect_query_max"]["same"],
    }
    summary = args.out / "summary.csv"
    rows = []
    if summary.is_file():
        with summary.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["tag"] != tag]
    rows.append(row)
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
