#!/usr/bin/env python3
"""Timed encode + hard and meaning-based scoring for one encoder config.

Hard scoring: only the labelled person counts (Rank-1/5/10, mAP).
Meaning-based scoring (labels from ``scripts/build_semantic_labels.py``):

- ``sem_r1`` / ``sem_r5``: a valid match is in the top 1 / top 5;
- ``map_at_r``: mAP@R (Chun et al., ECCV Caption 2022), R = number of valid matches;
- ``ndcg5``: nDCG@5 with graded relevance (Wray et al., CVPR 2021): the true person
  scores 1, an automatic or human "yes" scores 1, a "maybe" scores its share of
  matched attributes, a human "no" scores 0.

A person counts as a valid match if it is the true person, or ``human == yes``,
or (``human`` empty and ``auto == yes``). Queries use the reviewed captions
(``caption_fixed`` in ``datasets/alexandria/actions.csv``).

Similarity matrices are cached in ``results/rescore/sims/``; ``--from-cache``
re-scores without the GPU after the labels change.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.eval_loop import _group_clip_features, _prepare_window_gallery  # noqa: E402
from shawaf_vlm.metrics import evaluate_from_distmat  # noqa: E402
from shawaf_vlm.pooling import (  # noqa: E402
    global_mean_similarity,
    hybrid_global_local_similarity,
    query_max_similarity,
)
from shawaf_vlm.retrieval.datasets import load_splits  # noqa: E402

POOLS = ("query_max", "global_local", "mean")
OUT = ROOT / "results" / "rescore"


def tag_of(args) -> str:
    tag = f"{args.model}:{args.fit or 'native'}/n{args.max_frames}"
    return tag + (f"+{args.select}" if args.select != "uniform" else "")


def safe(tag: str) -> str:
    return tag.replace(":", "_").replace("/", "_").replace("+", "_")


def load_labels(actions: Path, pairs: Path):
    with actions.open(encoding="utf-8", newline="") as handle:
        caps = {r["track_id"]: r for r in csv.DictReader(handle)}
    binary: dict[str, set[str]] = {}
    graded: dict[tuple[str, str], float] = {}
    if pairs.is_file():
        with pairs.open(encoding="utf-8", newline="") as handle:
            for r in csv.DictReader(handle):
                h, a, score = r["human"].strip().lower(), r["auto"], float(r["score"])
                valid = h == "yes" or (h == "" and a == "yes")
                if valid:
                    binary.setdefault(r["query"], set()).add(r["candidate"])
                graded[(r["query"], r["candidate"])] = 0.0 if h == "no" else (1.0 if valid else score)
    return caps, binary, graded


def semantic_metrics(sim, q_tracks, g_tracks, binary, graded) -> dict[str, float]:
    r1, r5, apr, ndcg = [], [], [], []
    discounts = 1.0 / np.log2(np.arange(2, 7))
    for i, q in enumerate(q_tracks):
        order = np.argsort(-sim[i], kind="stable")
        ranked = [g_tracks[j] for j in order]
        valid = {q} | binary.get(q, set())
        rel = np.array([1.0 if t in valid else 0.0 for t in ranked])
        r1.append(rel[0])
        r5.append(rel[:5].max())
        R = int(len(valid))
        hits = rel[:R]
        prec = np.cumsum(hits) / np.arange(1, R + 1)
        apr.append(float((prec * hits).sum() / R))
        gains = np.array([1.0 if t == q else graded.get((q, t), 0.0) for t in ranked])
        dcg = float((gains[:5] * discounts).sum())
        ideal = float((np.sort(gains)[::-1][:5] * discounts).sum())
        ndcg.append(dcg / ideal if ideal > 0 else 0.0)
    pct = lambda x: round(100 * float(np.mean(x)), 2)  # noqa: E731
    return {"sem_r1": pct(r1), "sem_r5": pct(r5), "map_at_r": pct(apr), "ndcg5": pct(ndcg),
            "mean_valid_per_query": round(float(np.mean([1 + len(binary.get(q, ())) for q in q_tracks])), 2)}


def score_all(sims, q_tracks, g_tracks, qpids, gpids, binary, graded) -> dict[str, dict]:
    out = {}
    for pool, sim in sims.items():
        hard = evaluate_from_distmat(
            distmat=(1.0 - sim).astype(np.float32), query_pids=qpids, gallery_pids=gpids,
            query_camids=np.zeros_like(qpids), gallery_camids=np.ones_like(gpids), junk_same_camera=False,
        )
        out[pool] = {
            "r1": round(float(hard["Rank-1"]), 2), "r5": round(float(hard["Rank-5"]), 2),
            "r10": round(float(hard["Rank-10"]), 2), "map": round(float(hard["mAP"]), 2),
            **semantic_metrics(sim, q_tracks, g_tracks, binary, graded),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fit", default=None)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--select", default="uniform")
    parser.add_argument("--dataset", default="alexandria:all")
    parser.add_argument("--actions", type=Path, default=ROOT / "datasets" / "alexandria" / "actions.csv")
    parser.add_argument("--pairs", type=Path, default=ROOT / "datasets" / "alexandria" / "semantic_pairs.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-cache", action="store_true", help="Re-score cached similarities (no GPU)")
    args = parser.parse_args()

    tag = tag_of(args)
    (OUT / "sims").mkdir(parents=True, exist_ok=True)
    cache = OUT / "sims" / f"{safe(tag)}.npz"
    caps, binary, graded = load_labels(args.actions, args.pairs)
    timing: dict[str, object] = {}

    if args.from_cache:
        data = np.load(cache, allow_pickle=True)
        q_tracks, g_tracks = list(data["q_tracks"]), list(data["g_tracks"])
        qpids, gpids = data["qpids"], data["gpids"]
        sims = {p: data[p] for p in POOLS if p in data}
        prev = OUT / f"{safe(tag)}.json"
        if prev.is_file():
            timing = json.loads(prev.read_text(encoding="utf-8")).get("timing", {})
    else:
        import torch

        from shawaf_vlm.models.registry import build_encoder

        splits = load_splits(args.dataset, download=False)
        sync = torch.cuda.synchronize if args.device.startswith("cuda") and torch.cuda.is_available() else (lambda: None)
        t0 = time.perf_counter()
        encoder = build_encoder(args.model, device=args.device, fit=args.fit)
        sync()
        t1 = time.perf_counter()
        groups, gpids, _ = _prepare_window_gallery(
            splits.gallery, window=8, stride=4, sample_fps=8.0, max_frames=args.max_frames, select=args.select
        )
        flat, owners = [], []
        for index, windows in enumerate(groups):
            for window in windows:
                flat.append(window)
                owners.append(index)
        frames = len({str(p) for w in flat for p in w})
        feats = encoder.encode_videos(flat, batch_size=8)
        sync()
        t2 = time.perf_counter()
        grouped = _group_clip_features(feats, owners, len(groups))
        g_tracks = [g.track_id for g in splits.gallery]
        queries = [q for q in splits.query if q.track_id in caps]
        q_tracks = [q.track_id for q in queries]
        qpids = np.array([int(q.person_id) for q in queries])
        gpids = np.asarray(gpids)
        text = np.asarray(encoder.encode_texts([caps[t]["caption_fixed"] for t in q_tracks]), dtype=np.float32)
        sync()
        t3 = time.perf_counter()
        sims = {
            "query_max": query_max_similarity(text, grouped),
            "global_local": hybrid_global_local_similarity(text, grouped, alpha=0.5),
            "mean": global_mean_similarity(text, grouped),
        }
        np.savez(cache, q_tracks=np.array(q_tracks), g_tracks=np.array(g_tracks), qpids=qpids, gpids=gpids, **sims)
        timing = {
            "model_load_s": round(t1 - t0, 2), "encode_s": round(t2 - t1, 2), "text_s": round(t3 - t2, 2),
            "people": len(g_tracks), "frames": frames,
            "s_per_person": round((t2 - t1) / len(g_tracks), 4),
            "ms_per_frame": round(1000 * (t2 - t1) / max(frames, 1), 2),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
        }

    scores = score_all(sims, q_tracks, g_tracks, qpids, gpids, binary, graded)
    labelled = sum(1 for (q, _), v in graded.items() if v in (0.0, 1.0))
    result = {"tag": tag, "queries": len(q_tracks), "gallery": len(g_tracks), "timing": timing,
              "labels": {"pairs": len(graded), "queries_with_extra_matches": sum(1 for q in q_tracks if binary.get(q))},
              "scores": scores}
    (OUT / f"{safe(tag)}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    best = scores["query_max"]
    row = {"tag": tag, **{f"{k}": best[k] for k in ("r1", "r5", "r10", "map", "sem_r1", "sem_r5", "map_at_r", "ndcg5")},
           "gl_r1": scores["global_local"]["r1"], "gl_sem_r1": scores["global_local"]["sem_r1"],
           **{k: timing.get(k) for k in ("encode_s", "s_per_person", "ms_per_frame", "frames", "model_load_s", "peak_mem_mb")}}
    summary = OUT / "summary.csv"
    rows = []
    if summary.is_file():
        with summary.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["tag"] != tag]
    rows.append(row)
    tmp = summary.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(summary)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
