#!/usr/bin/env python3
"""Build Qdrant + on-disk numpy index for one (model, dataset) pair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shawaf_vlm.models.registry import build_encoder
from shawaf_vlm.retrieval.cache import index_dir, save_numpy_index
from shawaf_vlm.retrieval.datasets import dataset_status, load_splits
from shawaf_vlm.retrieval.tier_a_download import ensure_tier_a_datasets
from shawaf_vlm.retrieval.encode import encode_gallery_tracklets
from shawaf_vlm.retrieval.qdrant_store import get_client, qdrant_url, upsert_tracklets


def default_cache_root() -> Path:
    env = os.environ.get("SHAWAF_INDEX_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "shawaf_vlm" / "indexes"


def main() -> None:
    parser = argparse.ArgumentParser(description="Index gallery tracklets for demo / Qdrant.")
    parser.add_argument("--model", required=True, help="Encoder key from registry")
    parser.add_argument("--dataset", required=True, help="e.g. tvpreid:prid, groot:combined")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--frame-cache", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not fetch missing Tier-A archives before indexing.",
    )
    args = parser.parse_args()

    cache_root = args.cache_root or default_cache_root()
    out_dir = index_dir(cache_root, args.model, args.dataset)

    family = args.dataset.split(":", 1)[0]
    if not args.no_download and family != "local":
        ensure_tier_a_datasets(
            tvpreid=family == "tvpreid",
            groot=family == "groot",
            rstpreid=family == "rstpreid",
            split=args.split,
        )
    else:
        print("Dataset status:", dataset_status())

    splits = load_splits(args.dataset, split=args.split, download=False)
    encoder = build_encoder(args.model, device=args.device)
    tracklets = encode_gallery_tracklets(
        encoder,
        splits,
        num_frames=args.num_frames,
        stride=args.stride,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        frame_cache=args.frame_cache,
    )
    save_numpy_index(out_dir, tracklets, model_key=args.model, dataset_key=args.dataset)

    qdrant_status = "skipped"
    if not args.skip_qdrant:
        try:
            client = get_client()
            upsert_tracklets(client, args.model, args.dataset, tracklets)
            qdrant_status = qdrant_url()
        except Exception as exc:
            qdrant_status = f"failed: {exc}"

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "num_tracklets": len(tracklets),
        "numpy_index": str(out_dir),
        "qdrant": qdrant_status,
        "source": splits.source,
    }
    (out_dir / "index_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
