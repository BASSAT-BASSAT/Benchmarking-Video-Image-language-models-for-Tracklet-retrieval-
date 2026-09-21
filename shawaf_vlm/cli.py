from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shawaf_vlm.data.tv_mars import CaptionFilesNotFound, load_tv_mars
from shawaf_vlm.eval_loop import evaluate_text_to_tracklet, evaluate_text_to_tracklet_windows
from shawaf_vlm.metrics import format_metrics
from shawaf_vlm.models.registry import all_specs, build_encoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot text-to-tracklet retrieval on TV-MARS "
            "(frozen video-language encoders, no Re-ID fusion)."
        )
    )
    parser.add_argument("--model", required=True, choices=sorted(all_specs()))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="MARS crop root containing bbox_train / bbox_test.",
    )
    parser.add_argument(
        "--ann-root",
        type=Path,
        default=None,
        help=(
            "TV-MARS captions: official root with test_query_info and "
            "test_gallery_info, or a flat folder of JSON files."
        ),
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--windows",
        action="store_true",
        help="Sliding-window clip encode + pooling (mean / max / query_max).",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument(
        "--pools",
        type=str,
        default="mean,mean_s8,max,query_max",
        help="Comma-separated pooling configs when --windows is set.",
    )
    parser.add_argument(
        "--junk-same-camera",
        action="store_true",
        help="Discard same-pid same-camera gallery hits (off by default).",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def default_data_root() -> Path:
    env = os.environ.get("SHAWAF_MARS_ROOT")
    if env:
        return Path(env)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for bbox in kaggle_input.rglob("bbox_train"):
            return bbox.parent
    sibling = (
        PROJECT_ROOT.parent
        / "Person-ReID-BenchMark"
        / "datasets"
        / "MARS"
    )
    if sibling.is_dir():
        return sibling
    return PROJECT_ROOT / "datasets" / "MARS"


def default_ann_root(data_root: Path) -> Path:
    env = os.environ.get("SHAWAF_TV_MARS_ANN")
    if env:
        return Path(env)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for name in ("test_query_info", "partical_dataset"):
            for match in kaggle_input.rglob(name):
                return match.parent if name == "test_query_info" else match
    if (data_root / "test_query_info").is_dir():
        return data_root
    return data_root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = all_specs()[args.model]
    data_root = args.data_root or default_data_root()
    ann_root = args.ann_root or default_ann_root(data_root)
    output = args.output or (
        PROJECT_ROOT / "results" / f"{args.model}_tv_mars.json"
    )

    print(f"Model     : {spec.label} ({args.model})")
    print(f"Checkpoint: {spec.checkpoint}")
    print(f"Crops     : {data_root}")
    print(f"Captions  : {ann_root}")
    print(f"Frames    : {args.num_frames}")
    if args.windows:
        print(f"Windows   : stride={args.stride} fps={args.sample_fps:g} max={args.max_frames}")
        print(f"Pools     : {args.pools}")

    try:
        splits = load_tv_mars(data_root, ann_root)
    except CaptionFilesNotFound as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Queries   : {len(splits.query)}\n"
        f"Gallery   : {len(splits.gallery)}\n"
        f"Split     : {splits.source}"
    )

    encoder = build_encoder(args.model, device=args.device)
    if args.windows:
        pools = tuple(name.strip() for name in args.pools.split(",") if name.strip())
        scored = evaluate_text_to_tracklet_windows(
            encoder=encoder,
            splits=splits,
            num_frames=args.num_frames,
            stride=args.stride,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            pools=pools,
            batch_size=args.batch_size,
            text_batch_size=args.text_batch_size,
            junk_same_camera=args.junk_same_camera,
        )
        print()
        for pool, metrics in scored.items():
            print(f"[{pool}]")
            print(format_metrics(metrics))
            print()
        metrics_payload = scored
    else:
        metrics = evaluate_text_to_tracklet(
            encoder=encoder,
            splits=splits,
            num_frames=args.num_frames,
            batch_size=args.batch_size,
            text_batch_size=args.text_batch_size,
            junk_same_camera=args.junk_same_camera,
        )
        print()
        print(format_metrics(metrics))
        metrics_payload = metrics

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": spec.key,
        "model": spec.label,
        "checkpoint": spec.checkpoint,
        "notes": spec.notes,
        "dataset": "tv-mars",
        "split_source": splits.source,
        "data_root": str(data_root),
        "ann_root": str(ann_root),
        "num_frames": args.num_frames,
        "windows": args.windows,
        "stride": args.stride,
        "sample_fps": args.sample_fps,
        "max_frames": args.max_frames,
        "pools": args.pools,
        "batch_size": args.batch_size,
        "text_batch_size": args.text_batch_size,
        "device": args.device,
        "junk_same_camera": args.junk_same_camera,
        "num_queries": len(splits.query),
        "num_gallery": len(splits.gallery),
        "metrics": metrics_payload,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
