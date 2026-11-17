#!/usr/bin/env python3
"""Accuracy vs ingestion-cost sweep on one dataset (default ``alexandria:all``).

Each config runs ``scripts/eval_encoder.py --windows --profile`` in its own
process (clean GPU memory, cold model load) and appends one row per config to a
CSV: Rank-1 / mAP for each pool, wall and forward seconds, frames, ms/frame,
peak memory. Existing rows with the same ``tag`` are skipped, so a sweep can be
stopped and resumed.

Configs are ``model[@size][:fit][/nFRAMES]``, for example::

    pe_core_l14              native geometry, max 64 frames
    pe_core_l14@224          PE-L at 224 px
    mobileclip2_s2:letterbox pad tall crops instead of centre-cropping
    pe_core_b16/n16          16 frames per tracklet
    pe_core_b16/n8+diverse   8 frames chosen by thumbnail diversity
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("mean", "query_max", "global_local")
_CONFIG_RE = re.compile(
    r"^(?P<model>[a-z0-9_]+)(@(?P<size>\d+))?(:(?P<fit>[a-z]+))?(/n(?P<frames>\d+))?(\+(?P<select>[a-z]+))?$"
)

DEFAULT_CONFIGS = (
    "pe_core_l14",
    "pe_core_b16",
    "pe_core_s16",
    "pe_core_t16",
    "mobileclip2_s2",
    "mobileclip2_s4",
    "mobileclip2_b",
)


def parse_config(text: str, default_frames: int) -> dict[str, object]:
    match = _CONFIG_RE.match(text.strip())
    if not match:
        raise ValueError(f"Bad config {text!r}; expected model[@size][:fit][/nFRAMES]")
    size = match.group("size")
    return {
        "tag": text.strip(),
        "model": match.group("model"),
        "image_size": int(size) if size else None,
        "fit": match.group("fit") or "native",
        "max_frames": int(match.group("frames") or default_frames),
        "select": match.group("select") or "uniform",
    }


def run_config(cfg: dict[str, object], args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    safe = re.sub(r"[^a-z0-9_]+", "_", f"{args.dataset}__{cfg['tag']}".lower())
    result_path = out_dir / f"{safe}.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_encoder.py"),
        "--model", str(cfg["model"]),
        "--dataset", args.dataset,
        "--windows",
        "--num-frames", str(args.num_frames),
        "--stride", str(args.stride),
        "--sample-fps", str(args.sample_fps),
        "--max-frames", str(cfg["max_frames"]),
        "--pools", ",".join(POOLS),
        "--fit", str(cfg["fit"]),
        "--select", str(cfg["select"]),
        "--device", args.device,
        "--profile",
        "--output", str(result_path),
    ]
    if cfg["image_size"]:
        cmd += ["--image-size", str(cfg["image_size"])]
    print(">>", " ".join(cmd[1:]), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        (out_dir / f"{safe}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
        # Prefer the exception line over trailing help/URL lines.
        errors = [line for line in lines if "Error" in line or "error:" in line]
        reason = (errors[-1] if errors else (lines[-1] if lines else "unknown"))[-300:]
        return {**cfg, "dataset": args.dataset, "status": f"failed: {reason}"}

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return row_from_payload(cfg, args.dataset, payload)


def result_paths(cfg: dict[str, object], dataset: str, out_dir: Path) -> list[Path]:
    """Current name (dataset-prefixed) first, then the legacy tag-only name."""

    current = re.sub(r"[^a-z0-9_]+", "_", f"{dataset}__{cfg['tag']}".lower())
    legacy = re.sub(r"[^a-z0-9_]+", "_", str(cfg["tag"]).lower())
    return [out_dir / f"{current}.json", out_dir / f"{legacy}.json"]


def row_from_payload(cfg: dict[str, object], dataset: str, payload: dict) -> dict[str, object]:
    metrics = payload.get("metrics", {})
    profile = payload.get("profile") or {}
    stages = profile.get("stages_s", {})
    frames = int(profile.get("frames", 0))
    forward = float(stages.get("forward", 0.0))
    encode = sum(float(stages.get(k, 0.0)) for k in ("decode", "load", "preprocess", "forward", "pool"))
    row: dict[str, object] = {
        **cfg,
        "status": "ok",
        "dataset": dataset,
        "queries": payload.get("num_queries"),
        "gallery": payload.get("num_gallery"),
        "frames": frames,
        "model_load_s": round(float(stages.get("model_load", 0.0)), 2),
        "encode_s": round(encode, 2),
        "forward_s": round(forward, 2),
        "load_preprocess_s": round(float(stages.get("load", 0.0)) + float(stages.get("preprocess", 0.0)), 2),
        "ms_per_frame": round(1000 * encode / frames, 3) if frames else None,
        "tracklets_per_s": round(int(payload.get("num_gallery") or 0) / encode, 3) if encode else None,
        "peak_mem_mb": (profile.get("info") or {}).get("peak_mem_mb"),
    }
    for pool in POOLS:
        m = metrics.get(pool, {})
        row[f"r1_{pool}"] = _metric(m, "rank1", "Rank-1", "r1")
        row[f"map_{pool}"] = _metric(m, "mAP", "map")
    return row


def _metric(m: dict, *names: str):
    for name in names:
        if name in m and m[name] is not None:
            return round(float(m[name]), 2)  # eval JSON already reports percentages
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--dataset", default="alexandria:all")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ingestion" / "sweep.csv")
    parser.add_argument("--rerun", action="store_true", help="Run configs even if their tag is in the CSV")
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Do not run anything; rebuild CSV rows for the given configs from their saved result JSONs",
    )
    args = parser.parse_args()

    out_dir = args.out.parent / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {
        r["tag"] for r in _read(args.out) if r.get("status") == "ok" and r.get("dataset") == args.dataset
    }

    for text in args.configs:
        cfg = parse_config(text, args.max_frames)
        if args.import_only:
            found = next((p for p in result_paths(cfg, args.dataset, out_dir) if p.is_file()), None)
            if found is None:
                print(f"   {cfg['tag']}: no saved result JSON", flush=True)
                continue
            payload = json.loads(found.read_text(encoding="utf-8"))
            if payload.get("dataset") not in (None, args.dataset):
                print(f"   {cfg['tag']}: {found.name} is for {payload.get('dataset')}, skipped", flush=True)
                continue
            row = row_from_payload(cfg, args.dataset, payload)
        else:
            if cfg["tag"] in done and not args.rerun:
                print(f"skip {cfg['tag']} (already in {args.out.name})", flush=True)
                continue
            row = run_config(cfg, args, out_dir)
        _upsert(args.out, row)
        r1 = row.get("r1_global_local")
        print(f"   {cfg['tag']}: {row['status'][:200]}  R1(gl)={r1}  ms/frame={row.get('ms_per_frame')}", flush=True)


def _read(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _upsert(path: Path, row: dict[str, object]) -> None:
    """Re-read the CSV right before writing so concurrent sweeps merge, not overwrite."""

    rows = [
        r for r in _read(path) if not (r.get("tag") == row.get("tag") and r.get("dataset") == row.get("dataset"))
    ]
    rows.append(row)
    tmp = path.with_suffix(".tmp")
    _write(tmp, rows)
    tmp.replace(path)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
