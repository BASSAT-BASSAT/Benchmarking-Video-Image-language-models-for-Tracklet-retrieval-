#!/usr/bin/env python3
"""Accuracy vs ingestion cost (Pareto) from results/ingestion/sweep.csv.

x = encode ms per frame (log), y = Rank-1 for one pool. Colour = model family
(fixed categorical order), marker = what was varied. Frontier points are
labelled; everything else is identified by the legend and the CSV table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Reference palette, categorical slots in fixed order (blue, orange, aqua, yellow).
FAMILY_COLORS = {"PE-Core": "#2a78d6", "MobileCLIP2": "#eb6834", "SigLIP2": "#1baf7a", "Other": "#eda100"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MARKERS = {"model": "o", "fit": "s", "frames": "^", "resolution": "D", "select": "P"}


def family(model: str) -> str:
    if model.startswith("pe_core"):
        return "PE-Core"
    if model.startswith("mobileclip2"):
        return "MobileCLIP2"
    if model.startswith("siglip"):
        return "SigLIP2"
    return "Other"


def varied(row: dict[str, str], default_frames: int) -> str:
    if row.get("select", "uniform") not in ("", "uniform"):
        return "select"
    if row.get("image_size"):
        return "resolution"
    if int(row.get("max_frames") or default_frames) != default_frames:
        return "frames"
    if row.get("fit", "native") not in ("", "native"):
        return "fit"
    return "model"


def pareto(points: list[tuple[float, float, dict]]) -> list[tuple[float, float, dict]]:
    front, best = [], float("-inf")
    for x, y, row in sorted(points, key=lambda p: (p[0], -p[1])):
        if y > best:
            front.append((x, y, row))
            best = y
    return front


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "ingestion" / "sweep.csv")
    parser.add_argument("--pool", default="global_local", choices=("mean", "query_max", "global_local"))
    parser.add_argument("--metric", default="r1", choices=("r1", "map"))
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ingestion" / "pareto.png")
    parser.add_argument("--default-frames", type=int, default=64)
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    key = f"{args.metric}_{args.pool}"
    # Cost per tracklet, so a smaller frame budget shows up as cheaper.
    points = [
        (1000.0 * float(r["encode_s"]) / float(r["gallery"]), float(r[key]), r)
        for r in rows
        if r.get(key) and r.get("encode_s") and r.get("gallery")
    ]
    if not points:
        raise SystemExit(f"No ok rows with {key} in {args.csv}")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    front = pareto(points)
    ax.plot([p[0] for p in front], [p[1] for p in front], color=INK_2, linewidth=1.0,
            linestyle="--", zorder=1, label="Pareto front")
    for x, y, row in points:
        ax.scatter(x, y, s=46, marker=MARKERS[varied(row, args.default_frames)],
                   color=FAMILY_COLORS[family(row["model"])], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for x, y, row in front:
        ax.annotate(row["tag"], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK)

    ax.set_xscale("log")
    ax.set_xlabel("Ingestion cost (ms per tracklet, log scale; RTX 5070 Laptop, fp16)")
    label = {"r1": "Rank-1", "map": "mAP"}[args.metric]
    ax.set_ylabel(f"{label} (%), {args.pool} pooling")
    title = f"Alexandria text-to-tracklet retrieval: {label} vs ingestion cost"
    if any("battery" in (p[2].get("timing_note") or "") for p in points):
        title += "\nDRAFT: timings measured on battery (power-capped)"
    ax.set_title(title, loc="left", fontsize=10)

    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=f)
               for f, c in FAMILY_COLORS.items() if any(family(p[2]["model"]) == f for p in points)]
    handles += [Line2D([], [], marker=m, linestyle="", color=INK_2, label=f"varied: {k}")
                for k, m in MARKERS.items() if any(varied(p[2], args.default_frames) == k for p in points)]
    handles.append(Line2D([], [], color=INK_2, linestyle="--", label="Pareto front"))
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"Wrote {args.out} (+ .pdf); frontier: {[p[2]['tag'] for p in front]}")


if __name__ == "__main__":
    main()
