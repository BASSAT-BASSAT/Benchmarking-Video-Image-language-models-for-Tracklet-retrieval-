"""Generate README figures from Tier-A checkpoint numbers (top models only).

Run:  python scripts/make_figures.py   ->  assets/*.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
GEMMA_CSV = ROOT / "results" / "checkpoints" / "embeddinggemma2_tier_a_v1.csv"
ASSETS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

# Best-setting Rank-1 per column (PRID, iLIDS, Duke, RSTP, GroOT app/action/combined).
RANK1: dict[str, list[float]] = {
    "IRRA": [66.55, 30.00, 38.06, 53.30, 24.71, 2.20, 29.36],
    "Perception Encoder L/14": [38.38, 19.33, 17.66, 37.40, 25.29, 3.30, 29.36],
    "SigLIP 2 So400m": [35.56, 18.67, 21.97, 41.00, 25.58, 2.20, 27.91],
    "InternVideo2 CLIP-S": [32.39, 13.33, 18.49, 27.00, 19.19, 2.64, 24.71],
}
EXTENDED: set[str] = set()
COLUMNS = ["PRID", "iLIDS", "Duke", "RSTPReid", "GroOT\nappearance", "GroOT\naction", "GroOT\ncombined"]
MS_PER_CLIP = {
    "IRRA": 35.60,
    "SigLIP 2 So400m": 436.66,
    "Perception Encoder L/14": 248.91,
    "InternVideo2 CLIP-S": 361.04,
}

_GEMMA_KEYS = (
    ("tvpreid", "prid"),
    ("tvpreid", "ilids"),
    ("tvpreid", "duke"),
    ("rstpreid", "test"),
    ("groot_mot17", "appearance"),
    ("groot_mot17", "action"),
    ("groot_mot17", "combined"),
)


def _load_embeddinggemma_rank1() -> list[float]:
    if not GEMMA_CSV.is_file():
        raise FileNotFoundError(f"Missing {GEMMA_CSV}")
    bests: dict[tuple[str, str], float] = {}
    with GEMMA_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["subset"])
            value = float(row["Rank-1"])
            bests[key] = max(bests.get(key, value), value)
    return [bests[key] for key in _GEMMA_KEYS]


def _merge_embeddinggemma() -> None:
    label = "EmbeddingGemma 2 *"
    RANK1[label] = _load_embeddinggemma_rank1()
    EXTENDED.add(label)


def mean_six(name: str) -> float:
    values = [v for v in RANK1[name][:5] + [RANK1[name][6]] if not np.isnan(v)]
    return float(np.mean(values))


def heatmap() -> None:
    names = sorted(RANK1, key=mean_six, reverse=True)
    matrix = np.array([RANK1[name] for name in names])
    cmap = matplotlib.colormaps["YlGnBu"]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=70)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels(COLUMNS, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 32 else "#1a1a1a",
            )
    for x in (2.5, 3.5):
        ax.axvline(x, color="0.25", lw=1.4)
    ax.set_title("Zero-shot Rank-1 (%) — top models, best protocol per cell", fontsize=11)
    fig.colorbar(image, ax=ax, label="Rank-1 (%)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(ASSETS / "rank1_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def groot_bars() -> None:
    order = sorted(RANK1, key=lambda n: RANK1[n][6], reverse=True)
    appearance = [RANK1[n][4] for n in order]
    action = [RANK1[n][5] for n in order]
    combined = [RANK1[n][6] for n in order]
    y = np.arange(len(order))[::-1]
    height = 0.26
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.barh(y + height, appearance, height=height, color="#4C72B0", label="appearance")
    ax.barh(y, action, height=height, color="#C44E52", label="action")
    ax.barh(y - height, combined, height=height, color="#55A868", label="combined")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(0, 33)
    ax.set_xlabel("Rank-1 (%)")
    ax.set_title("GroOT-MOT17 by caption type", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "groot_caption_types.png", bbox_inches="tight")
    plt.close(fig)


def speed_scatter() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = list(RANK1.keys())
    ax.scatter(
        [MS_PER_CLIP[n] for n in names if n in MS_PER_CLIP],
        [mean_six(n) for n in names if n in MS_PER_CLIP],
        s=100,
        color="#4C72B0",
        edgecolor="white",
        zorder=3,
    )
    short = {
        "IRRA": "IRRA",
        "SigLIP 2 So400m": "SigLIP 2",
        "Perception Encoder L/14": "PE L/14",
        "InternVideo2 CLIP-S": "IV2 CLIP-S",
        "EmbeddingGemma 2 *": "Emb. Gemma 2",
    }
    for name in names:
        if name not in MS_PER_CLIP:
            continue
        ax.annotate(
            short.get(name, name),
            (MS_PER_CLIP[name], mean_six(name)),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_xlim(25, 500)
    ax.set_ylim(15, 45)
    ax.set_xlabel("ms per 8-frame clip (T4, log scale)")
    ax.set_ylabel("Mean Rank-1 (six non-action Tier-A sets)")
    ax.set_title("Speed vs accuracy (base group + IRRA)", fontsize=11)
    ax.grid(True, which="both", axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "speed_accuracy.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _merge_embeddinggemma()
    heatmap()
    groot_bars()
    speed_scatter()
    print("wrote", *[p.name for p in sorted(ASSETS.glob("*.png"))])
