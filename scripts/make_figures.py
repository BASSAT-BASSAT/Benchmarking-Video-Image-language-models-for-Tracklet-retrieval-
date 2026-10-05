"""Generate the README figures from the executed Kaggle zero-shot notebooks.

Numbers are the best-setting Rank-1 (and speed) values printed by the three
executed notebooks:

- base group: one session with tvpreid + rstpreid + groot_mot17
- extended_modern: two sessions (groot_mot17 / tvpreid + rstpreid)

Run:  python scripts/make_figures.py   ->  assets/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ASSETS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

# Best-setting Rank-1 per test set, ordered by the 6-set mean (action excluded).
# Columns: PRID, iLIDS, Duke, RSTPReid, GroOT appearance, GroOT action, GroOT combined.
# GME-Qwen2-VL-2B (dagger) ran on TVPReid only; missing sets are NaN.
RANK1 = {
    "IRRA": [66.55, 30.00, 38.06, 53.30, 24.71, 2.20, 29.36],
    "SigLIP 2 So400m": [35.56, 18.67, 21.97, 41.00, 25.58, 2.20, 27.91],
    "Perception Encoder L/14": [38.38, 19.33, 17.66, 37.40, 25.29, 3.30, 29.36],
    "InternVideo2 CLIP-S": [32.39, 13.33, 18.49, 27.00, 19.19, 2.64, 24.71],
    "Qwen3-VL-Embedding-2B *": [24.65, 11.33, 13.18, 30.05, 20.06, 3.52, 22.38],
    "LanguageBind": [16.20, 12.67, 7.30, 15.55, 14.83, 2.20, 16.28],
    "GME-Qwen2-VL-2B \u2020": [11.62, 10.67, 9.45, np.nan, np.nan, np.nan, np.nan],
    "OpenAI CLIP L/14 *": [10.92, 9.33, 5.14, 11.05, 11.92, 1.76, 10.17],
    "Jina CLIP v2 *": [7.75, 11.33, 4.15, 12.15, 11.92, 1.32, 8.14],
    "X-CLIP": [4.93, 5.33, 1.41, 1.40, 2.62, 0.44, 2.91],
    "InternVideo2-1B-s2": [4.23, 4.00, 1.82, 5.15, 0.87, 0.66, 1.45],
}
EXTENDED = {name for name in RANK1 if name.endswith("*") or name.endswith("\u2020")}
COLUMNS = ["PRID", "iLIDS", "Duke", "RSTPReid", "GroOT\nappearance", "GroOT\naction", "GroOT\ncombined"]
MS_PER_CLIP = {
    "IRRA": 35.60,
    "SigLIP 2 So400m": 436.66,
    "Perception Encoder L/14": 248.91,
    "InternVideo2 CLIP-S": 361.04,
    "Qwen3-VL-Embedding-2B *": 342.51,
    "LanguageBind": 200.43,
    "GME-Qwen2-VL-2B \u2020": 1749.55,
    "OpenAI CLIP L/14 *": 106.27,
    "Jina CLIP v2 *": 1328.85,
    "X-CLIP": 31.59,
    "InternVideo2-1B-s2": 267.11,
}


def mean_six(name: str) -> float:
    values = [v for v in RANK1[name][:5] + [RANK1[name][6]] if not np.isnan(v)]
    return float(np.mean(values))


def heatmap() -> None:
    names = sorted(RANK1, key=mean_six, reverse=True)
    matrix = np.ma.masked_invalid(np.array([RANK1[name] for name in names]))
    cmap = matplotlib.colormaps["YlGnBu"].copy()
    cmap.set_bad("#e6e6e6")
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=70)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels(COLUMNS, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if np.ma.is_masked(value):
                ax.text(j, i, "\u2014", ha="center", va="center", fontsize=8, color="0.35")
            else:
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value > 32 else "#1a1a1a",
                )
    for x in (2.5, 3.5):
        ax.axvline(x, color="0.25", lw=1.4)
    ax.set_title("Zero-shot text-to-tracklet Rank-1, best setting per model (Kaggle / Colab T4)", fontsize=11)
    fig.colorbar(image, ax=ax, label="Rank-1 (%)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(ASSETS / "rank1_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def groot_bars() -> None:
    order = sorted((n for n in RANK1 if not np.isnan(RANK1[n][6])), key=lambda n: RANK1[n][6], reverse=True)
    appearance = [RANK1[n][4] for n in order]
    action = [RANK1[n][5] for n in order]
    combined = [RANK1[n][6] for n in order]
    y = np.arange(len(order))[::-1]
    height = 0.26
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.barh(y + height, appearance, height=height, color="#4C72B0", label="appearance")
    ax.barh(y, action, height=height, color="#C44E52", label="action")
    ax.barh(y - height, combined, height=height, color="#55A868", label="combined")
    for values, offset in ((appearance, height), (action, 0.0), (combined, -height)):
        for yy, value in zip(y, values):
            ax.text(value + 0.35, yy + offset, f"{value:.1f}", va="center", fontsize=7.5, color="0.15")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlim(0, 33)
    ax.set_xlabel("Rank-1 (%)")
    ax.set_title("GroOT-MOT17: what the query mentions changes everything", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "groot_caption_types.png", bbox_inches="tight")
    plt.close(fig)


def speed_scatter() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for label, color, names in (
        ("base", "#4C72B0", [n for n in RANK1 if n not in EXTENDED]),
        ("extended (Colab/Kaggle)", "#DD8452", [n for n in RANK1 if n in EXTENDED]),
    ):
        ax.scatter(
            [MS_PER_CLIP[n] for n in names], [mean_six(n) for n in names],
            s=95, color=color, label=label, edgecolor="white", zorder=3,
        )
    short = {
        "IRRA": "IRRA", "SigLIP 2 So400m": "SigLIP 2", "Perception Encoder L/14": "PE L/14",
        "InternVideo2 CLIP-S": "IV2 CLIP-S", "Qwen3-VL-Embedding-2B *": "Qwen3-VL",
        "LanguageBind": "LanguageBind", "OpenAI CLIP L/14 *": "CLIP L/14", "Jina CLIP v2 *": "Jina v2",
        "GME-Qwen2-VL-2B \u2020": "GME-Qwen2", "X-CLIP": "X-CLIP", "InternVideo2-1B-s2": "IV2-1B-s2",
    }
    # (dx, dy, ha) offsets keep the labels off each other.
    offsets = {
        "IRRA": (8, 2, "left"), "X-CLIP": (8, -3, "left"), "InternVideo2-1B-s2": (8, 0, "left"),
        "SigLIP 2 So400m": (8, 2, "left"), "Perception Encoder L/14": (-8, 7, "right"),
        "InternVideo2 CLIP-S": (-8, 6, "right"), "Qwen3-VL-Embedding-2B *": (8, -4, "left"),
        "LanguageBind": (8, 0, "left"), "OpenAI CLIP L/14 *": (8, 0, "left"), "Jina CLIP v2 *": (8, 0, "left"),
        "GME-Qwen2-VL-2B \u2020": (0, 9, "center"),
    }
    for name in RANK1:
        dx, dy, ha = offsets[name]
        ax.annotate(
            short[name], (MS_PER_CLIP[name], mean_six(name)),
            xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5, color="0.1",
        )
    ax.set_xscale("log")
    ax.set_xlim(20, 4400)
    ax.set_ylim(0, 45)
    ax.set_xlabel("ms per 8-frame clip (T4, log scale)")
    ax.set_ylabel("Mean Rank-1 over six non-action test sets")
    ax.set_title("Cost vs accuracy: TVPReid, RSTPReid, and GroOT-MOT17", fontsize=11)
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "speed_accuracy.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    heatmap()
    groot_bars()
    speed_scatter()
    print("wrote", *[p.name for p in sorted(ASSETS.glob("*.png"))])
