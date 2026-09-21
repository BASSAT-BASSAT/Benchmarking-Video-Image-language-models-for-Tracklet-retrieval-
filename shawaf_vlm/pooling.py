from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shawaf_vlm.metrics import l2_normalize

DEFAULT_POOLS = ("mean", "mean_s8", "max", "query_max")


def pool_clip_features(clips: np.ndarray, pool: str) -> np.ndarray:
    """Reduce (K, D) clip embeddings to one L2-normalized (D,) vector."""

    if clips.ndim != 2 or clips.shape[0] < 1:
        raise ValueError(f"Expected clip matrix (K, D), got {clips.shape}")
    vectors = l2_normalize(clips)
    if pool == "mean":
        pooled = vectors.mean(axis=0, keepdims=True)
    elif pool == "max":
        pooled = vectors.max(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown embedding pool {pool!r}")
    return l2_normalize(pooled)[0]


def subsample_windows(
    windows: Sequence[Sequence[object]],
    factor: int,
) -> list:
    """Keep every ``factor``-th window (stride 8 from stride-4 encodes)."""

    if factor <= 1:
        return [list(window) for window in windows]
    if not windows:
        return []
    picked = [windows[index] for index in range(0, len(windows), factor)]
    if picked[-1] is not windows[-1]:
        picked.append(windows[-1])
    return [list(window) for window in picked]


def query_max_similarity(
    text_features: np.ndarray,
    clip_groups: Sequence[np.ndarray],
) -> np.ndarray:
    """(Q, G) cosine: each gallery score is max over that tracklet's clips."""

    text = l2_normalize(text_features)
    columns: list[np.ndarray] = []
    for clips in clip_groups:
        gallery_clips = l2_normalize(clips)
        columns.append((text @ gallery_clips.T).max(axis=1))
    return np.stack(columns, axis=1).astype(np.float32)
