"""Retrieval scoring: mean (global), query_max (local), global_local (Type C)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from shawaf_vlm.metrics import l2_normalize
from shawaf_vlm.pooling import (
    hybrid_global_local_similarity,
    query_max_similarity,
)
from shawaf_vlm.retrieval.encode import TrackletVectors

RetrievalMode = Literal["mean", "query_max", "global_local"]


def score_tracklets(
    query_vector: np.ndarray,
    tracklets: list[TrackletVectors],
    mode: RetrievalMode,
    hybrid_alpha: float = 0.5,
) -> np.ndarray:
    """Return (G,) similarity scores aligned with ``tracklets``."""

    if not tracklets:
        return np.array([], dtype=np.float32)
    q = l2_normalize(query_vector.reshape(1, -1))
    grouped = [t.clip_vectors for t in tracklets]
    if mode == "mean":
        globals = np.stack([t.global_vector for t in tracklets], axis=0)
        globals = l2_normalize(globals)
        return (q @ globals.T).reshape(-1).astype(np.float32)
    if mode == "query_max":
        return query_max_similarity(q, grouped).reshape(-1).astype(np.float32)
    if mode == "global_local":
        return hybrid_global_local_similarity(q, grouped, alpha=hybrid_alpha).reshape(-1).astype(
            np.float32
        )
    raise ValueError(f"Unknown retrieval mode {mode!r}")


def top_k_tracklets(
    query_vector: np.ndarray,
    tracklets: list[TrackletVectors],
    mode: RetrievalMode,
    k: int = 5,
    hybrid_alpha: float = 0.5,
) -> list[tuple[TrackletVectors, float, float, float]]:
    """Top-k results with optional global/local/fused breakdown for Type C."""

    scores = score_tracklets(query_vector, tracklets, mode, hybrid_alpha=hybrid_alpha)
    order = np.argsort(-scores)[:k]
    q = l2_normalize(query_vector.reshape(1, -1))
    results: list[tuple[TrackletVectors, float, float, float]] = []
    for rank in order:
        t = tracklets[int(rank)]
        fused = float(scores[int(rank)])
        g = float((q @ l2_normalize(t.global_vector.reshape(1, -1)).T).item())
        local = float(query_max_similarity(q, [t.clip_vectors]).item())
        results.append((t, fused, g, local))
    return results
