"""Bootstrap confidence intervals for Rank-1 / mAP on small query sets."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_mean(
    per_query: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI for the mean of per-query binary or continuous scores.

    With Alexandria n≈129, one query is ~0.78 Rank-1 points; report the CI
    beside every headline Rank-1 in the paper.
    """

    values = np.asarray(per_query, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("per_query is empty")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = sample.mean()
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1.0 - alpha / 2))
    return {
        "mean": float(values.mean()),
        "low": low,
        "high": high,
        "n": int(values.size),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def bootstrap_metric_from_ranks(
    ranks: np.ndarray,
    *,
    rank_k: int = 1,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap Rank-k from 1-indexed ranks (one per query)."""

    hits = (np.asarray(ranks, dtype=np.int64) <= int(rank_k)).astype(np.float64)
    out = bootstrap_mean(hits, n_boot=n_boot, seed=seed)
    out["rank_k"] = int(rank_k)
    return out


def format_ci(result: dict[str, float], pct: bool = True) -> str:
    scale = 100.0 if pct else 1.0
    return (
        f"{result['mean'] * scale:.1f} "
        f"[{result['low'] * scale:.1f}, {result['high'] * scale:.1f}]"
    )
