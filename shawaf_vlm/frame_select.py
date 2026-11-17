"""Choose which crops of a tracklet to embed under a frame budget.

Strategies (all return paths in temporal order):

- ``uniform``  evenly spaced indices (the original behaviour)
- ``quality``  top-K by sharpness x box area, from a 64-px grey preview
- ``diverse``  greedy farthest-point on 16x32 grey thumbnails, seeded with the
               sharpest crop. Consecutive 10 fps crops are near-duplicates, so
               this spends the budget on poses/views that actually differ.

Previews are decoded with PIL ``draft`` mode (JPEG DCT scaling), which is far
cheaper than a full decode. Cost is charged to the ``decode`` stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

STRATEGIES = ("uniform", "quality", "diverse")
_THUMB = (16, 32)  # (w, h) person aspect


def uniform_indices(n: int, k: int) -> list[int]:
    if k >= n:
        return list(range(n))
    return [int(round(i)) for i in np.linspace(0, n - 1, k)]


def _preview(path: Path) -> tuple[np.ndarray, float, int]:
    """Grey thumbnail, Laplacian-variance sharpness, and original pixel area."""

    with Image.open(path) as image:
        width, height = image.size
        image.draft("L", (64, 128))
        grey = image.convert("L")
        small = np.asarray(grey.resize((64, 128)), dtype=np.float32)
        thumb = np.asarray(grey.resize(_THUMB), dtype=np.float32).ravel()
    lap = (
        -4 * small[1:-1, 1:-1]
        + small[:-2, 1:-1]
        + small[2:, 1:-1]
        + small[1:-1, :-2]
        + small[1:-1, 2:]
    )
    thumb = thumb - thumb.mean()
    norm = float(np.linalg.norm(thumb)) or 1.0
    return thumb / norm, float(lap.var()), width * height


def _previews(paths: Sequence[Path], threads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if threads > 1 and len(paths) > 8:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            rows = list(pool.map(_preview, paths))
    else:
        rows = [_preview(path) for path in paths]
    thumbs = np.stack([r[0] for r in rows])
    sharp = np.array([r[1] for r in rows], dtype=np.float64)
    area = np.array([r[2] for r in rows], dtype=np.float64)
    return thumbs, sharp, area


def _quality_score(sharp: np.ndarray, area: np.ndarray) -> np.ndarray:
    # log keeps one huge box from dominating; both terms scaled to [0, 1]
    def unit(x: np.ndarray) -> np.ndarray:
        x = np.log1p(x)
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 0 else np.ones_like(x)

    return unit(sharp) + unit(area)


def select_frames(
    paths: Sequence[Path],
    k: int,
    strategy: str = "uniform",
    threads: int = 4,
) -> list[Path]:
    """Pick at most ``k`` crops from ``paths`` (temporal order is kept)."""

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    items = [Path(p) for p in paths]
    n = len(items)
    if k < 1:
        raise ValueError("k must be >= 1")
    if n <= k:
        return items
    if strategy == "uniform":
        return [items[i] for i in uniform_indices(n, k)]

    thumbs, sharp, area = _previews(items, threads)
    quality = _quality_score(sharp, area)
    if strategy == "quality":
        chosen = np.argsort(-quality, kind="stable")[:k]
        return [items[i] for i in sorted(chosen.tolist())]

    # diverse: farthest-point sampling in thumbnail space (cosine distance),
    # ties broken toward higher quality.
    sims = thumbs @ thumbs.T
    chosen = [int(np.argmax(quality))]
    min_dist = 1.0 - sims[chosen[0]]
    for _ in range(k - 1):
        score = min_dist + 1e-3 * quality
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - sims[nxt])
    return [items[i] for i in sorted(chosen)]
