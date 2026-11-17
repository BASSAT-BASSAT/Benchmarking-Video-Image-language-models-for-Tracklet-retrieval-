"""Frame budget selection: uniform / quality / diverse. CPU only, synthetic crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from shawaf_vlm.frame_select import select_frames, uniform_indices
from shawaf_vlm.sampling import resolve_dense_frame_paths


def _pattern(seed: int, size=(40, 100)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1] // 10, size[0] // 10, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr).resize(size, Image.NEAREST)


def _write(root: Path, images: list[Image.Image]) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = root / f"crop_{i:03d}.jpg"
        img.save(p, quality=95)
        paths.append(p)
    return paths


def test_uniform_indices_spread_and_cap() -> None:
    assert uniform_indices(5, 10) == [0, 1, 2, 3, 4]
    assert uniform_indices(9, 3) == [0, 4, 8]


def test_diverse_picks_one_per_distinct_pose(tmp_path) -> None:
    # 3 "poses", each repeated 6 times in a row (like 10 fps near-duplicates)
    images = [_pattern(pose) for pose in (1, 2, 3) for _ in range(6)]
    paths = _write(tmp_path, images)
    chosen = select_frames(paths, 3, "diverse", threads=1)
    poses = sorted({int(p.stem.split("_")[1]) // 6 for p in chosen})
    assert poses == [0, 1, 2]
    assert chosen == sorted(chosen)  # temporal order kept


def test_quality_prefers_sharp_and_large(tmp_path) -> None:
    sharp = _pattern(7, size=(60, 150))
    blurry = _pattern(7, size=(40, 100)).filter(ImageFilter.GaussianBlur(4))
    paths = _write(tmp_path, [blurry, blurry, sharp, blurry])
    assert select_frames(paths, 1, "quality", threads=1) == [paths[2]]


def test_resolve_dense_uses_selector_only_above_budget(tmp_path) -> None:
    paths = _write(tmp_path, [_pattern(i) for i in range(10)])
    assert resolve_dense_frame_paths(paths, max_frames=20, select="diverse") == paths
    out = resolve_dense_frame_paths(paths, max_frames=4, select="diverse")
    assert len(out) == 4 and set(out) <= set(paths)
    with pytest.raises(ValueError):
        select_frames(paths, 2, "random")
