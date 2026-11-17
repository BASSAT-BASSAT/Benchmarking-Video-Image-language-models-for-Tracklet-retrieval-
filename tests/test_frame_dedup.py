"""Unique-frame encoding matches the old per-window mean and encodes less."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shawaf_vlm.models.frame_encoders import _FrameEncoder, _load_frames, mean_pool_clips
from shawaf_vlm.models.runtime import to_numpy
from shawaf_vlm.sampling import sliding_windows


class _PixelEncoder(_FrameEncoder):
    def __init__(self) -> None:
        super().__init__(device="cpu", frame_microbatch=64, num_workers=0)

    def _encode_images(self, images):
        rows = []
        for image in images:
            red, green, blue = image.getpixel((0, 0))
            rows.append(torch.tensor([red, green, blue, 1.0], dtype=torch.float32))
        return torch.stack(rows, dim=0)


def _write_frames(root: Path, count: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = root / f"frame_{index:03d}.jpg"
        Image.new("RGB", (4, 4), color=(index % 255, (index * 3) % 255, 20)).save(path)
        paths.append(path)
    return paths


def _old_encode(encoder: _PixelEncoder, windows: list[list[Path]]) -> np.ndarray:
    images = []
    lengths = []
    for window in windows:
        frames = _load_frames(window)
        images.extend(frames)
        lengths.append(len(frames))
    features = encoder._encode_frame_list(images)
    return to_numpy(mean_pool_clips(features, lengths))


def test_overlapping_windows_match_old_path_and_encode_about_half(tmp_path) -> None:
    paths = _write_frames(tmp_path / "dense", 64)
    windows = sliding_windows(paths, window=8, stride=4)
    fresh = _PixelEncoder()
    got = fresh.encode_videos(windows, batch_size=1)
    reference_encoder = _PixelEncoder()
    expected = _old_encode(reference_encoder, windows)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert fresh.encoded_frames == 64
    assert reference_encoder.encoded_frames == sum(len(window) for window in windows)
    assert fresh.encoded_frames <= reference_encoder.encoded_frames * 0.6


def test_uniform8_encode_count_unchanged(tmp_path) -> None:
    paths = _write_frames(tmp_path / "clip", 8)
    windows = [paths]
    fresh = _PixelEncoder()
    got = fresh.encode_videos(windows)
    reference = _PixelEncoder()
    expected = _old_encode(reference, windows)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert fresh.encoded_frames == reference.encoded_frames == 8
