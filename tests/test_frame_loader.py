"""Frame microbatch backoff, order, and DataLoader path. CPU only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shawaf_vlm.models.frame_encoders import _FrameEncoder


def _save(root: Path, sizes: list[tuple[int, int]]) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (width, height) in enumerate(sizes):
        path = root / f"{index}.jpg"
        Image.new("RGB", (width, height), color=(index, 0, 0)).save(path)
        paths.append(path)
    return paths


class _OomEncoder(_FrameEncoder):
    def __init__(self) -> None:
        super().__init__(device="cpu", frame_microbatch=4, num_workers=0)
        self.seen: list[int] = []

    def _encode_images(self, images):
        self.seen.append(len(images))
        if len(images) > 2:
            raise RuntimeError("CUDA out of memory")
        rows = [torch.tensor([image.size[0], image.size[1]], dtype=torch.float32) for image in images]
        return torch.stack(rows, dim=0)


class _ColorEncoder(_FrameEncoder):
    def __init__(self, microbatch: int) -> None:
        super().__init__(device="cpu", frame_microbatch=microbatch, num_workers=0)

    def _encode_images(self, images):
        rows = []
        for image in images:
            red, green, blue = image.getpixel((0, 0))
            rows.append(torch.tensor([red, green, blue], dtype=torch.float32))
        return torch.stack(rows, dim=0)


class _LoaderEncoder(_FrameEncoder):
    def __init__(self) -> None:
        super().__init__(device="cpu", frame_microbatch=2, num_workers=0)
        self.preprocess = lambda image: torch.tensor(
            [float(image.size[0]), float(image.size[1]), float(image.getpixel((0, 0))[0])],
            dtype=torch.float32,
        )
        self.image_calls = 0

    def _encode_images(self, images):
        self.image_calls += 1
        raise AssertionError("loader path should encode pixels, not PIL batches")

    def _encode_pixels(self, pixels):
        return pixels


def test_oom_backoff_halves_then_preserves_order(tmp_path) -> None:
    sizes = [(index + 1, index + 2) for index in range(5)]
    paths = _save(tmp_path, sizes)
    encoder = _OomEncoder()
    _unique, feats = encoder.encode_frames(paths)
    assert encoder.seen[0] == 4
    assert encoder.seen[1] == 2
    assert encoder.frame_microbatch == 2
    torch.testing.assert_close(feats[:, 0], torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))


def test_microbatch_size_does_not_change_vectors(tmp_path) -> None:
    paths = _save(tmp_path, [(8, 8)] * 6)
    for path, color in zip(paths, range(6), strict=True):
        Image.new("RGB", (8, 8), color=(color, color * 2, 3)).save(path)
    small = _ColorEncoder(1)
    large = _ColorEncoder(7)
    clips = [[path] for path in paths]
    np.testing.assert_allclose(small.encode_videos(clips), large.encode_videos(clips), atol=1e-5)


class _OomLoaderEncoder(_LoaderEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.frame_microbatch = 4
        self.batches: list[int] = []

    def _encode_pixels(self, pixels):
        self.batches.append(int(pixels.shape[0]))
        if pixels.shape[0] > 1:
            raise RuntimeError("CUDA out of memory")
        return pixels


def test_loader_oom_splits_without_reloading_and_keeps_order(tmp_path) -> None:
    paths = _save(tmp_path / "oom", [(index + 2, 3) for index in range(5)])
    encoder = _OomLoaderEncoder()
    _unique, feats = encoder.encode_frames(paths)
    assert encoder.frame_microbatch == 1
    torch.testing.assert_close(feats[:, 0], torch.tensor([2, 3, 4, 5, 6], dtype=torch.float32))
    assert encoder.encoded_frames == 5


def test_loader_keeps_path_order(tmp_path) -> None:
    root = tmp_path / "loader"
    root.mkdir()
    sizes = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
    paths = []
    for index, size in enumerate(sizes):
        path = root / f"{index}.png"
        Image.new("RGB", size, color=(index, 0, 0)).save(path)
        paths.append(path)
    encoder = _LoaderEncoder()
    unique, feats = encoder.encode_frames(paths)
    assert encoder.image_calls == 0
    assert [path.name for path in unique] == [path.name for path in paths]
    got = feats.detach().cpu()
    expected = torch.tensor([[3, 4, 0], [5, 6, 1], [7, 8, 2], [9, 10, 3], [11, 12, 4]], dtype=torch.float32)
    torch.testing.assert_close(got, expected)
