from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def is_video_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def sample_frame_paths(
    crop_paths: Sequence[Path],
    num_frames: int = 8,
) -> list[Path]:
    """
    Uniformly sample ``num_frames`` crop paths.

    If the tracklet is shorter than ``num_frames``, every crop is kept and
    the last frame is repeated so video encoders receive a fixed length.
    """

    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    paths = [Path(path) for path in crop_paths]
    if not paths:
        return []

    if len(paths) == num_frames:
        return paths

    if len(paths) < num_frames:
        return paths + [paths[-1]] * (num_frames - len(paths))

    indices = np.linspace(0, len(paths) - 1, num=num_frames, dtype=np.int64)
    unique = np.unique(indices)
    sampled = [paths[int(index)] for index in unique]
    if len(sampled) < num_frames:
        sampled = sampled + [sampled[-1]] * (num_frames - len(sampled))
    return sampled


def resolve_frame_paths(
    crop_paths: Sequence[Path],
    num_frames: int = 8,
    frame_cache: Path | str | None = None,
) -> list[Path]:
    """Return JPEG paths for a tracklet of crops or a single video file."""

    paths = [Path(path) for path in crop_paths]
    if len(paths) == 1 and is_video_file(paths[0]):
        return sample_video_frame_paths(
            paths[0],
            num_frames=num_frames,
            cache_dir=frame_cache,
        )
    return sample_frame_paths(paths, num_frames=num_frames)


def sample_video_frame_paths(
    video_path: Path | str,
    num_frames: int = 8,
    cache_dir: Path | str | None = None,
) -> list[Path]:
    """
    Uniformly sample ``num_frames`` from an mp4 and cache them as JPEGs.

    Encoders still consume image paths; this is the video adapter.
    """

    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    video_path = Path(video_path)
    if not video_path.is_file():
        return []

    if cache_dir is None:
        cache_dir = video_path.parent / f".frames_{num_frames}"
    cache_root = Path(cache_dir)
    out_dir = cache_root / video_path.parent.parent.name / video_path.stem
    expected = [out_dir / f"frame_{index:03d}.jpg" for index in range(num_frames)]
    if expected and all(path.is_file() for path in expected):
        return expected

    frames = _read_uniform_rgb_frames(video_path, num_frames=num_frames)
    if not frames:
        return []

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, frame in enumerate(frames):
        path = out_dir / f"frame_{index:03d}.jpg"
        Image.fromarray(frame).save(path, quality=95)
        written.append(path)
    return written


def _read_uniform_rgb_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    errors: list[str] = []
    for reader in (_read_frames_cv2, _read_frames_torchvision):
        try:
            frames = reader(video_path, num_frames)
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")
            continue
        if frames:
            return _pad_frames(frames, num_frames)
    joined = "; ".join(errors) if errors else "no decoder available"
    print(f"Could not decode {video_path}: {joined}")
    return []


def _pad_frames(frames: list[np.ndarray], num_frames: int) -> list[np.ndarray]:
    if not frames:
        return []
    if len(frames) >= num_frames:
        return frames[:num_frames]
    return frames + [frames[-1]] * (num_frames - len(frames))


def _target_indices(length: int, num_frames: int) -> list[int]:
    if length <= 0:
        return []
    if length == 1:
        return [0] * num_frames
    indices = np.linspace(0, length - 1, num=num_frames, dtype=np.int64)
    return [int(index) for index in indices]


def _read_frames_cv2(video_path: Path, num_frames: int) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            collected: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                collected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not collected:
                raise RuntimeError(f"OpenCV read 0 frames from {video_path}")
            indices = _target_indices(len(collected), num_frames)
            return [collected[index] for index in indices]

        frames: list[np.ndarray] = []
        for index in _target_indices(total, num_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            raise RuntimeError(f"OpenCV seeked 0 frames from {video_path}")
        return frames
    finally:
        capture.release()


def _read_frames_torchvision(video_path: Path, num_frames: int) -> list[np.ndarray]:
    import torchvision.io

    video, _, _info = torchvision.io.read_video(
        str(video_path),
        pts_unit="sec",
        output_format="THWC",
    )
    if video.numel() == 0:
        raise RuntimeError(f"torchvision read 0 frames from {video_path}")
    array = video.numpy()
    indices = _target_indices(array.shape[0], num_frames)
    return [array[index] for index in indices]
