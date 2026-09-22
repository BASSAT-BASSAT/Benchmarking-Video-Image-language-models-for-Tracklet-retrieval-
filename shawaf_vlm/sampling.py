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
    sample: str = "uniform",
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

    indices = _frame_indices(len(paths), num_frames, sample)
    unique = np.unique(indices)
    sampled = [paths[int(index)] for index in unique]
    if len(sampled) < num_frames:
        sampled = sampled + [sampled[-1]] * (num_frames - len(sampled))
    return sampled


def resolve_frame_paths(
    crop_paths: Sequence[Path],
    num_frames: int = 8,
    frame_cache: Path | str | None = None,
    sample: str = "uniform",
) -> list[Path]:
    """Return JPEG paths for a tracklet of crops or a single video file."""

    paths = [Path(path) for path in crop_paths]
    if len(paths) == 1 and is_video_file(paths[0]):
        return sample_video_frame_paths(
            paths[0],
            num_frames=num_frames,
            cache_dir=frame_cache,
            sample=sample,
        )
    return sample_frame_paths(paths, num_frames=num_frames, sample=sample)


def sample_video_frame_paths(
    video_path: Path | str,
    num_frames: int = 8,
    cache_dir: Path | str | None = None,
    sample: str = "uniform",
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
    sample_name = _normalize_sample(sample)
    if sample_name == "uniform":
        out_dir = cache_root / video_path.parent.parent.name / video_path.stem
    else:
        out_dir = cache_root / sample_name / video_path.parent.parent.name / video_path.stem
    expected = [out_dir / f"frame_{index:03d}.jpg" for index in range(num_frames)]
    if expected and all(path.is_file() for path in expected):
        return expected

    frames = _read_uniform_rgb_frames(
        video_path,
        num_frames=num_frames,
        sample=sample_name,
    )
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


def _normalize_sample(sample: str) -> str:
    key = sample.strip().lower()
    if key not in {"uniform", "middle"}:
        raise ValueError(f"Unknown frame sample {sample!r}. Use uniform or middle.")
    return key


def middle_frame_indices(length: int, num_frames: int) -> list[int]:
    """Pick the midpoint of each equal interval, matching InternVideo2's middle sample."""

    if length <= 0:
        return []
    if length == 1:
        return [0] * num_frames
    count = min(num_frames, length)
    edges = np.linspace(0, length, num=count + 1).astype(int)
    indices: list[int] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        end = int(stop) - 1
        begin = int(start)
        if end < begin:
            end = begin
        indices.append((begin + end) // 2)
    if len(indices) < num_frames:
        indices = indices + [indices[-1]] * (num_frames - len(indices))
    return indices


def _frame_indices(length: int, num_frames: int, sample: str = "uniform") -> list[int]:
    kind = _normalize_sample(sample)
    if kind == "middle":
        return middle_frame_indices(length, num_frames)
    return _target_indices(length, num_frames)


def _read_uniform_rgb_frames(
    video_path: Path,
    num_frames: int,
    sample: str = "uniform",
) -> list[np.ndarray]:
    errors: list[str] = []
    for reader in (_read_frames_cv2, _read_frames_torchvision):
        try:
            frames = reader(video_path, num_frames, sample=sample)
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


def _read_frames_cv2(
    video_path: Path,
    num_frames: int,
    sample: str = "uniform",
) -> list[np.ndarray]:
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
            indices = _frame_indices(len(collected), num_frames, sample)
            return [collected[index] for index in indices]

        frames: list[np.ndarray] = []
        for index in _frame_indices(total, num_frames, sample):
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


def _read_frames_torchvision(
    video_path: Path,
    num_frames: int,
    sample: str = "uniform",
) -> list[np.ndarray]:
    import torchvision.io

    video, _, _info = torchvision.io.read_video(
        str(video_path),
        pts_unit="sec",
        output_format="THWC",
    )
    if video.numel() == 0:
        raise RuntimeError(f"torchvision read 0 frames from {video_path}")
    array = video.numpy()
    indices = _frame_indices(array.shape[0], num_frames, sample)
    return [array[index] for index in indices]


def sliding_windows(
    paths: Sequence[Path],
    window: int = 8,
    stride: int = 4,
) -> list[list[Path]]:
    """Split a dense frame list into overlapping ``window``-frame clips."""

    if window < 1:
        raise ValueError("window must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    frames = [Path(path) for path in paths]
    if not frames:
        return []
    if len(frames) <= window:
        return [sample_frame_paths(frames, num_frames=window)]

    last_start = len(frames) - window
    starts: list[int] = []
    start = 0
    while start < last_start:
        starts.append(start)
        start += stride
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [frames[begin : begin + window] for begin in starts]


def resolve_dense_frame_paths(
    crop_paths: Sequence[Path],
    sample_fps: float = 2.0,
    max_frames: int = 32,
    frame_cache: Path | str | None = None,
) -> list[Path]:
    """Decode a tracklet densely enough for sliding windows, cap at max_frames."""

    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    paths = [Path(path) for path in crop_paths]
    if not paths:
        return []
    if len(paths) == 1 and is_video_file(paths[0]):
        return sample_video_frames_by_fps(
            paths[0],
            sample_fps=sample_fps,
            max_frames=max_frames,
            cache_dir=frame_cache,
        )
    if len(paths) > max_frames:
        return sample_frame_paths(paths, num_frames=max_frames)
    return paths


def sample_video_frames_by_fps(
    video_path: Path | str,
    sample_fps: float = 2.0,
    max_frames: int = 32,
    cache_dir: Path | str | None = None,
) -> list[Path]:
    """Sample ~``sample_fps`` frames, covering the whole video, at most ``max_frames``."""

    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    video_path = Path(video_path)
    if not video_path.is_file():
        return []

    if cache_dir is None:
        cache_dir = video_path.parent / f".frames_fps{sample_fps:g}_n{max_frames}"
    out_dir = (
        Path(cache_dir)
        / f"fps{sample_fps:g}_n{max_frames}"
        / video_path.parent.parent.name
        / video_path.stem
    )
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing:
        return existing

    frames = _read_fps_rgb_frames(
        video_path, sample_fps=sample_fps, max_frames=max_frames
    )
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


def _read_fps_rgb_frames(
    video_path: Path,
    sample_fps: float,
    max_frames: int,
) -> list[np.ndarray]:
    errors: list[str] = []
    for reader in (_read_fps_frames_cv2, _read_fps_frames_torchvision):
        try:
            frames = reader(video_path, sample_fps, max_frames)
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")
            continue
        if frames:
            return frames
    joined = "; ".join(errors) if errors else "no decoder available"
    print(f"Could not decode {video_path}: {joined}")
    return []


def _fps_target_indices(
    length: int, src_fps: float, sample_fps: float, max_frames: int
) -> list[int]:
    if length <= 0:
        return []
    if length == 1:
        return [0]
    fps = src_fps if src_fps > 1e-3 else 25.0
    duration = length / fps
    count = int(round(duration * sample_fps))
    count = max(1, min(int(max_frames), count, length))
    return _target_indices(length, count)


def _read_fps_frames_cv2(
    video_path: Path, sample_fps: float, max_frames: int
) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        src_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0:
            collected: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                collected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not collected:
                raise RuntimeError(f"OpenCV read 0 frames from {video_path}")
            indices = _fps_target_indices(
                len(collected), src_fps, sample_fps, max_frames
            )
            return [collected[index] for index in indices]

        frames: list[np.ndarray] = []
        for index in _fps_target_indices(total, src_fps, sample_fps, max_frames):
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


def _read_fps_frames_torchvision(
    video_path: Path, sample_fps: float, max_frames: int
) -> list[np.ndarray]:
    import torchvision.io

    video, _, info = torchvision.io.read_video(
        str(video_path),
        pts_unit="sec",
        output_format="THWC",
    )
    if video.numel() == 0:
        raise RuntimeError(f"torchvision read 0 frames from {video_path}")
    array = video.numpy()
    src_fps = 0.0
    if isinstance(info, dict):
        src_fps = float(info.get("video_fps") or 0.0)
    indices = _fps_target_indices(array.shape[0], src_fps, sample_fps, max_frames)
    return [array[index] for index in indices]
