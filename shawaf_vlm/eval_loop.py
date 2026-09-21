from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits
from shawaf_vlm.metrics import evaluate_from_distmat, evaluate_text_retrieval
from shawaf_vlm.models.protocol import VideoTextEncoder
from shawaf_vlm.pooling import (
    DEFAULT_POOLS,
    pool_clip_features,
    query_max_similarity,
)
from shawaf_vlm.sampling import resolve_dense_frame_paths, resolve_frame_paths, sliding_windows


def evaluate_text_to_tracklet(
    encoder: VideoTextEncoder,
    splits: TVMarsSplits,
    num_frames: int = 8,
    batch_size: int = 4,
    text_batch_size: int = 32,
    junk_same_camera: bool = False,
    frame_cache: Path | str | None = None,
) -> dict[str, float]:
    """Encode gallery tracklets and caption queries, then rank by cosine."""

    t0 = time.perf_counter()
    gallery_videos, gallery_pids, gallery_camids = _prepare_gallery(
        splits.gallery,
        num_frames=num_frames,
        frame_cache=frame_cache,
    )
    query_texts, query_pids, query_camids = _prepare_queries(splits.query)
    decode_s = time.perf_counter() - t0
    if not gallery_videos:
        raise RuntimeError("No gallery tracklets with existing crop files.")
    if not query_texts:
        raise RuntimeError("No caption queries with existing crop files.")

    _gpu_reset_peak()
    t_video = time.perf_counter()
    video_features = encoder.encode_videos(
        gallery_videos,
        batch_size=batch_size,
    )
    _cuda_sync()
    video_s = time.perf_counter() - t_video
    t_text = time.perf_counter()
    text_features = encoder.encode_texts(
        query_texts,
        batch_size=text_batch_size,
    )
    _cuda_sync()
    text_s = time.perf_counter() - t_text
    peak_gpu_gb, reserved_gpu_gb = _gpu_memory_gb()
    if video_features.ndim != 2 or text_features.ndim != 2:
        raise RuntimeError(
            "Encoder must return 2-D embeddings; "
            f"got video {video_features.shape} and text {text_features.shape}."
        )
    if video_features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            "Video and text embedding sizes do not match: "
            f"{video_features.shape[1]} vs {text_features.shape[1]}."
        )
    if video_features.shape[0] != len(gallery_pids):
        raise RuntimeError(
            f"Encoded {video_features.shape[0]} videos for "
            f"{len(gallery_pids)} gallery tracklets."
        )
    if text_features.shape[0] != len(query_pids):
        raise RuntimeError(
            f"Encoded {text_features.shape[0]} texts for "
            f"{len(query_pids)} queries."
        )

    t_score = time.perf_counter()
    metrics = evaluate_text_retrieval(
        query_features=text_features,
        gallery_features=video_features,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        junk_same_camera=junk_same_camera,
    )
    score_s = time.perf_counter() - t_score
    _attach_runtime(
        metrics,
        decode_s=decode_s,
        video_s=video_s,
        text_s=text_s,
        score_s=score_s,
        n_video=len(gallery_videos),
        peak_gpu_gb=peak_gpu_gb,
        reserved_gpu_gb=reserved_gpu_gb,
    )
    return metrics


def evaluate_text_to_tracklet_windows(
    encoder: VideoTextEncoder,
    splits: TVMarsSplits,
    num_frames: int = 8,
    stride: int = 4,
    sample_fps: float = 2.0,
    max_frames: int = 32,
    pools: Sequence[str] = DEFAULT_POOLS,
    batch_size: int = 4,
    text_batch_size: int = 32,
    junk_same_camera: bool = False,
    frame_cache: Path | str | None = None,
) -> dict[str, dict[str, float]]:
    """Encode sliding-window clips once, then score several pooling configs.

    ``mean`` / ``max`` average or max-pool clip vectors (CLIP4Clip-style).
    ``mean_s8`` keeps every other stride-4 window (approx. stride 8).
    ``query_max`` scores max clip–text cosine (late interaction / X-Pool-lite).
    """

    t0 = time.perf_counter()
    clip_groups, gallery_pids, gallery_camids = _prepare_window_gallery(
        splits.gallery,
        window=num_frames,
        stride=stride,
        sample_fps=sample_fps,
        max_frames=max_frames,
        frame_cache=frame_cache,
    )
    query_texts, query_pids, query_camids = _prepare_queries(splits.query)
    decode_s = time.perf_counter() - t0
    if not clip_groups:
        raise RuntimeError("No gallery tracklets with existing crop files.")
    if not query_texts:
        raise RuntimeError("No caption queries with existing crop files.")

    flat_windows: list[list[Path]] = []
    owners: list[int] = []
    for track_index, windows in enumerate(clip_groups):
        for window in windows:
            flat_windows.append(window)
            owners.append(track_index)

    print(
        f"Window encode: {len(clip_groups)} tracklets, {len(flat_windows)} clips "
        f"(window={num_frames} stride={stride} fps={sample_fps:g} max={max_frames})",
        flush=True,
    )
    _gpu_reset_peak()
    t_video = time.perf_counter()
    clip_features = encoder.encode_videos(flat_windows, batch_size=batch_size)
    _cuda_sync()
    video_s = time.perf_counter() - t_video
    t_text = time.perf_counter()
    text_features = encoder.encode_texts(query_texts, batch_size=text_batch_size)
    _cuda_sync()
    text_s = time.perf_counter() - t_text
    peak_gpu_gb, reserved_gpu_gb = _gpu_memory_gb()
    grouped = _group_clip_features(clip_features, owners, len(clip_groups))

    scored: dict[str, dict[str, float]] = {}
    for pool in pools:
        t_score = time.perf_counter()
        if pool == "query_max":
            similarity = query_max_similarity(text_features, grouped)
            metrics = evaluate_from_distmat(
                distmat=(1.0 - similarity).astype(np.float32),
                query_pids=query_pids,
                gallery_pids=gallery_pids,
                query_camids=query_camids,
                gallery_camids=gallery_camids,
                junk_same_camera=junk_same_camera,
            )
        else:
            gallery_features = _pooled_gallery(grouped, pool=pool, encode_stride=stride)
            metrics = evaluate_text_retrieval(
                query_features=text_features,
                gallery_features=gallery_features,
                query_pids=query_pids,
                gallery_pids=gallery_pids,
                query_camids=query_camids,
                gallery_camids=gallery_camids,
                junk_same_camera=junk_same_camera,
            )
        score_s = time.perf_counter() - t_score
        metrics["num_clips"] = float(len(flat_windows))
        _attach_runtime(
            metrics,
            decode_s=decode_s,
            video_s=video_s,
            text_s=text_s,
            score_s=score_s,
            n_video=len(flat_windows),
            peak_gpu_gb=peak_gpu_gb,
            reserved_gpu_gb=reserved_gpu_gb,
        )
        scored[pool] = metrics
    return scored


def _attach_runtime(
    metrics: dict[str, float],
    *,
    decode_s: float,
    video_s: float,
    text_s: float,
    score_s: float,
    n_video: int,
    peak_gpu_gb: float,
    reserved_gpu_gb: float,
) -> None:
    total_s = decode_s + video_s + text_s + score_s
    metrics["decode_s"] = float(decode_s)
    metrics["video_s"] = float(video_s)
    metrics["text_s"] = float(text_s)
    metrics["score_s"] = float(score_s)
    metrics["total_s"] = float(total_s)
    metrics["video_ms_per_item"] = (
        float(video_s * 1000.0 / n_video) if n_video else 0.0
    )
    metrics["peak_gpu_gb"] = float(peak_gpu_gb)
    metrics["reserved_gpu_gb"] = float(reserved_gpu_gb)


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _gpu_reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        return


def _gpu_memory_gb() -> tuple[float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = float(torch.cuda.max_memory_allocated()) / (1024**3)
        reserved = float(torch.cuda.max_memory_reserved()) / (1024**3)
        return allocated, reserved
    except Exception:
        return 0.0, 0.0


def _pooled_gallery(
    grouped: list[np.ndarray],
    pool: str,
    encode_stride: int,
) -> np.ndarray:
    name = pool
    factor = 1
    if pool == "mean_s8":
        name = "mean"
        factor = max(1, 8 // max(int(encode_stride), 1))
    vectors = [
        pool_clip_features(_select_clip_rows(clips, factor), pool=name)
        for clips in grouped
    ]
    return np.stack(vectors, axis=0)


def _select_clip_rows(clips: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1 or clips.shape[0] <= 1:
        return clips
    indices = list(range(0, clips.shape[0], factor))
    if indices[-1] != clips.shape[0] - 1:
        indices.append(clips.shape[0] - 1)
    return clips[np.asarray(indices, dtype=np.int64)]


def _group_clip_features(
    clip_features: np.ndarray,
    owners: Sequence[int],
    num_tracklets: int,
) -> list[np.ndarray]:
    grouped: list[list[np.ndarray]] = [[] for _ in range(num_tracklets)]
    for row, owner in zip(clip_features, owners, strict=True):
        grouped[int(owner)].append(row)
    out: list[np.ndarray] = []
    for clips in grouped:
        if not clips:
            raise RuntimeError("A gallery tracklet produced zero window clips.")
        out.append(np.stack(clips, axis=0))
    return out


def _prepare_gallery(
    gallery: Sequence[GalleryTracklet],
    num_frames: int,
    frame_cache: Path | str | None = None,
) -> tuple[list[list[Path]], np.ndarray, np.ndarray]:
    videos: list[list[Path]] = []
    pids: list[int] = []
    camids: list[int] = []
    skipped = 0
    cache = Path(frame_cache) if frame_cache is not None else None
    for tracklet in tqdm(gallery, desc="Decode frames", unit="video"):
        sampled = resolve_frame_paths(
            tracklet.crop_paths,
            num_frames=num_frames,
            frame_cache=cache,
        )
        if not sampled:
            skipped += 1
            continue
        videos.append(sampled)
        pids.append(int(tracklet.person_id))
        camids.append(int(tracklet.camera_id))
    if skipped:
        print(f"Gallery: skipped {skipped} tracklets without frames")
    return videos, np.asarray(pids, dtype=np.int64), np.asarray(camids, dtype=np.int64)


def _prepare_window_gallery(
    gallery: Sequence[GalleryTracklet],
    window: int,
    stride: int,
    sample_fps: float,
    max_frames: int,
    frame_cache: Path | str | None = None,
) -> tuple[list[list[list[Path]]], np.ndarray, np.ndarray]:
    groups: list[list[list[Path]]] = []
    pids: list[int] = []
    camids: list[int] = []
    skipped = 0
    cache = Path(frame_cache) if frame_cache is not None else None
    for tracklet in tqdm(gallery, desc="Decode windows", unit="video"):
        dense = resolve_dense_frame_paths(
            tracklet.crop_paths,
            sample_fps=sample_fps,
            max_frames=max_frames,
            frame_cache=cache,
        )
        windows = sliding_windows(dense, window=window, stride=stride)
        if not windows:
            skipped += 1
            continue
        groups.append(windows)
        pids.append(int(tracklet.person_id))
        camids.append(int(tracklet.camera_id))
    if skipped:
        print(f"Gallery: skipped {skipped} tracklets without frames")
    return groups, np.asarray(pids, dtype=np.int64), np.asarray(camids, dtype=np.int64)


def _prepare_queries(
    queries: Sequence[CaptionQuery],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts: list[str] = []
    pids: list[int] = []
    camids: list[int] = []
    for query in queries:
        texts.append(query.text)
        pids.append(int(query.person_id))
        camids.append(int(query.camera_id))
    return texts, np.asarray(pids, dtype=np.int64), np.asarray(camids, dtype=np.int64)
