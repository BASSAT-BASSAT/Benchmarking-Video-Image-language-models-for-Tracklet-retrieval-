"""Encode gallery tracklets into global and per-clip vectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shawaf_vlm.data.local_annotations import tracklet_preview_media_path
from shawaf_vlm.data.tv_mars import TVMarsSplits
from shawaf_vlm.eval_loop import _group_clip_features, _prepare_window_gallery
from shawaf_vlm.models.protocol import VideoTextEncoder
from shawaf_vlm.pooling import pool_clip_features


@dataclass
class TrackletVectors:
    tracklet_id: str
    person_id: int
    media_path: str
    global_vector: np.ndarray
    clip_vectors: np.ndarray


def encode_gallery_tracklets(
    encoder: VideoTextEncoder,
    splits: TVMarsSplits,
    *,
    num_frames: int = 8,
    stride: int = 4,
    sample_fps: float = 8.0,
    max_frames: int = 64,
    batch_size: int = 4,
    frame_cache: Path | str | None = None,
) -> list[TrackletVectors]:
    clip_groups, gallery_pids, _ = _prepare_window_gallery(
        splits.gallery,
        window=num_frames,
        stride=stride,
        sample_fps=sample_fps,
        max_frames=max_frames,
        frame_cache=frame_cache,
    )
    flat: list[list[Path]] = []
    owners: list[int] = []
    for track_index, windows in enumerate(clip_groups):
        for window in windows:
            flat.append(window)
            owners.append(track_index)

    if not flat:
        raise RuntimeError("No window clips to encode.")

    clip_features = encoder.encode_videos(flat, batch_size=batch_size)
    grouped = _group_clip_features(clip_features, owners, len(clip_groups))

    out: list[TrackletVectors] = []
    for index, (clips, tracklet) in enumerate(zip(grouped, splits.gallery, strict=True)):
        if index >= len(gallery_pids):
            break
        preview = tracklet_preview_media_path(list(tracklet.crop_paths))
        if preview is not None:
            media = str(preview)
        else:
            media = str(tracklet.crop_paths[0]) if tracklet.crop_paths else ""
        out.append(
            TrackletVectors(
                tracklet_id=tracklet.track_id,
                person_id=int(gallery_pids[index]),
                media_path=media,
                global_vector=pool_clip_features(clips, pool="mean"),
                clip_vectors=clips.astype(np.float32),
            )
        )
    return out
