from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits
from shawaf_vlm.metrics import evaluate_text_retrieval
from shawaf_vlm.models.protocol import VideoTextEncoder
from shawaf_vlm.sampling import resolve_frame_paths


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

    gallery_videos, gallery_pids, gallery_camids = _prepare_gallery(
        splits.gallery,
        num_frames=num_frames,
        frame_cache=frame_cache,
    )
    query_texts, query_pids, query_camids = _prepare_queries(splits.query)
    if not gallery_videos:
        raise RuntimeError("No gallery tracklets with existing crop files.")
    if not query_texts:
        raise RuntimeError("No caption queries with existing crop files.")

    video_features = encoder.encode_videos(
        gallery_videos,
        batch_size=batch_size,
    )
    text_features = encoder.encode_texts(
        query_texts,
        batch_size=text_batch_size,
    )
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

    return evaluate_text_retrieval(
        query_features=text_features,
        gallery_features=video_features,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        junk_same_camera=junk_same_camera,
    )


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
