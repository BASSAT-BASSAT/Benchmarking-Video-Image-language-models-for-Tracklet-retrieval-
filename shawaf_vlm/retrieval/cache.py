"""On-disk numpy index when Qdrant is unavailable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shawaf_vlm.retrieval.encode import TrackletVectors


def index_dir(base: Path | str, model_key: str, dataset_key: str) -> Path:
    return Path(base) / model_key / dataset_key.replace(":", "_")


def save_numpy_index(
    path: Path,
    tracklets: list[TrackletVectors],
    *,
    model_key: str,
    dataset_key: str,
    pooling_supported: tuple[str, ...] = ("mean", "query_max", "global_local"),
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta = []
    globals_list = []
    clip_counts = []
    clip_flat: list[np.ndarray] = []
    offset = 0
    for t in tracklets:
        meta.append(
            {
                "tracklet_id": t.tracklet_id,
                "person_id": t.person_id,
                "media_path": t.media_path,
                "clip_offset": offset,
                "clip_count": t.clip_vectors.shape[0],
            }
        )
        globals_list.append(t.global_vector)
        clip_counts.append(t.clip_vectors.shape[0])
        for row in t.clip_vectors:
            clip_flat.append(row)
            offset += 1

    np.savez_compressed(
        path / "vectors.npz",
        global_vecs=np.stack(globals_list, axis=0).astype(np.float32),
        clips=np.stack(clip_flat, axis=0).astype(np.float32) if clip_flat else np.zeros((0, 1)),
    )
    manifest = {
        "model_key": model_key,
        "dataset_key": dataset_key,
        "pooling_supported": list(pooling_supported),
        "num_tracklets": len(tracklets),
        "tracklets": meta,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_numpy_index(path: Path) -> tuple[list[TrackletVectors], dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(path / "vectors.npz")
    global_vecs = data["global_vecs"]
    clips = data["clips"]
    tracklets: list[TrackletVectors] = []
    for index, row in enumerate(manifest["tracklets"]):
        off = int(row["clip_offset"])
        cnt = int(row["clip_count"])
        clip_vecs = clips[off : off + cnt] if cnt else np.zeros((0, global_vecs.shape[1]))
        tracklets.append(
            TrackletVectors(
                tracklet_id=str(row["tracklet_id"]),
                person_id=int(row["person_id"]),
                media_path=str(row["media_path"]),
                global_vector=global_vecs[index],
                clip_vectors=clip_vecs.astype(np.float32),
            )
        )
    return tracklets, manifest
