"""Qdrant collections shawaf_global / shawaf_local."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from shawaf_vlm.retrieval.encode import TrackletVectors

GLOBAL_COLLECTION = "shawaf_global"
LOCAL_COLLECTION = "shawaf_local"
POOLING_SUPPORTED = ("mean", "query_max", "global_local")


def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


def _point_id(model_key: str, dataset_key: str, tracklet_id: str, clip_index: int | None) -> int:
    raw = f"{model_key}|{dataset_key}|{tracklet_id}|{clip_index}"
    digest = hashlib.sha256(raw.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError("Install demo extras: pip install -e '.[demo]'") from exc
    return QdrantClient(url=qdrant_url())


def ensure_collections(client: Any, vector_size: int) -> None:
    from qdrant_client.http import models as qm

    for name in (GLOBAL_COLLECTION, LOCAL_COLLECTION):
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            )


def _payload(
    model_key: str,
    dataset_key: str,
    tracklet: TrackletVectors,
    clip_index: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_key": model_key,
        "dataset_key": dataset_key,
        "tracklet_id": tracklet.tracklet_id,
        "person_id": tracklet.person_id,
        "video_path": tracklet.media_path,
        "pooling_supported": list(POOLING_SUPPORTED),
    }
    if clip_index is not None:
        payload["clip_index"] = clip_index
    return payload


def upsert_tracklets(
    client: Any,
    model_key: str,
    dataset_key: str,
    tracklets: list[TrackletVectors],
    batch_size: int = 64,
) -> None:
    from qdrant_client.http import models as qm

    if not tracklets:
        return
    dim = int(tracklets[0].global_vector.shape[0])
    ensure_collections(client, dim)

    global_points: list[qm.PointStruct] = []
    local_points: list[qm.PointStruct] = []
    for t in tracklets:
        gid = _point_id(model_key, dataset_key, t.tracklet_id, None)
        global_points.append(
            qm.PointStruct(
                id=gid,
                vector=t.global_vector.astype(np.float32).tolist(),
                payload=_payload(model_key, dataset_key, t),
            )
        )
        for clip_index, vec in enumerate(t.clip_vectors):
            lid = _point_id(model_key, dataset_key, t.tracklet_id, clip_index)
            local_points.append(
                qm.PointStruct(
                    id=lid,
                    vector=vec.astype(np.float32).tolist(),
                    payload=_payload(model_key, dataset_key, t, clip_index),
                )
            )

    for start in range(0, len(global_points), batch_size):
        client.upsert(collection_name=GLOBAL_COLLECTION, points=global_points[start : start + batch_size])
    for start in range(0, len(local_points), batch_size):
        client.upsert(collection_name=LOCAL_COLLECTION, points=local_points[start : start + batch_size])


def search_global(
    client: Any,
    query_vector: np.ndarray,
    model_key: str,
    dataset_key: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    from qdrant_client.http import models as qm

    hits = client.search(
        collection_name=GLOBAL_COLLECTION,
        query_vector=query_vector.astype(np.float32).tolist(),
        limit=limit,
        query_filter=qm.Filter(
            must=[
                qm.FieldCondition(key="model_key", match=qm.MatchValue(value=model_key)),
                qm.FieldCondition(key="dataset_key", match=qm.MatchValue(value=dataset_key)),
            ]
        ),
    )
    return [{"score": h.score, "payload": h.payload} for h in hits]
