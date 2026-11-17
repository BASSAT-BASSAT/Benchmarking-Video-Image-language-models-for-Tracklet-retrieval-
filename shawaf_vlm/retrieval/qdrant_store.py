"""Qdrant storage: one global and one local (per-window) collection per model.

Collections are named ``shawaf_global__<model>`` / ``shawaf_local__<model>``
so models with different embedding sizes never share a collection. Points are
filtered by ``dataset_key`` (keyword payload index).

Ingestion uses ``upload_collection`` with numpy arrays (no per-point
``PointStruct`` or ``.tolist()``) in batches of 256. By default collections
store int8 scalar-quantized copies in RAM with the float vectors kept for
rescoring (``SHAWAF_QDRANT_QUANT=none`` turns this off).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from shawaf_vlm.retrieval.encode import TrackletVectors

GLOBAL_COLLECTION = "shawaf_global"
LOCAL_COLLECTION = "shawaf_local"
POOLING_SUPPORTED = ("mean", "query_max", "global_local")
UPLOAD_BATCH = 256


def qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


def collection_names(model_key: str) -> tuple[str, str]:
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model_key)
    return f"{GLOBAL_COLLECTION}__{safe}", f"{LOCAL_COLLECTION}__{safe}"


def _point_id(model_key: str, dataset_key: str, tracklet_id: str, clip_index: int | None) -> int:
    raw = f"{model_key}|{dataset_key}|{tracklet_id}|{clip_index}"
    digest = hashlib.sha256(raw.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def get_client(location: str | None = None):
    """``location=":memory:"`` gives an in-process client (tests, no server).

    Set ``QDRANT_PREFER_GRPC=1`` when port 6334 is reachable for faster bulk upload.
    """

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError("Install demo extras: pip install -e '.[demo]'") from exc
    if location is not None:
        return QdrantClient(location=location)
    grpc = os.environ.get("QDRANT_PREFER_GRPC", "0") == "1"
    return QdrantClient(url=qdrant_url(), prefer_grpc=grpc)


def _quantization_config(qm):
    mode = os.environ.get("SHAWAF_QDRANT_QUANT", "int8").lower()
    if mode in ("none", "off", "0"):
        return None
    if mode == "binary":
        return qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=True))
    return qm.ScalarQuantization(
        scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, quantile=0.99, always_ram=True)
    )


def ensure_collections(client: Any, model_key: str, vector_size: int) -> tuple[str, str]:
    from qdrant_client.http import models as qm

    names = collection_names(model_key)
    for name in names:
        if client.collection_exists(name):
            info = client.get_collection(name)
            vectors = info.config.params.vectors
            size = getattr(vectors, "size", None)
            if size is not None and int(size) != int(vector_size):
                raise ValueError(
                    f"Collection {name} holds {size}-d vectors but {model_key} produced {vector_size}-d. "
                    "Delete the collection or use a different model key."
                )
            continue
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            quantization_config=_quantization_config(qm),
        )
        for field in ("dataset_key", "tracklet_id"):
            client.create_payload_index(name, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD)
    return names


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
    batch_size: int = UPLOAD_BATCH,
) -> dict[str, int]:
    """Upload global and per-window vectors. Returns point counts per collection."""

    if not tracklets:
        return {}
    dim = int(tracklets[0].global_vector.shape[0])
    global_name, local_name = ensure_collections(client, model_key, dim)

    global_vecs = np.stack([t.global_vector for t in tracklets]).astype(np.float32, copy=False)
    global_ids = [_point_id(model_key, dataset_key, t.tracklet_id, None) for t in tracklets]
    global_payload = [_payload(model_key, dataset_key, t) for t in tracklets]

    local_rows: list[np.ndarray] = []
    local_ids: list[int] = []
    local_payload: list[dict[str, Any]] = []
    for t in tracklets:
        clips = np.asarray(t.clip_vectors, dtype=np.float32).reshape(-1, dim)
        local_rows.append(clips)
        for clip_index in range(clips.shape[0]):
            local_ids.append(_point_id(model_key, dataset_key, t.tracklet_id, clip_index))
            local_payload.append(_payload(model_key, dataset_key, t, clip_index))
    local_vecs = np.concatenate(local_rows, axis=0) if local_rows else np.zeros((0, dim), np.float32)

    client.upload_collection(
        collection_name=global_name,
        vectors=global_vecs,
        payload=global_payload,
        ids=global_ids,
        batch_size=batch_size,
        wait=True,
    )
    if local_ids:
        client.upload_collection(
            collection_name=local_name,
            vectors=local_vecs,
            payload=local_payload,
            ids=local_ids,
            batch_size=batch_size,
            wait=True,
        )
    return {global_name: len(global_ids), local_name: len(local_ids)}


def search_global(
    client: Any,
    query_vector: np.ndarray,
    model_key: str,
    dataset_key: str,
    limit: int = 200,
) -> list[dict[str, Any]]:
    from qdrant_client.http import models as qm

    global_name, _ = collection_names(model_key)
    response = client.query_points(
        collection_name=global_name,
        query=np.asarray(query_vector, dtype=np.float32).tolist(),
        limit=limit,
        query_filter=qm.Filter(
            must=[qm.FieldCondition(key="dataset_key", match=qm.MatchValue(value=dataset_key))]
        ),
        search_params=qm.SearchParams(
            quantization=qm.QuantizationSearchParams(rescore=True, oversampling=2.0)
        ),
        with_payload=True,
    )
    return [{"score": point.score, "payload": point.payload} for point in response.points]
