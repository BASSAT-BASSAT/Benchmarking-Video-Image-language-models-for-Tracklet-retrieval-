"""Qdrant store against the in-process client. No server needed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from shawaf_vlm.retrieval.encode import TrackletVectors  # noqa: E402
from shawaf_vlm.retrieval.qdrant_store import (  # noqa: E402
    collection_names,
    get_client,
    search_global,
    upsert_tracklets,
)


def _tracklets(dim: int, n: int = 3, clips: int = 2) -> list[TrackletVectors]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        g = rng.normal(size=dim).astype(np.float32)
        g /= np.linalg.norm(g)
        out.append(
            TrackletVectors(
                tracklet_id=f"t{i}",
                person_id=i,
                media_path=f"/m/{i}.mp4",
                global_vector=g,
                clip_vectors=rng.normal(size=(clips, dim)).astype(np.float32),
            )
        )
    return out


def test_upload_and_search_per_model_collections() -> None:
    client = get_client(":memory:")
    tracks = _tracklets(8)
    counts = upsert_tracklets(client, "pe_core_l14", "alexandria:all", tracks)
    g, l = collection_names("pe_core_l14")
    assert counts == {g: 3, l: 6}
    hits = search_global(client, tracks[1].global_vector, "pe_core_l14", "alexandria:all", limit=3)
    assert hits[0]["payload"]["tracklet_id"] == "t1"
    assert search_global(client, tracks[1].global_vector, "pe_core_l14", "other:ds", limit=3) == []


def test_models_with_different_dims_do_not_collide() -> None:
    client = get_client(":memory:")
    upsert_tracklets(client, "pe_core_l14", "d", _tracklets(8))
    upsert_tracklets(client, "pe_core_t16", "d", _tracklets(4))
    assert len(search_global(client, _tracklets(4)[0].global_vector, "pe_core_t16", "d")) == 3


def test_reupload_is_idempotent() -> None:
    client = get_client(":memory:")
    tracks = _tracklets(8)
    upsert_tracklets(client, "m", "d", tracks)
    upsert_tracklets(client, "m", "d", tracks)
    g, _ = collection_names("m")
    assert client.count(g).count == 3
