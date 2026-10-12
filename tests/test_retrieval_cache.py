from __future__ import annotations

from pathlib import Path

import numpy as np

from shawaf_vlm.retrieval.cache import load_numpy_index, save_numpy_index
from shawaf_vlm.retrieval.encode import TrackletVectors
from shawaf_vlm.retrieval.search import score_tracklets


def test_numpy_index_roundtrip(tmp_path: Path) -> None:
    t = TrackletVectors(
        tracklet_id="v1",
        person_id=1,
        media_path="/fake/v1.mp4",
        global_vector=np.array([1.0, 0.0], dtype=np.float32),
        clip_vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    save_numpy_index(tmp_path, [t], model_key="siglip2", dataset_key="tvpreid:prid")
    loaded, manifest = load_numpy_index(tmp_path)
    assert manifest["num_tracklets"] == 1
    assert loaded[0].tracklet_id == "v1"
    scores = score_tracklets(np.array([1.0, 0.0]), loaded, "mean")
    assert scores.shape == (1,)
    assert scores[0] > 0.99
