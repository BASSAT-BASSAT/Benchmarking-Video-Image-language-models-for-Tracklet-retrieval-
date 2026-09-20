from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shawaf_vlm.data.tv_mars import (
    CaptionFilesNotFound,
    camera_id_from_crop,
    load_tv_mars,
    remap_crop_path,
)
from shawaf_vlm.metrics import evaluate_text_retrieval
from shawaf_vlm.sampling import sample_frame_paths


def test_remap_strips_author_prefix(tmp_path: Path) -> None:
    data_root = tmp_path / "MARS"
    raw = "/media/jqzhu/e/jjsu/datasets/bbox_train/0009/0009C1T0001F001.jpg"
    mapped = remap_crop_path(raw, data_root)
    assert mapped == data_root / "bbox_train" / "0009" / "0009C1T0001F001.jpg"


def test_camera_id_from_filename() -> None:
    assert camera_id_from_crop("0009C5T0004F001.jpg") == 5


def test_sample_pads_short_tracklets(tmp_path: Path) -> None:
    paths = [tmp_path / f"f{index}.jpg" for index in range(3)]
    sampled = sample_frame_paths(paths, num_frames=8)
    assert len(sampled) == 8
    assert sampled[-1] == paths[-1]


def test_load_flat_json_and_official_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "MARS"
    crop = data_root / "bbox_train" / "0009" / "0009C1T0001F001.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"fake")

    record = {
        "img_path": [f"/media/host/datasets/bbox_train/0009/{crop.name}"],
        "person_id": "0009",
        "track_id": "T0001",
        "captions": ["A person riding a bicycle."],
    }

    flat = tmp_path / "partical_dataset"
    flat.mkdir()
    (flat / "0009.json").write_text(json.dumps([record]), encoding="utf-8")
    splits = load_tv_mars(data_root, flat)
    assert splits.source == "flat_json"
    assert len(splits.query) == 1
    assert splits.query[0].person_id == 9
    assert splits.gallery[0].crop_paths[0] == crop

    official = tmp_path / "tv_mars"
    (official / "test_query_info").mkdir(parents=True)
    (official / "test_gallery_info").mkdir()
    (official / "test_query_info" / "q.json").write_text(
        json.dumps([record]),
        encoding="utf-8",
    )
    (official / "test_gallery_info" / "g.json").write_text(
        json.dumps([record]),
        encoding="utf-8",
    )
    official_splits = load_tv_mars(data_root, official)
    assert official_splits.source == "official"


def test_missing_captions_error(tmp_path: Path) -> None:
    data_root = tmp_path / "MARS"
    data_root.mkdir()
    with pytest.raises(CaptionFilesNotFound, match="agreement"):
        load_tv_mars(data_root, tmp_path / "missing")


def test_text_retrieval_ranks_matching_identity() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    metrics = evaluate_text_retrieval(
        query_features=query,
        gallery_features=gallery,
        query_pids=np.array([1, 2]),
        gallery_pids=np.array([1, 2, 2]),
    )
    assert metrics["Rank-1"] == 100.0
    assert metrics["num_valid_queries"] == 2.0
