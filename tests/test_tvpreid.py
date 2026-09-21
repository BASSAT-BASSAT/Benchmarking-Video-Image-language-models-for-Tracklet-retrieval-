from __future__ import annotations

import json
from pathlib import Path

import pytest

from shawaf_vlm.data.tvpreid import (
    TVPReidAccessError,
    load_tvpreid_from_root,
    video_rels_from_jsonl,
)
from shawaf_vlm.sampling import resolve_frame_paths, sample_frame_paths


def _write_dummy_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-video")


def test_load_tvpreid_explodes_captions(tmp_path: Path) -> None:
    video = tmp_path / "TVPReid-PRID" / "videos" / "person0001.mp4"
    _write_dummy_mp4(video)
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    row = {
        "video_id": "person0001",
        "video": "TVPReid-PRID/videos/person0001.mp4",
        "captions": ["A person in a white jacket.", "Someone walking with a bag."],
        "num_captions": 2,
        "subset": "prid",
        "source_dataset": "PRID-2011",
        "split": "test",
    }
    (metadata / "prid-test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    splits = load_tvpreid_from_root(tmp_path, config="prid", split="test")
    assert splits.source == "tvpreid:prid:test"
    assert len(splits.gallery) == 1
    assert len(splits.query) == 2
    assert splits.query[0].person_id == splits.gallery[0].person_id
    assert splits.gallery[0].crop_paths[0] == video
    assert video_rels_from_jsonl(metadata / "prid-test.jsonl") == [
        "TVPReid-PRID/videos/person0001.mp4"
    ]


def test_load_tvpreid_skips_missing_videos(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    row = {
        "video_id": "person0001",
        "video": "TVPReid-PRID/videos/missing.mp4",
        "captions": ["A caption."],
        "subset": "prid",
        "split": "test",
    }
    (metadata / "prid-test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No TVPReid"):
        load_tvpreid_from_root(tmp_path, config="prid", split="test")


def test_load_tvpreid_missing_jsonl(tmp_path: Path) -> None:
    with pytest.raises(TVPReidAccessError, match="unofficial mirror"):
        load_tvpreid_from_root(tmp_path, config="prid", split="test")


def test_resolve_image_tracklet_unchanged(tmp_path: Path) -> None:
    paths = [tmp_path / f"f{index}.jpg" for index in range(3)]
    sampled = resolve_frame_paths(paths, num_frames=8)
    assert sampled == sample_frame_paths(paths, num_frames=8)


def test_imagenet_preprocess_btchw_shape(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    from PIL import Image

    from shawaf_vlm.models.runtime import imagenet_preprocess_btchw

    paths = []
    for index in range(8):
        path = tmp_path / f"frame_{index:03d}.jpg"
        Image.new("RGB", (48, 80), color=(index * 8, 40, 90)).save(path)
        paths.append(path)
    tensor = imagenet_preprocess_btchw([paths], image_size=224)
    assert tuple(tensor.shape) == (1, 8, 3, 224, 224)
