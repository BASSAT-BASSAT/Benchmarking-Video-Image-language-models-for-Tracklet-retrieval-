from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from shawaf_vlm.data.local_annotations import (
    load_local_annotations,
    prepare_project,
    rows_to_local_splits,
    slugify,
)


def _write_mini_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_vid"
    frames = project / "frames"
    frames.mkdir(parents=True)
    img = Image.new("RGB", (100, 80), color=(200, 100, 50))
    img.save(frames / "frame_000000.jpg")
    reid = {
        "captionScope": "identity",
        "identities": {
            "person_1": {
                "reidId": "person_1",
                "caption": "red shirt pedestrian",
                "instances": [
                    {
                        "frameIndex": 0,
                        "frameFile": "frames/frame_000000.jpg",
                        "bbox": [10.0, 5.0, 60.0, 75.0],
                    }
                ],
            }
        },
    }
    (project / "reid_index.json").write_text(json.dumps(reid), encoding="utf-8")
    return project


def test_slugify() -> None:
    assert slugify("wearing cap1") == "wearing_cap1"


def test_prepare_and_load(tmp_path: Path) -> None:
    project = _write_mini_project(tmp_path)
    dest = tmp_path / "out"
    jsonl = prepare_project(project, dest)
    assert jsonl.is_file()
    splits = load_local_annotations("demo_vid", root=dest)
    assert len(splits.gallery) == 1
    assert splits.gallery[0].crop_paths[0].is_file()
    assert splits.query[0].text == "red shirt pedestrian"


def test_rows_to_local_splits_skips_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    rows = [{"tracklet_id": "a/p1", "subset": "a", "crops": ["missing.jpg"], "captions": ["x"]}]
    queries, gallery = rows_to_local_splits(rows, root)
    assert not gallery
    assert not queries
