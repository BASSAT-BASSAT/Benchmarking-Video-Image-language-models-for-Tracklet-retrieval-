from __future__ import annotations

import json
from pathlib import Path

import pytest

from shawaf_vlm.data.groot import GroOTAccessError, load_groot
from shawaf_vlm.data.groot_build import (
    FrameSource,
    clean_caption,
    collect_tracks,
    expand_box,
    metadata_rows,
    write_metadata,
    write_track_videos,
)
from shawaf_vlm.data.rstpreid import camera_id_from_name, load_rstpreid
from shawaf_vlm.sampling import resolve_frame_paths


def _write_rstpreid(root: Path) -> None:
    images = root / "imgs"
    images.mkdir(parents=True)
    records = []
    for person, split in ((7, "test"), (9, "test"), (3, "train")):
        for index in range(2):
            name = f"{person:04d}_c{index + 4}_{index:04d}.jpg"
            (images / name).write_bytes(b"jpg")
            records.append(
                {
                    "id": person,
                    "img_path": name,
                    "captions": [f"person {person} caption a", f"person {person} caption b"],
                    "split": split,
                }
            )
    (root / "data_captions.json").write_text(json.dumps(records), encoding="utf-8")


def test_rstpreid_split_and_identity(tmp_path: Path) -> None:
    _write_rstpreid(tmp_path)
    splits = load_rstpreid(tmp_path, split="test")
    assert splits.source == "rstpreid:test"
    assert len(splits.gallery) == 4
    assert len(splits.query) == 8
    assert {g.person_id for g in splits.gallery} == {7, 9}
    positives = [g for g in splits.gallery if g.person_id == splits.query[0].person_id]
    assert len(positives) == 2
    assert splits.gallery[0].camera_id == 4


def test_rstpreid_single_image_is_repeated_clip(tmp_path: Path) -> None:
    _write_rstpreid(tmp_path)
    tracklet = load_rstpreid(tmp_path).gallery[0]
    frames = resolve_frame_paths(tracklet.crop_paths, num_frames=8)
    assert frames == [tracklet.crop_paths[0]] * 8


def test_rstpreid_camera_parse() -> None:
    assert camera_id_from_name("0000_c14_0031.jpg") == 14
    assert camera_id_from_name("weird.jpg") == 0


def _groot_coco() -> dict:
    images = []
    annotations = []
    image_id = 0
    ann_id = 0
    for sequence in ("MOT17-02-FRCNN", "MOT17-02-DPM"):
        for frame in range(1, 11):
            images.append({"id": image_id, "file_name": f"data/MOT17/x/{sequence}_{frame:06d}.jpg"})
            tracks = [
                (1, [10, 10, 40, 100], ["man in red shirt", "man walking"]),
                (2, [100, 20, 30, 90], ["woman in blue coat", None]),
                (3, [150, 20, 10, 20], ["tiny person", "person walking"]),
                (4, [60, 10, 40, 100], [None, None]),
            ]
            for track_id, bbox, captions in tracks:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "bbox": bbox,
                        "track_id": track_id,
                        "category_id": 1,
                        "ignore": 0,
                        "visibility": 1.0,
                        "captions": captions,
                    }
                )
                ann_id += 1
            image_id += 1
    return {"images": images, "annotations": annotations}


def test_groot_collect_tracks_filters_and_dedups() -> None:
    tracks = collect_tracks(_groot_coco(), min_height=50, min_frames=8)
    assert [(t.sequence, t.track_id) for t in tracks] == [
        ("MOT17-02-FRCNN", 1),
        ("MOT17-02-FRCNN", 2),
    ]
    assert tracks[1].action is None
    assert tracks[0].video_rel == "videos/MOT17-02_t0001.mp4"
    assert clean_caption("None") is None


def test_groot_metadata_configs_share_gallery() -> None:
    tracks = collect_tracks(_groot_coco(), min_height=50, min_frames=8)
    rows = {config: metadata_rows(tracks, config) for config in ("all", "appearance", "action")}
    assert [r["video"] for r in rows["all"]] == [r["video"] for r in rows["action"]]
    assert sum(len(r["captions"]) for r in rows["all"]) == 3
    assert sum(len(r["captions"]) for r in rows["appearance"]) == 2
    assert rows["action"][0]["caption_types"] == ["action"]
    assert rows["action"][1]["captions"] == []


def test_expand_box_keeps_aspect_and_clips() -> None:
    x0, y0, x1, y1 = expand_box((0, 0, 20, 100), aspect=0.5, image_w=1000, image_h=1000)
    assert (x0, y0) == (0, 0)
    assert y1 == 100 and x1 == 35


def test_groot_build_and_load(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    img1 = tmp_path / "MOT17" / "train" / "MOT17-02-FRCNN" / "img1"
    img1.mkdir(parents=True)
    (img1.parent / "seqinfo.ini").write_text("[Sequence]\nframeRate=30\n", encoding="utf-8")
    for frame in range(1, 11):
        cv2.imwrite(str(img1 / f"{frame:06d}.jpg"), np.full((200, 300, 3), frame * 20, np.uint8))

    out = tmp_path / "GroOT-MOT17"
    tracks = collect_tracks(_groot_coco(), min_height=50, min_frames=8)
    tracks = write_track_videos(tracks, FrameSource(tmp_path / "MOT17"), out, height=64)
    write_metadata(tracks, out)
    assert len(tracks) == 2
    assert all((out / t.video_rel).is_file() for t in tracks)

    everything = load_groot("all", root=out)
    action = load_groot("action", root=out)
    assert len(everything.gallery) == len(action.gallery) == 2
    assert len(everything.query) == 3
    assert len(action.query) == 1
    assert action.source == "groot_mot17:action:test"

    frames = resolve_frame_paths(
        everything.gallery[0].crop_paths, num_frames=8, frame_cache=tmp_path / "cache"
    )
    assert len(frames) == 8 and all(path.is_file() for path in frames)


def test_groot_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(GroOTAccessError, match="GroOT-MOT17"):
        load_groot("all", root=tmp_path)
