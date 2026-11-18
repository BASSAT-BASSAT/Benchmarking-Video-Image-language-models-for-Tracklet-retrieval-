from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from shawaf_vlm.data import alexandria as alex  # noqa: E402
from shawaf_vlm.retrieval.datasets import load_splits  # noqa: E402


def _clip_zip(captions: dict[str, str], n_frames: int = 3) -> bytes:
    buf = io.BytesIO()
    identities = {}
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n_frames):
            img = io.BytesIO()
            Image.new("RGB", (64, 48), (i * 40, 10, 10)).save(img, format="JPEG")
            zf.writestr(f"frames/frame_{i:06d}.jpg", img.getvalue())
        for pid, cap in captions.items():
            identities[pid] = {
                "reidId": pid,
                "caption": cap,
                "instances": [
                    {"frameIndex": i, "frameFile": f"frames/frame_{i:06d}.jpg", "bbox": [4, 4, 30, 40]}
                    for i in range(n_frames)
                ],
            }
        zf.writestr("reid_index.json", json.dumps({"identities": identities}))
        zf.writestr(
            "annotations.json",
            json.dumps({"videoName": "x.MOV", "sampleFps": 10, "durationSec": 1.0, "frames": [{}] * n_frames}),
        )
    return buf.getvalue()


def _outer(path: Path, day: int, clips: dict[int, dict[str, str]]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for clip, captions in clips.items():
            zf.writestr(f"day{day}/{clip}.zip", _clip_zip(captions))


@pytest.fixture()
def layout(tmp_path: Path) -> dict[str, Path]:
    zips = tmp_path / "Annotations"
    zips.mkdir()
    _outer(zips / "day1-2026.zip", 1, {1: {"person_1": "girl in red"}, 2: {"person_1": "man with bag"}})
    _outer(zips / "day2-2026.zip", 2, {11: {"person_1": "boy on phone", "person_2": ""}})
    videos = tmp_path / "Videos"
    (videos / "Day one").mkdir(parents=True)
    (videos / "Day Two").mkdir(parents=True)
    (videos / "Day one" / "1.mov").write_bytes(b"")
    (videos / "Day Two" / "11 drinking3 11.mov").write_bytes(b"")
    return {"zips": zips, "source": tmp_path / "src", "videos": videos, "prepared": tmp_path / "prep"}


def test_index_videos_matches_leading_clip_number(layout):
    found = alex.index_videos(layout["videos"])
    assert set(found) == {(1, 1), (2, 11)}


def test_unpack_is_idempotent_and_records_videos(layout):
    entries = alex.unpack_alexandria(layout["zips"], layout["source"], layout["videos"])
    assert [e["project"] for e in entries] == ["day1_1", "day1_2", "day2_11"]
    assert entries[1]["video"] is None
    assert entries[2]["captioned"] == 1 and entries[2]["identities"] == 2
    again = alex.unpack_alexandria(layout["zips"], layout["source"], layout["videos"])
    assert [e["project"] for e in again] == [e["project"] for e in entries]


def test_prepare_and_load_configs(layout):
    alex.unpack_alexandria(layout["zips"], layout["source"], layout["videos"])
    alex.prepare_alexandria(layout["source"], layout["prepared"])
    splits = load_splits("alexandria:all", root=layout["prepared"])
    assert len(splits.gallery) == 4
    assert sorted(q.text for q in splits.query) == ["boy on phone", "girl in red", "man with bag"]
    day1 = load_splits("alexandria:day1", root=layout["prepared"])
    assert len(day1.gallery) == 2
    clip = load_splits("alexandria:day2_11", root=layout["prepared"])
    assert len(clip.gallery) == 2
    with pytest.raises(FileNotFoundError):
        load_splits("alexandria:day9", root=layout["prepared"])


def test_excluded_tracks_are_filtered(layout, monkeypatch):
    alex.unpack_alexandria(layout["zips"], layout["source"], layout["videos"])
    alex.prepare_alexandria(layout["source"], layout["prepared"])
    monkeypatch.setitem(alex.EXCLUDED_TRACKS, "day1_1/person_1", "test")
    splits = load_splits("alexandria:all", root=layout["prepared"])
    assert "day1_1/person_1" not in {g.track_id for g in splits.gallery}
    assert "day1_1/person_1" not in {q.track_id for q in splits.query}
    assert len(splits.gallery) == 3
