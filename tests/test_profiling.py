"""Stage timer and indexer --profile. No checkpoints and no Qdrant."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from PIL import Image

from shawaf_vlm.data.tv_mars import GalleryTracklet, TVMarsSplits
from shawaf_vlm.models.frame_encoders import _FrameEncoder
from shawaf_vlm.profiling import StageTimer


def test_stage_timer_accumulates_and_summarizes() -> None:
    ticks = iter([0.0, 1.5, 1.5, 3.0, 3.0, 3.5])

    def clock() -> float:
        return next(ticks)

    timer = StageTimer(clock=clock)
    with timer.stage("load"):
        pass
    with timer.stage("forward"):
        pass
    timer.tracklets = 4
    timer.frames = 8
    summary = timer.summary()
    assert summary["stages_s"]["load"] == 1.5
    assert summary["stages_s"]["forward"] == 1.5
    assert summary["stages_s"]["preprocess"] == 0.0
    assert summary["wall_s"] == 3.0
    assert summary["tracklets_per_s"] == 4 / 3.0
    assert summary["frames_per_s"] == 8 / 3.0


def test_stage_timer_true_wall_and_sync() -> None:
    ticks = iter([0.0, 1.0, 2.0, 10.0])
    calls: list[int] = []
    timer = StageTimer(clock=lambda: next(ticks), sync=lambda: calls.append(1))
    timer.start()  # 0.0
    with timer.stage("model_load"):  # 1.0 -> 2.0
        pass
    timer.stop()  # 10.0
    timer.tracklets = 4
    summary = timer.summary()
    assert summary["wall_s"] == 10.0
    assert summary["unaccounted_s"] == 9.0
    assert summary["ingest_s"] == 9.0
    assert summary["ingest_tracklets_per_s"] == 4 / 9.0
    assert len(calls) == 4  # start, stage enter, stage exit, stop


class _CountingEncoder(_FrameEncoder):
    def __init__(self) -> None:
        super().__init__(device="cpu", frame_microbatch=16, num_workers=0)

    def _encode_images(self, images):
        return torch.ones(len(images), 4)


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "index_qdrant.py"
    spec = importlib.util.spec_from_file_location("index_qdrant_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_index_profile_writes_summary(tmp_path, monkeypatch) -> None:
    root = tmp_path / "frames"
    root.mkdir()
    gallery = []
    for track in range(3):
        paths = []
        for frame in range(12):
            path = root / f"t{track}_f{frame}.jpg"
            Image.new("RGB", (8, 8), color=(track, frame, 0)).save(path)
            paths.append(path)
        gallery.append(
            GalleryTracklet(
                crop_paths=tuple(paths),
                person_id=track,
                camera_id=1,
                track_id=f"track-{track}",
            )
        )
    splits = TVMarsSplits(query=[], gallery=gallery, source="unit")
    module = _script()
    monkeypatch.setattr(module, "load_splits", lambda *args, **kwargs: splits)
    monkeypatch.setattr(module, "dataset_status", lambda: {})
    monkeypatch.setattr(module, "build_encoder", lambda *args, **kwargs: _CountingEncoder())
    cache = tmp_path / "cache"
    module.main(
        [
            "--model",
            "pe_core_l14",
            "--dataset",
            "local:all",
            "--device",
            "cpu",
            "--no-download",
            "--skip-qdrant",
            "--profile",
            "--cache-root",
            str(cache),
        ]
    )
    out = cache / "pe_core_l14" / "local_all"
    profile = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "index_summary.json").read_text(encoding="utf-8"))
    assert profile["tracklets"] == 3
    assert profile["frames"] == 36
    assert "forward" in profile["stages_s"]
    assert summary["profile"]["tracklets"] == 3
    assert (out / "manifest.json").is_file()
