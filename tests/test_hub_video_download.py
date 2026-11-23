"""Hub video downloader: archive-first fetch and wrapped-429 detection (offline)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from shawaf_vlm.data import hub_video


def test_rate_limit_detected_through_exception_chain() -> None:
    try:
        try:
            raise RuntimeError("429 Client Error: Too Many Requests")
        except RuntimeError as inner:
            raise LookupError("cannot find the requested files in the local cache") from inner
    except LookupError as outer:
        assert hub_video._is_rate_limit(outer)
    assert not hub_video._is_rate_limit(ValueError("404 Not Found"))


def test_download_prefers_split_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    rel = "TVPReid-PRID/videos/person0001.mp4"
    (src / rel).parent.mkdir(parents=True)
    (src / rel).write_bytes(b"fake mp4")
    archive = tmp_path / "prid-train.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(src / rel, arcname=rel)
    table = {"video_id": "person0001", "video": rel, "captions": ["a man"], "subset": "prid"}

    requested: list[str] = []

    def fake_hub_file(repo_id, filename, token, dest, retries=8):
        requested.append(filename)
        target = Path(dest) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".jsonl"):
            target.write_text(json.dumps(table) + "\n", encoding="utf-8")
        elif filename == "archives/prid-train.tar":
            target.write_bytes(archive.read_bytes())
        else:
            raise AssertionError(f"per-file fetch should not happen: {filename}")
        return str(target)

    monkeypatch.setattr(hub_video, "hub_file", fake_hub_file)
    dest = tmp_path / "dest"
    hub_video.download_hub_video_dataset("x/y", ("prid",), "train", dest)
    assert (dest / rel).read_bytes() == b"fake mp4"
    assert requested == ["metadata/prid-train.jsonl", "archives/prid-train.tar"]
    assert not (dest / "archives/prid-train.tar").exists()
