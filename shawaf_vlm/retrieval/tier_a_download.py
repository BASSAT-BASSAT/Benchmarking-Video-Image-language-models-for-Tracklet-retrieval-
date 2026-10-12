"""Download Tier-A datasets when not present locally (CLI / CI — not the Gradio UI)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadResult:
    name: str
    action: str
    path: str | None = None


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def ensure_tvpreid(split: str = "test") -> DownloadResult:
    from shawaf_vlm.data.tvpreid import discover_local_tvpreid, download_tvpreid

    found = discover_local_tvpreid()
    if found is not None:
        return DownloadResult("TVPReid", "skipped", str(found))
    path = download_tvpreid(
        configs=("prid", "ilids", "duke"),
        split=split,
        token=hf_token(),
    )
    return DownloadResult("TVPReid", "downloaded", str(path))


def ensure_groot(split: str = "test") -> DownloadResult:
    from shawaf_vlm.data.groot import discover_local_groot, download_groot

    found = discover_local_groot()
    if found is not None:
        return DownloadResult("GroOT-MOT17", "skipped", str(found))
    path = download_groot(split=split, token=hf_token())
    return DownloadResult("GroOT-MOT17", "downloaded", str(path))


def ensure_rstpreid() -> DownloadResult:
    from shawaf_vlm.data.rstpreid import discover_local_rstpreid, download_rstpreid

    found = discover_local_rstpreid()
    if found is not None:
        return DownloadResult("RSTPReid", "skipped", str(found))
    path = download_rstpreid()
    return DownloadResult("RSTPReid", "downloaded", str(path))


def ensure_tier_a_datasets(
    *,
    tvpreid: bool = True,
    groot: bool = True,
    rstpreid: bool = True,
    split: str = "test",
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    if tvpreid:
        results.append(ensure_tvpreid(split=split))
    if groot:
        results.append(ensure_groot(split=split))
    if rstpreid:
        results.append(ensure_rstpreid())
    return results
