from __future__ import annotations

import os
from pathlib import Path

from shawaf_vlm.data.hub_video import (
    default_download_root,
    download_hub_video_dataset,
    load_hub_video_split,
)
from shawaf_vlm.data.tv_mars import TVMarsSplits

HF_REPO_ID = "bassatbassat/GroOT-MOT17"
HF_DATASET_URL = "https://huggingface.co/datasets/bassatbassat/GroOT-MOT17"
CONFIGS = ("all", "appearance", "action", "combined")
SPLIT = "test"


class GroOTAccessError(FileNotFoundError):
    """Raised when the GroOT-MOT17 Hub snapshot or local files are missing."""


def missing_groot_message(detail: str) -> str:
    return (
        f"{detail}\n"
        f"GroOT-MOT17 tracklet mirror: {HF_DATASET_URL}\n"
        "Enable Internet on Kaggle so the downloader can pull the mp4s, or build "
        "it with scripts/build_groot_mot17.py. Cite Nguyen et al. NeurIPS 2023 "
        "(Type-to-Track) and MOT17."
    )


def load_groot(
    config: str = "all",
    split: str = SPLIT,
    root: Path | str | None = None,
    repo_id: str = HF_REPO_ID,
    token: str | None = None,
) -> TVMarsSplits:
    """
    Load GroOT-MOT17 queries and track videos.

    ``config`` picks the queries: ``all`` (appearance + action), ``appearance``,
    or ``action``. The gallery (one mp4 per track) is the same for all three.
    """

    config = _normalize_config(config)
    split = _normalize_split(split)
    snapshot = Path(root) if root is not None else discover_local_groot()
    if snapshot is None:
        snapshot = download_groot(configs=(config,), split=split, repo_id=repo_id, token=token)
    return load_hub_video_split(
        snapshot,
        config=config,
        split=split,
        source=f"groot_mot17:{config}:{split}",
        missing=lambda detail: GroOTAccessError(missing_groot_message(detail)),
        name="GroOT-MOT17",
    )


def discover_local_groot() -> Path | None:
    env = os.environ.get("SHAWAF_GROOT_ROOT")
    if env and _looks_like_groot_root(Path(env)):
        return Path(env)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for match in kaggle_input.rglob("all-test.jsonl"):
            if _looks_like_groot_root(match.parent.parent):
                return match.parent.parent

    for candidate in (
        Path("/kaggle/working/GroOT-MOT17"),
        Path.home() / ".cache" / "shawaf_vlm" / "GroOT-MOT17",
    ):
        if _looks_like_groot_root(candidate):
            return candidate
    return None


def download_groot(
    configs: tuple[str, ...] = CONFIGS,
    split: str = SPLIT,
    repo_id: str = HF_REPO_ID,
    token: str | None = None,
    local_dir: Path | str | None = None,
    max_workers: int = 8,
) -> Path:
    """Download the GroOT-MOT17 split tables and their mp4s from the Hub."""

    normalized = tuple(_normalize_config(item) for item in configs)
    split = _normalize_split(split)
    dest = default_download_root("GroOT-MOT17", local_dir)
    try:
        return download_hub_video_dataset(
            repo_id=repo_id,
            configs=normalized,
            split=split,
            dest=dest,
            token=token,
            max_workers=max_workers,
        )
    except Exception as exc:
        raise GroOTAccessError(
            missing_groot_message(f"Failed to download {repo_id}: {exc}")
        ) from exc


def _looks_like_groot_root(path: Path) -> bool:
    metadata = path / "metadata"
    return all((metadata / f"{config}-{SPLIT}.jsonl").is_file() for config in CONFIGS)


def _normalize_config(config: str) -> str:
    key = config.strip().lower()
    if key not in CONFIGS:
        raise ValueError(f"Unknown GroOT config {config!r}. Use {', '.join(CONFIGS)}.")
    return key


def _normalize_split(split: str) -> str:
    key = split.strip().lower()
    if key != SPLIT:
        raise ValueError(f"GroOT-MOT17 mirror only has the {SPLIT!r} split, got {split!r}.")
    return key
