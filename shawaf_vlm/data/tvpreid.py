from __future__ import annotations

import os
from pathlib import Path

from shawaf_vlm.data.hub_video import (
    default_download_root,
    download_hub_video_dataset,
    hub_file as _hub_file,
    load_hub_video_split,
    read_jsonl as _read_jsonl,
    rows_to_splits as _rows_to_splits,
    video_rels_from_jsonl,
)
from shawaf_vlm.data.tv_mars import TVMarsSplits

HF_REPO_ID = "bassatbassat/TVPReid"
HF_DATASET_URL = "https://huggingface.co/datasets/bassatbassat/TVPReid"

SUBSET_FOLDERS = {
    "prid": "TVPReid-PRID",
    "ilids": "TVPReid-iLIDs",
    "duke": "TVPReid-Duke",
}

VALID_SPLITS = ("train", "val", "test")

__all__ = [
    "HF_DATASET_URL",
    "HF_REPO_ID",
    "SUBSET_FOLDERS",
    "TVPReidAccessError",
    "discover_local_tvpreid",
    "download_tvpreid",
    "load_tvpreid",
    "load_tvpreid_from_root",
    "missing_tvpreid_message",
    "video_rels_from_jsonl",
    "_hub_file",
    "_read_jsonl",
    "_rows_to_splits",
]


class TVPReidAccessError(FileNotFoundError):
    """Raised when the Hub snapshot or local TVPReid files are missing."""


def missing_tvpreid_message(detail: str) -> str:
    return (
        f"{detail}\n"
        f"TVPReid unofficial mirror: {HF_DATASET_URL}\n"
        "The Hub repo is public; enable Internet on Kaggle so the downloader "
        "can pull the test mp4s. Cite Zhang et al. ACM MM 2024 and the PRID / "
        "iLIDS / Duke source papers."
    )


def load_tvpreid(
    config: str = "prid",
    split: str = "test",
    root: Path | str | None = None,
    repo_id: str = HF_REPO_ID,
    token: str | None = None,
) -> TVMarsSplits:
    """
    Load TVPReid text queries and gallery videos for one subset.

    ``config`` is ``prid``, ``ilids``, or ``duke``. Evaluation is video-level:
    each caption retrieves its source mp4 (two captions per video).
    """

    config = _normalize_config(config)
    split = _normalize_split(split)
    snapshot = Path(root) if root is not None else None
    if snapshot is None:
        snapshot = discover_local_tvpreid()
    if snapshot is None:
        snapshot = download_tvpreid(
            repo_id=repo_id,
            configs=(config,),
            split=split,
            token=token,
        )
    return load_tvpreid_from_root(snapshot, config=config, split=split)


def discover_local_tvpreid() -> Path | None:
    env = os.environ.get("SHAWAF_TVPREID_ROOT")
    if env:
        path = Path(env)
        if path.is_dir():
            return path

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "datasets"
        if _looks_like_tvpreid_root(candidate):
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for match in kaggle_input.rglob("metadata"):
            if _looks_like_tvpreid_root(match.parent):
                return match.parent

    for candidate in (
        Path("/kaggle/working/TVPReid"),
        Path.home() / ".cache" / "shawaf_vlm" / "TVPReid",
    ):
        if _looks_like_tvpreid_root(candidate):
            return candidate
    return None


def download_tvpreid(
    repo_id: str = HF_REPO_ID,
    configs: tuple[str, ...] = ("prid",),
    split: str = "test",
    token: str | None = None,
    local_dir: Path | str | None = None,
    max_workers: int = 8,
) -> Path:
    """Download only the requested split's videos, with a tqdm file bar."""

    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise TVPReidAccessError(
            missing_tvpreid_message(
                "huggingface_hub is required to download TVPReid. "
                'Install with pip install "shawaf-vlm[all]".'
            )
        ) from exc

    normalized = tuple(_normalize_config(item) for item in configs)
    split = _normalize_split(split)
    dest = default_download_root("TVPReid", local_dir)
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
        raise TVPReidAccessError(
            missing_tvpreid_message(f"Failed to download {repo_id}: {exc}")
        ) from exc


def load_tvpreid_from_root(
    root: Path | str,
    config: str = "prid",
    split: str = "test",
) -> TVMarsSplits:
    config = _normalize_config(config)
    split = _normalize_split(split)
    return load_hub_video_split(
        root,
        config=config,
        split=split,
        source=f"tvpreid:{config}:{split}",
        missing=lambda detail: TVPReidAccessError(missing_tvpreid_message(detail)),
        name="TVPReid",
    )


def _looks_like_tvpreid_root(path: Path) -> bool:
    metadata = path / "metadata"
    if not metadata.is_dir():
        return False
    return any(
        (metadata / f"{config}-test.jsonl").is_file()
        for config in SUBSET_FOLDERS
    )


def _normalize_config(config: str) -> str:
    alias = {
        "prid": "prid",
        "prid-2011": "prid",
        "ilids": "ilids",
        "ilids-vid": "ilids",
        "ilids_vid": "ilids",
        "duke": "duke",
        "dukemtmc": "duke",
    }
    key = config.strip().lower().replace(" ", "")
    if key not in alias:
        raise ValueError(
            f"Unknown TVPReid config {config!r}. Use prid, ilids, or duke."
        )
    return alias[key]


def _normalize_split(split: str) -> str:
    key = split.strip().lower()
    if key == "validation":
        key = "val"
    if key not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Use train, val, or test.")
    return key
