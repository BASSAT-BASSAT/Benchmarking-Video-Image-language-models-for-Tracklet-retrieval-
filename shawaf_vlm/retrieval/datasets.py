"""Load Tier-A benchmark splits for indexing and demo search."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from shawaf_vlm.data.groot import load_groot
from shawaf_vlm.data.rstpreid import discover_local_rstpreid, download_rstpreid, load_rstpreid
from shawaf_vlm.data.tv_mars import TVMarsSplits
from shawaf_vlm.data.local_annotations import list_prepared_configs, local_annotations_status
from shawaf_vlm.data.tvpreid import download_tvpreid, load_tvpreid


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    subset: str
    modality: str


# Families run by kaggle-zero-shot.ipynb / embeddinggemma2_tier_a_zero_shot.ipynb
TIER_A_ZERO_SHOT_FAMILIES: tuple[str, ...] = ("tvpreid", "rstpreid", "groot_mot17")

LOCAL_ANNOTATIONS_FAMILY = "local"

TIER_A_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("tvpreid:prid", "TVPReid PRID", "prid", "video"),
    DatasetSpec("tvpreid:ilids", "TVPReid iLIDS", "ilids", "video"),
    DatasetSpec("tvpreid:duke", "TVPReid Duke", "duke", "video"),
    DatasetSpec("rstpreid:test", "RSTPReid test", "test", "image"),
    DatasetSpec("groot:combined", "GroOT-MOT17 combined", "combined", "video"),
    DatasetSpec("groot:appearance", "GroOT-MOT17 appearance", "appearance", "video"),
    DatasetSpec("groot:action", "GroOT-MOT17 action", "action", "video"),
    DatasetSpec("groot:all", "GroOT-MOT17 all", "all", "video"),
)


def tier_a_and_local_datasets() -> tuple[DatasetSpec, ...]:
    """Tier-A benchmarks plus prepared local annotation configs (``local:<slug>``)."""

    extra: list[DatasetSpec] = []
    for config in list_prepared_configs():
        if config == "all":
            label = "Local annotations (all projects)"
        else:
            label = f"Local annotations — {config.replace('_', ' ')}"
        extra.append(
            DatasetSpec(
                f"{LOCAL_ANNOTATIONS_FAMILY}:{config}",
                label,
                config,
                "image",
            )
        )
    return TIER_A_DATASETS + tuple(extra)


def parse_dataset_key(key: str) -> tuple[str, str]:
    if ":" not in key:
        raise ValueError(f"Dataset key must be 'family:subset', got {key!r}")
    family, subset = key.split(":", 1)
    return family, subset


def load_splits(
    dataset_key: str,
    split: str = "test",
    root: Path | str | None = None,
    token: str | None = None,
    download: bool = False,
) -> TVMarsSplits:
    family, subset = parse_dataset_key(dataset_key)
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if family == "tvpreid":
        if root is None and download:
            from shawaf_vlm.data.tvpreid import discover_local_tvpreid

            if discover_local_tvpreid() is None:
                download_tvpreid(configs=(subset,), split=split, token=token)
        return load_tvpreid(config=subset, split=split, root=root, token=token)

    if family == "rstpreid":
        if root is None:
            root = discover_local_rstpreid()
        if root is None and download:
            root = download_rstpreid()
        if root is None:
            raise FileNotFoundError("RSTPReid not found; download from Google Drive.")
        return load_rstpreid(root, split=subset)

    if family == "groot":
        if root is None and download:
            from shawaf_vlm.data.groot import discover_local_groot, download_groot

            if discover_local_groot() is None:
                download_groot(configs=(subset,), split=split, token=token)
        return load_groot(config=subset, split=split, root=root, token=token)

    if family == LOCAL_ANNOTATIONS_FAMILY:
        from shawaf_vlm.data.local_annotations import load_local_annotations

        return load_local_annotations(config=subset, split=split, root=root)

    raise ValueError(f"Unknown dataset family {family!r}")


def dataset_status() -> dict[str, str]:
    from shawaf_vlm.data.groot import discover_local_groot
    from shawaf_vlm.data.tvpreid import discover_local_tvpreid

    tv = "ready" if discover_local_tvpreid() else "missing"
    groot = "ready" if discover_local_groot() else "missing"
    rstp = "ready" if discover_local_rstpreid() else "missing"
    return {
        "tvpreid": tv,
        "groot": groot,
        "rstpreid": rstp,
        "local_annotations": local_annotations_status(),
    }
