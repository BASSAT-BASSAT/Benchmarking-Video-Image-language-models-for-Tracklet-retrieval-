from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits

HF_REPO_ID = "bassatbassat/TVPReid"
HF_DATASET_URL = "https://huggingface.co/datasets/bassatbassat/TVPReid"

SUBSET_FOLDERS = {
    "prid": "TVPReid-PRID",
    "ilids": "TVPReid-iLIDs",
    "duke": "TVPReid-Duke",
}

VALID_SPLITS = ("train", "val", "test")


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

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    normalized = tuple(_normalize_config(item) for item in configs)
    split = _normalize_split(split)
    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    dest = _download_root(local_dir)

    jsonl_names = [f"metadata/{config}-{split}.jsonl" for config in normalized]
    print(f"Hub repo : {repo_id}", flush=True)
    print(f"Split    : {split} only (skips train/val mp4s)", flush=True)
    print(f"Subsets  : {', '.join(normalized)}", flush=True)

    try:
        for name in tqdm(jsonl_names, desc="Metadata", unit="file"):
            _hub_file(
                repo_id=repo_id,
                filename=name,
                token=auth_token,
                dest=dest,
            )
        video_rels = []
        for name in jsonl_names:
            jsonl_path = dest / name
            video_rels.extend(video_rels_from_jsonl(jsonl_path))
        video_rels = sorted(set(video_rels))
        print(f"Videos   : {len(video_rels)} files", flush=True)

        def _fetch(rel: str) -> str:
            return _hub_file(
                repo_id=repo_id,
                filename=rel,
                token=auth_token,
                dest=dest,
            )

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [pool.submit(_fetch, rel) for rel in video_rels]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Videos {split}",
                unit="file",
            ):
                future.result()
    except Exception as exc:
        raise TVPReidAccessError(
            missing_tvpreid_message(f"Failed to download {repo_id}: {exc}")
        ) from exc
    return dest


def video_rels_from_jsonl(path: Path | str) -> list[str]:
    rels: list[str] = []
    for row in _read_jsonl(Path(path)):
        rel = str(row.get("video", "")).replace("\\", "/")
        if rel:
            rels.append(rel)
    return rels


def _download_root(local_dir: Path | str | None) -> Path:
    if local_dir is not None:
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root
    kaggle_work = Path("/kaggle/working")
    if kaggle_work.is_dir():
        root = kaggle_work / "TVPReid"
        root.mkdir(parents=True, exist_ok=True)
        return root
    cache = Path.home() / ".cache" / "shawaf_vlm" / "TVPReid"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _hub_file(
    repo_id: str,
    filename: str,
    token: str | None,
    dest: Path,
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=str(dest),
    )


def load_tvpreid_from_root(
    root: Path | str,
    config: str = "prid",
    split: str = "test",
) -> TVMarsSplits:
    root = Path(root)
    config = _normalize_config(config)
    split = _normalize_split(split)
    jsonl = root / "metadata" / f"{config}-{split}.jsonl"
    if not jsonl.is_file():
        raise TVPReidAccessError(
            missing_tvpreid_message(f"Missing split table {jsonl}")
        )

    rows = _read_jsonl(jsonl)
    queries, gallery = _rows_to_splits(rows, root)
    if not queries or not gallery:
        raise RuntimeError(
            f"No TVPReid {config}/{split} samples with existing video files "
            f"under {root}."
        )
    return TVMarsSplits(
        query=queries,
        gallery=gallery,
        source=f"tvpreid:{config}:{split}",
    )


def _rows_to_splits(
    rows: list[dict[str, object]],
    root: Path,
) -> tuple[list[CaptionQuery], list[GalleryTracklet]]:
    id_map: dict[str, int] = {}
    queries: list[CaptionQuery] = []
    gallery: list[GalleryTracklet] = []

    for row in rows:
        video_rel = str(row.get("video", "")).replace("\\", "/")
        video_id = str(row.get("video_id", Path(video_rel).stem))
        subset = str(row.get("subset", ""))
        video_path = root / video_rel
        if not video_path.is_file():
            continue
        key = f"{subset}/{video_id}"
        if key not in id_map:
            id_map[key] = len(id_map) + 1
        person_id = id_map[key]
        crop_paths = (video_path,)
        gallery.append(
            GalleryTracklet(
                crop_paths=crop_paths,
                person_id=person_id,
                camera_id=0,
                track_id=video_id,
            )
        )
        captions = row.get("captions", [])
        if not isinstance(captions, list):
            captions = [captions]
        for caption in captions:
            text = str(caption).strip()
            if not text:
                continue
            queries.append(
                CaptionQuery(
                    text=text,
                    person_id=person_id,
                    camera_id=0,
                    track_id=video_id,
                    crop_paths=crop_paths,
                )
            )
    return queries, gallery


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


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
