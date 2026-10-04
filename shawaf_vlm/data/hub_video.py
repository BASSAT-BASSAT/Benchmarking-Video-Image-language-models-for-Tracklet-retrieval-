"""Shared loader for Hugging Face video datasets laid out like bassatbassat/TVPReid.

A repo holds ``metadata/<config>-<split>.jsonl`` split tables and the mp4s
they reference. Each JSONL row has ``video`` (repo-relative mp4 path),
``video_id``, ``captions``, and ``subset``. Every row is one gallery video;
every caption is one query whose only positive is that video.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits


def hf_token(token: str | None = None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def default_download_root(dirname: str, local_dir: Path | str | None = None) -> Path:
    if local_dir is not None:
        root = Path(local_dir)
    elif Path("/kaggle/working").is_dir():
        root = Path("/kaggle/working") / dirname
    else:
        root = Path.home() / ".cache" / "shawaf_vlm" / dirname
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_hub_video_dataset(
    repo_id: str,
    configs: tuple[str, ...],
    split: str,
    dest: Path,
    token: str | None = None,
    max_workers: int = 4,
) -> Path:
    """Download the split tables for ``configs``, then only the mp4s they list."""

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    auth_token = hf_token(token)
    jsonl_names = [f"metadata/{config}-{split}.jsonl" for config in configs]
    print(f"Hub repo : {repo_id}", flush=True)
    print(f"Split    : {split} only", flush=True)
    print(f"Configs  : {', '.join(configs)}", flush=True)

    for name in tqdm(jsonl_names, desc="Metadata", unit="file"):
        hub_file(repo_id=repo_id, filename=name, token=auth_token, dest=dest)

    video_rels: list[str] = []
    for name in jsonl_names:
        video_rels.extend(video_rels_from_jsonl(dest / name))
    video_rels = sorted(set(video_rels))
    missing = [rel for rel in video_rels if not (dest / rel).is_file()]
    print(f"Videos   : {len(video_rels)} files ({len(missing)} to fetch)", flush=True)

    def _fetch(rel: str) -> str:
        return hub_file(repo_id=repo_id, filename=rel, token=auth_token, dest=dest)

    if missing:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [pool.submit(_fetch, rel) for rel in missing]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Videos {split}",
                unit="file",
            ):
                future.result()
    return dest


def hub_file(
    repo_id: str,
    filename: str,
    token: str | None,
    dest: Path,
    retries: int = 6,
) -> str:
    import time

    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
                local_dir=str(dest),
            )
        except Exception as exc:
            last = exc
            # HF Xet token endpoint returns 429 when >1000 file requests / 5 min.
            # TVPReid (820) + GroOT (454) in one session trips it with 8 workers.
            if "429" not in str(exc) and "Too Many Requests" not in str(exc):
                raise
            wait = min(300, 10 * (2**attempt))
            print(
                f"HF rate limit (429) on {filename}, retry {attempt + 1}/{retries} "
                f"in {wait}s...",
                flush=True,
            )
            time.sleep(wait)
    assert last is not None
    raise last


def load_hub_video_split(
    root: Path | str,
    config: str,
    split: str,
    source: str,
    missing: Callable[[str], Exception],
    name: str,
) -> TVMarsSplits:
    root = Path(root)
    jsonl = root / "metadata" / f"{config}-{split}.jsonl"
    if not jsonl.is_file():
        raise missing(f"Missing split table {jsonl}")
    queries, gallery = rows_to_splits(read_jsonl(jsonl), root)
    if not queries or not gallery:
        raise RuntimeError(
            f"No {name} {config}/{split} samples with existing video files under {root}."
        )
    return TVMarsSplits(query=queries, gallery=gallery, source=source)


def rows_to_splits(
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
            if caption is None:
                continue
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


def video_rels_from_jsonl(path: Path | str) -> list[str]:
    rels: list[str] = []
    for row in read_jsonl(Path(path)):
        rel = str(row.get("video", "")).replace("\\", "/")
        if rel:
            rels.append(rel)
    return rels


def read_jsonl(path: Path) -> list[dict[str, object]]:
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
