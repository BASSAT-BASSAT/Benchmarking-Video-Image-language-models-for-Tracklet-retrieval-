"""Shared loader for Hugging Face video datasets laid out like bassatbassat/TVPReid.

A repo holds ``metadata/<config>-<split>.jsonl`` split tables and the mp4s
they reference. Each JSONL row has ``video`` (repo-relative mp4 path),
``video_id``, ``captions``, and ``subset``. Every row is one gallery video;
every caption is one query whose only positive is that video.

Hugging Face rate-limits high-volume *file* fetches: about 1000 requests / 5
minutes, and TVPReid train alone is ~5k mp4s. Each split is therefore also
published as ``archives/<config>-<split>.tar`` (see
``scripts/pack_hub_video_archives.py``) and downloaded in one request; per-file
fetching with a global 429 cooldown is only the fallback. The dataset is public;
429 is not an access error.
"""

from __future__ import annotations

import json
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits

# Shared pause when any worker hits HTTP 429.
_RATE_LOCK = threading.Lock()
_RATE_COOLDOWN_UNTIL = 0.0


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


def _disable_hf_xet() -> None:
    """Xet auth endpoints 429 under many parallel file downloads; use classic Hub."""

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def download_hub_video_dataset(
    repo_id: str,
    configs: tuple[str, ...],
    split: str,
    dest: Path,
    token: str | None = None,
    max_workers: int | None = None,
) -> Path:
    """Download the split tables for ``configs``, then only the mp4s they list."""

    _disable_hf_xet()
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
    print("HF_HUB_DISABLE_XET=1 (avoids Xet 429 storms)", flush=True)

    for name in tqdm(jsonl_names, desc="Metadata", unit="file"):
        hub_file(repo_id=repo_id, filename=name, token=auth_token, dest=dest)

    video_rels: list[str] = []
    for name in jsonl_names:
        video_rels.extend(video_rels_from_jsonl(dest / name))
    video_rels = sorted(set(video_rels))
    missing = [rel for rel in video_rels if not (dest / rel).is_file()]
    print(f"Videos   : {len(video_rels)} files ({len(missing)} to fetch)", flush=True)

    if not missing:
        return dest

    for config in configs:
        _fetch_archive(repo_id, f"archives/{config}-{split}.tar", auth_token, dest)
    missing = [rel for rel in video_rels if not (dest / rel).is_file()]
    if not missing:
        return dest
    print(f"Per-file fallback for {len(missing)} videos", flush=True)

    # Large splits (TVPReid train ~5k) must stay at 1–2 workers or HF 429s.
    if max_workers is None:
        max_workers = 1 if len(missing) > 1500 else 2 if len(missing) > 400 else 4
    max_workers = max(1, int(max_workers))
    print(f"Workers  : {max_workers}", flush=True)

    def _fetch(rel: str) -> str:
        return hub_file(repo_id=repo_id, filename=rel, token=auth_token, dest=dest)

    # Fetch in chunks so a 429 cooldown applies before the next wave.
    chunk = max(50, max_workers * 40)
    done = 0
    with tqdm(total=len(missing), desc=f"Videos {split}", unit="file") as bar:
        for start in range(0, len(missing), chunk):
            batch = missing[start : start + chunk]
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_fetch, rel) for rel in batch]
                for future in as_completed(futures):
                    future.result()
                    done += 1
                    bar.update(1)
            # Brief pause between chunks on large jobs.
            if start + chunk < len(missing) and len(missing) > 400:
                time.sleep(2.0)
    return dest


def _fetch_archive(repo_id: str, filename: str, token: str | None, dest: Path) -> bool:
    """Download + extract one split tar (one request instead of thousands)."""

    try:
        tar_path = Path(hub_file(repo_id=repo_id, filename=filename, token=token, dest=dest))
    except Exception as exc:
        if "404" in _error_text(exc) or "EntryNotFound" in type(exc).__name__:
            print(f"No {filename} on the Hub; using per-file downloads", flush=True)
            return False
        raise
    print(f"Extracting {filename} ({tar_path.stat().st_size / 2**20:.0f} MB)…", flush=True)
    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(dest)
    tar_path.unlink()
    return True


def _error_text(exc: BaseException) -> str:
    """Message of ``exc`` plus its causes (hf_hub wraps 429s in LocalEntryNotFoundError)."""

    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _is_rate_limit(exc: BaseException) -> bool:
    text = _error_text(exc)
    return any(key in text for key in ("429", "Too Many Requests", "server under pressure"))


def hub_file(
    repo_id: str,
    filename: str,
    token: str | None,
    dest: Path,
    retries: int = 8,
) -> str:
    from huggingface_hub import hf_hub_download

    global _RATE_COOLDOWN_UNTIL

    last: Exception | None = None
    for attempt in range(max(1, retries)):
        # Honor a global cooldown set by any worker that hit 429.
        with _RATE_LOCK:
            wait_for = _RATE_COOLDOWN_UNTIL - time.time()
        if wait_for > 0:
            time.sleep(wait_for)
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
            if not _is_rate_limit(exc):
                raise
            # Cool down longer than before; share the pause across workers.
            wait = min(360, 30 * (2**attempt))
            with _RATE_LOCK:
                _RATE_COOLDOWN_UNTIL = max(_RATE_COOLDOWN_UNTIL, time.time() + wait)
            print(
                f"HF rate limit (429) on {filename}, retry {attempt + 1}/{retries} "
                f"in {wait}s (global cooldown)…",
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
