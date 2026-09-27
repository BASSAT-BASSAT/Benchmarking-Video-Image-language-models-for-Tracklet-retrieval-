from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits

GDRIVE_FILE_ID = "1HTeDZUVrZr6nL56ZlkYBNqjSWh3IGV2X"
RSTPREID_REPO = "https://github.com/NjtechCVLab/RSTPReid-Dataset"
CAPTIONS_FILE = "data_captions.json"
IMAGE_DIR = "imgs"
VALID_SPLITS = ("train", "val", "test")

_CAMERA_RE = re.compile(r"_c(\d+)_", re.IGNORECASE)


class RSTPReidAccessError(FileNotFoundError):
    """Raised when data_captions.json or the imgs folder is missing."""


def missing_rstpreid_message(detail: str) -> str:
    return (
        f"{detail}\n"
        f"RSTPReid: {RSTPREID_REPO} (Google Drive id {GDRIVE_FILE_ID}).\n"
        "Enable Internet on Kaggle so gdown can fetch RSTPReid.zip, or attach "
        "the unzipped dataset as a Kaggle input. Cite Zhu et al. ACM MM 2021 "
        "and MSMT17."
    )


def discover_local_rstpreid() -> Path | None:
    env = os.environ.get("SHAWAF_RSTPREID_ROOT")
    if env and _looks_like_rstpreid_root(Path(env)):
        return Path(env)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for match in kaggle_input.rglob(CAPTIONS_FILE):
            if _looks_like_rstpreid_root(match.parent):
                return match.parent

    for candidate in (
        Path("/kaggle/working/RSTPReid"),
        Path.home() / ".cache" / "shawaf_vlm" / "RSTPReid",
    ):
        if _looks_like_rstpreid_root(candidate):
            return candidate
    return None


def download_rstpreid(
    local_dir: Path | str | None = None,
    file_id: str = GDRIVE_FILE_ID,
) -> Path:
    """Return an RSTPReid root, downloading and unzipping the Drive archive if needed."""

    found = discover_local_rstpreid() if local_dir is None else None
    if found is not None:
        print(f"RSTPReid : {found}", flush=True)
        return found

    dest = _download_root(local_dir)
    if _looks_like_rstpreid_root(dest):
        print(f"RSTPReid : {dest}", flush=True)
        return dest

    archive = dest / "RSTPReid.zip"
    if not archive.is_file():
        try:
            import gdown
        except ImportError as exc:
            raise RSTPReidAccessError(
                missing_rstpreid_message(
                    'gdown is required. Install with pip install "shawaf-vlm[all]".'
                )
            ) from exc
        print(f"RSTPReid : downloading Drive id {file_id}", flush=True)
        result = gdown.download(id=file_id, output=str(archive), quiet=False)
        if result is None or not archive.is_file():
            raise RSTPReidAccessError(
                missing_rstpreid_message("gdown could not download RSTPReid.zip")
            )

    with zipfile.ZipFile(archive) as handle:
        handle.extractall(dest)
    root = _find_root(dest)
    if root is None:
        raise RSTPReidAccessError(
            missing_rstpreid_message(f"No {CAPTIONS_FILE} inside {archive}")
        )
    archive.unlink(missing_ok=True)
    print(f"RSTPReid : {root}", flush=True)
    return root


def load_rstpreid(
    root: Path | str | None = None,
    split: str = "test",
) -> TVMarsSplits:
    """
    Load RSTPReid captions and images for one split.

    Each image is a one-frame gallery tracklet. Each caption is a query whose
    positives are every image of the same identity (five per ID on test).
    """

    split = _normalize_split(split)
    base = Path(root) if root is not None else download_rstpreid()
    found = _find_root(base)
    if found is None:
        raise RSTPReidAccessError(
            missing_rstpreid_message(f"Missing {CAPTIONS_FILE} under {base}")
        )

    with (found / CAPTIONS_FILE).open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    queries: list[CaptionQuery] = []
    gallery: list[GalleryTracklet] = []
    for record in records:
        if str(record.get("split", "")).lower() != split:
            continue
        image = _image_path(found, str(record.get("img_path", "")))
        if not image.is_file():
            continue
        person_id = int(record["id"])
        camera_id = camera_id_from_name(image.name)
        crop_paths = (image,)
        track_id = image.stem
        gallery.append(
            GalleryTracklet(
                crop_paths=crop_paths,
                person_id=person_id,
                camera_id=camera_id,
                track_id=track_id,
            )
        )
        captions = record.get("captions", [])
        if not isinstance(captions, list):
            captions = [captions]
        for caption in captions:
            text = str(caption or "").strip()
            if not text:
                continue
            queries.append(
                CaptionQuery(
                    text=text,
                    person_id=person_id,
                    camera_id=camera_id,
                    track_id=track_id,
                    crop_paths=crop_paths,
                )
            )

    if not queries or not gallery:
        raise RuntimeError(f"No RSTPReid {split} samples with images under {found}.")
    return TVMarsSplits(query=queries, gallery=gallery, source=f"rstpreid:{split}")


def camera_id_from_name(name: str) -> int:
    match = _CAMERA_RE.search(name)
    return int(match.group(1)) if match else 0


def _image_path(root: Path, img_path: str) -> Path:
    rel = img_path.replace("\\", "/").lstrip("/")
    if rel.startswith(f"{IMAGE_DIR}/"):
        return root / rel
    return root / IMAGE_DIR / rel


def _find_root(base: Path) -> Path | None:
    if _looks_like_rstpreid_root(base):
        return base
    if not base.is_dir():
        return None
    for match in base.rglob(CAPTIONS_FILE):
        if _looks_like_rstpreid_root(match.parent):
            return match.parent
    return None


def _looks_like_rstpreid_root(path: Path) -> bool:
    return (path / CAPTIONS_FILE).is_file() and (path / IMAGE_DIR).is_dir()


def _download_root(local_dir: Path | str | None) -> Path:
    if local_dir is not None:
        root = Path(local_dir)
    elif Path("/kaggle/working").is_dir():
        root = Path("/kaggle/working/RSTPReid")
    else:
        root = Path.home() / ".cache" / "shawaf_vlm" / "RSTPReid"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_split(split: str) -> str:
    key = split.strip().lower()
    if key == "validation":
        key = "val"
    if key not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Use train, val, or test.")
    return key
