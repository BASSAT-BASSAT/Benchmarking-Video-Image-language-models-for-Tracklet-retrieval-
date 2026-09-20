from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

AGREEMENT_URL = (
    "https://github.com/Tedysu0916/TV-MARS/raw/main/TV-MARS%20Agreement.pdf"
)
CONTACT_EMAIL = "jiajunsu@stu.hqu.edu.cn"
TV_MARS_REPO = "https://github.com/Tedysu0916/TV-MARS"

_BBOX_MARKERS = ("bbox_train/", "bbox_test/")
_CAMERA_RE = re.compile(r"C(\d)", re.IGNORECASE)


class CaptionFilesNotFound(FileNotFoundError):
    """Raised when TV-MARS caption JSON folders are missing."""


@dataclass(frozen=True)
class CaptionQuery:
    text: str
    person_id: int
    camera_id: int
    track_id: str
    crop_paths: tuple[Path, ...]


@dataclass(frozen=True)
class GalleryTracklet:
    crop_paths: tuple[Path, ...]
    person_id: int
    camera_id: int
    track_id: str


@dataclass(frozen=True)
class TVMarsSplits:
    query: list[CaptionQuery]
    gallery: list[GalleryTracklet]
    source: str


def remap_crop_path(raw_path: str, data_root: Path) -> Path:
    """Map an author-machine path onto local bbox_train / bbox_test crops."""

    normalized = str(raw_path).replace("\\", "/")
    for marker in _BBOX_MARKERS:
        index = normalized.find(marker)
        if index != -1:
            return data_root / normalized[index:]
    return data_root / normalized.lstrip("/")


def camera_id_from_crop(path: Path | str) -> int:
    match = _CAMERA_RE.search(Path(path).name)
    if match is None:
        return 0
    return int(match.group(1))


def parse_person_id(value: object) -> int:
    text = str(value).strip()
    return int(text)


def missing_captions_message(ann_root: Path) -> str:
    return (
        f"TV-MARS captions were not found at {ann_root}.\n"
        "MARS crops (bbox_train / bbox_test) are public, but the text files "
        "(train_info, test_query_info, test_gallery_info) are gated.\n"
        f"Sign the agreement: {AGREEMENT_URL}\n"
        f"Email: {CONTACT_EMAIL}\n"
        f"Until the full splits arrive, you can smoke-test with the public "
        f"partical_dataset folder from {TV_MARS_REPO}."
    )


def load_tv_mars(
    data_root: Path | str,
    ann_root: Path | str,
) -> TVMarsSplits:
    """
    Load TV-MARS text queries and gallery tracklets.

    ``ann_root`` may be:

    * the official root containing ``test_query_info`` and
      ``test_gallery_info``
    * a flat folder of the same JSON files (for example the public
      ``partical_dataset`` smoke split)
    """

    data_root = Path(data_root)
    ann_root = Path(ann_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"MARS crop root is not available: {data_root}")
    if not ann_root.exists():
        raise CaptionFilesNotFound(missing_captions_message(ann_root))

    query_dir = ann_root / "test_query_info"
    gallery_dir = ann_root / "test_gallery_info"
    if query_dir.is_dir() and gallery_dir.is_dir():
        records = _load_json_dir(query_dir)
        gallery_records = _load_json_dir(gallery_dir)
        return TVMarsSplits(
            query=_explode_captions(records, data_root),
            gallery=_to_gallery(gallery_records, data_root),
            source="official",
        )

    json_files = sorted(ann_root.glob("*.json"))
    if json_files:
        records = _load_json_files(json_files)
        return TVMarsSplits(
            query=_explode_captions(records, data_root),
            gallery=_to_gallery(records, data_root),
            source="flat_json",
        )

    raise CaptionFilesNotFound(missing_captions_message(ann_root))


def _load_json_dir(directory: Path) -> list[dict[str, object]]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise CaptionFilesNotFound(
            f"No JSON files in {directory}.\n{missing_captions_message(directory.parent)}"
        )
    return _load_json_files(files)


def _load_json_files(files: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            records.append(payload)
        else:
            raise ValueError(f"Unexpected JSON payload in {path}")
    return records


def _explode_captions(
    records: list[dict[str, object]],
    data_root: Path,
) -> list[CaptionQuery]:
    queries: list[CaptionQuery] = []
    for record in records:
        crop_paths = _crop_paths(record, data_root)
        if not crop_paths:
            continue
        person_id = parse_person_id(record.get("person_id", "0"))
        track_id = str(record.get("track_id", ""))
        camera_id = camera_id_from_crop(crop_paths[0])
        captions = record.get("captions", [])
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
                    camera_id=camera_id,
                    track_id=track_id,
                    crop_paths=crop_paths,
                )
            )
    if not queries:
        raise RuntimeError("No caption queries could be built from the annotation files.")
    return queries


def _to_gallery(
    records: list[dict[str, object]],
    data_root: Path,
) -> list[GalleryTracklet]:
    gallery: list[GalleryTracklet] = []
    for record in records:
        crop_paths = _crop_paths(record, data_root)
        if not crop_paths:
            continue
        gallery.append(
            GalleryTracklet(
                crop_paths=crop_paths,
                person_id=parse_person_id(record.get("person_id", "0")),
                camera_id=camera_id_from_crop(crop_paths[0]),
                track_id=str(record.get("track_id", "")),
            )
        )
    if not gallery:
        raise RuntimeError("No gallery tracklets could be built from the annotation files.")
    return gallery


def _crop_paths(record: dict[str, object], data_root: Path) -> tuple[Path, ...]:
    raw_paths = record.get("img_path", [])
    if not isinstance(raw_paths, list):
        return ()
    mapped: list[Path] = []
    for raw in raw_paths:
        path = remap_crop_path(str(raw), data_root)
        if path.is_file():
            mapped.append(path)
    return tuple(mapped)
