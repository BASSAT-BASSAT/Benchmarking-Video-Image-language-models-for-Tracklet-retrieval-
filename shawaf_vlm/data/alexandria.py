"""Alexandria: in-house campus surveillance clips with per-identity captions.

Source layout (as exported by the annotation tool)::

    Annotations/dayN-<timestamp>.zip      outer zip, one per recording day
        dayN/<k>.zip                       inner zip, one per clip k
            annotations.json identities.json reid_index.json
            frames/frame_XXXXXX.jpg labels/frame_XXXXXX.txt
    Videos-taken_by_us/Day <word>/<k>[ name].MOV|mp4   raw video, clip k

``unpack_alexandria`` extracts each clip into ``<source>/dayN_<k>/`` so the
generic VRush loader in ``local_annotations`` can crop tracklets from it.
``prepare_alexandria`` writes crops and split tables into a prepared root that
is kept apart from ``datasets/shawaf_annotations`` (older, partly overlapping
export). Configs: ``all``, ``day1`` … ``day5`` and one per clip (``day1_3``).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

from shawaf_vlm.data.hub_video import read_jsonl
from shawaf_vlm.data.local_annotations import (
    METADATA_SUBDIR,
    SPLIT_NAME,
    _write_jsonl,
    list_annotation_projects,
    load_local_annotations,
    prepare_project,
)
from shawaf_vlm.data.tv_mars import TVMarsSplits

SOURCE_DIRNAME = "alexandria"
PREPARED_DIRNAME = "alexandria"
MANIFEST_NAME = "alexandria_manifest.json"
VIDEO_SIDECAR = "source_video.json"

_DAY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_OUTER_RE = re.compile(r"^day(\d+)", re.IGNORECASE)
_INNER_RE = re.compile(r"^day(\d+)/(\d+)\.zip$", re.IGNORECASE)
_VIDEO_RE = re.compile(r"^\s*(\d+)")
_VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
_PROJECT_RE = re.compile(r"^day(\d+)_(\d+)$")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def default_zip_dir() -> Path:
    return repo_root() / "Annotations"


def default_source_root() -> Path:
    return repo_root() / "Annotations" / SOURCE_DIRNAME


def default_prepared_root() -> Path:
    return repo_root() / "datasets" / PREPARED_DIRNAME


def default_video_root() -> Path:
    return repo_root() / "Videos-taken_by_us"


def _day_number(folder_name: str) -> int | None:
    parts = folder_name.strip().lower().split()
    if len(parts) == 2 and parts[0] == "day":
        word = parts[1]
        if word.isdigit():
            return int(word)
        return _DAY_WORDS.get(word)
    match = _OUTER_RE.match(folder_name.strip())
    return int(match.group(1)) if match else None


def index_videos(video_root: Path | str | None = None) -> dict[tuple[int, int], Path]:
    """Map ``(day, clip)`` to the raw video whose name starts with the clip number."""

    root = Path(video_root) if video_root is not None else default_video_root()
    found: dict[tuple[int, int], Path] = {}
    if not root.is_dir():
        return found
    for day_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        day = _day_number(day_dir.name)
        if day is None:
            continue
        for video in sorted(day_dir.iterdir()):
            if not video.is_file() or video.suffix.lower() not in _VIDEO_EXTS:
                continue
            match = _VIDEO_RE.match(video.name)
            if match:
                found.setdefault((day, int(match.group(1))), video)
    return found


def unpack_alexandria(
    zip_dir: Path | str | None = None,
    source_root: Path | str | None = None,
    video_root: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Extract every ``dayN/<k>.zip`` into ``<source_root>/dayN_<k>/``. Idempotent."""

    zips = Path(zip_dir) if zip_dir is not None else default_zip_dir()
    dest = Path(source_root) if source_root is not None else default_source_root()
    dest.mkdir(parents=True, exist_ok=True)
    videos = index_videos(video_root)

    entries: list[dict[str, object]] = []
    for outer in sorted(zips.glob("*.zip")):
        if not _OUTER_RE.match(outer.name):
            continue
        with zipfile.ZipFile(outer) as outer_zip:
            for name in sorted(outer_zip.namelist()):
                match = _INNER_RE.match(name.replace("\\", "/"))
                if not match:
                    continue
                day, clip = int(match.group(1)), int(match.group(2))
                project = dest / f"day{day}_{clip}"
                if overwrite or not (project / "reid_index.json").is_file():
                    project.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(io.BytesIO(outer_zip.read(name))) as inner:
                        _safe_extract(inner, project)
                video = videos.get((day, clip))
                sidecar = {"day": day, "clip": clip, "video": str(video) if video else None}
                (project / VIDEO_SIDECAR).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                entries.append({**sidecar, "project": project.name, **_project_stats(project)})

    entries.sort(key=lambda e: (int(e["day"]), int(e["clip"])))
    (dest / MANIFEST_NAME).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def _safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    base = dest.resolve()
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        if base not in target.parents and target != base:
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")
    archive.extractall(dest)


def _project_stats(project: Path) -> dict[str, object]:
    reid = project / "reid_index.json"
    ann = project / "annotations.json"
    stats: dict[str, object] = {"identities": 0, "captioned": 0, "instances": 0}
    if reid.is_file():
        identities = json.loads(reid.read_text(encoding="utf-8")).get("identities", {})
        stats["identities"] = len(identities)
        stats["captioned"] = sum(1 for v in identities.values() if str(v.get("caption", "")).strip())
        stats["instances"] = sum(len(v.get("instances", [])) for v in identities.values())
    if ann.is_file():
        payload = json.loads(ann.read_text(encoding="utf-8"))
        stats["video_name"] = payload.get("videoName")
        stats["sample_fps"] = payload.get("sampleFps")
        stats["duration_s"] = payload.get("durationSec")
        stats["n_frames"] = len(payload.get("frames", []))
    return stats


def prepare_alexandria(
    source_root: Path | str | None = None,
    prepared_root: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Crop tracklets and write ``all``, per-day and per-clip split tables."""

    src = Path(source_root) if source_root is not None else default_source_root()
    dest = Path(prepared_root) if prepared_root is not None else default_prepared_root()
    dest.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    by_day: dict[int, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    for project in list_annotation_projects(src):
        match = _PROJECT_RE.match(project.name)
        if not match:
            continue
        path = prepare_project(project, dest, overwrite=overwrite)
        if path is None:
            continue
        written.append(path)
        rows = read_jsonl(path)
        by_day.setdefault(int(match.group(1)), []).extend(rows)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(f"No Alexandria tracklets prepared from {src}. Run unpack first.")

    meta = dest / METADATA_SUBDIR
    for day, rows in sorted(by_day.items()):
        path = meta / f"day{day}-{SPLIT_NAME}.jsonl"
        _write_jsonl(path, rows)
        written.append(path)
    combined = meta / f"all-{SPLIT_NAME}.jsonl"
    _write_jsonl(combined, all_rows)
    written.append(combined)
    return written


def load_alexandria(
    config: str = "all",
    split: str = SPLIT_NAME,
    root: Path | str | None = None,
) -> TVMarsSplits:
    prepared = Path(root) if root is not None else default_prepared_root()
    if not (prepared / METADATA_SUBDIR / f"{config}-{split}.jsonl").is_file():
        raise FileNotFoundError(
            f"Alexandria config {config!r} not prepared under {prepared}. "
            "Run scripts/prepare_alexandria.py first."
        )
    splits = load_local_annotations(config=config, split=split, root=prepared)
    query = [q for q in splits.query if q.track_id not in EXCLUDED_TRACKS]
    gallery = [g for g in splits.gallery if g.track_id not in EXCLUDED_TRACKS]
    return TVMarsSplits(query=query, gallery=gallery, source=f"alexandria:{config}/{split}")


# Tracklets removed from the benchmark by the annotator, with the reason.
# Filtered at load time, so re-running the preparation cannot bring them back.
EXCLUDED_TRACKS: dict[str, str] = {
    "day5_53/person_3": "badly labelled (annotator review, 2026-11-18)",
    "day5_53/person_4": "badly labelled (annotator review, 2026-11-18)",
}


def alexandria_status() -> str:
    meta = default_prepared_root() / METADATA_SUBDIR
    if (meta / f"all-{SPLIT_NAME}.jsonl").is_file():
        return "ready"
    if (default_source_root() / MANIFEST_NAME).is_file():
        return "unpacked (run scripts/prepare_alexandria.py)"
    return "missing"
