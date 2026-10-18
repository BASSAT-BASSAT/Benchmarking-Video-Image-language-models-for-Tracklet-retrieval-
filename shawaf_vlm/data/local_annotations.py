"""Load VRush / SHAWAF manual annotation exports as TVMarsSplits tracklets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from shawaf_vlm.data.hub_video import read_jsonl
from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits

PREPARED_DIRNAME = "shawaf_annotations"
METADATA_SUBDIR = "metadata"
PREPARED_SUBDIR = "prepared"
SPLIT_NAME = "test"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    text = name.strip().lower()
    text = _SLUG_RE.sub("_", text)
    return text.strip("_") or "project"


def discover_annotations_source() -> Path | None:
    env = os.environ.get("SHAWAF_ANNOTATIONS_ROOT")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Annotations"
        if _looks_like_annotations_root(candidate):
            return candidate
    return None


def _repo_roots() -> list[Path]:
    here = Path(__file__).resolve()
    roots: list[Path] = []
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() or (parent / "Annotations").is_dir():
            roots.append(parent)
    return roots


def discover_prepared_root() -> Path | None:
    env = os.environ.get("SHAWAF_LOCAL_DATA_ROOT")
    if env:
        path = Path(env)
        if _looks_like_prepared_root(path):
            return path
    for parent in _repo_roots():
        candidate = parent / "datasets" / PREPARED_DIRNAME
        if _looks_like_prepared_root(candidate):
            return candidate
    cache = Path.home() / ".cache" / "shawaf_vlm" / PREPARED_DIRNAME
    if _looks_like_prepared_root(cache):
        return cache
    wrong = Path(__file__).resolve().parent / "datasets" / PREPARED_DIRNAME
    if _looks_like_prepared_root(wrong):
        return wrong
    return None


def default_prepared_root() -> Path:
    found = discover_prepared_root()
    if found is not None:
        return found
    for parent in _repo_roots():
        root = parent / "datasets" / PREPARED_DIRNAME
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = Path.home() / ".cache" / "shawaf_vlm" / PREPARED_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_annotation_projects(source: Path | str) -> list[Path]:
    root = Path(source)
    if not root.is_dir():
        return []
    projects: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "reid_index.json").is_file():
            projects.append(child)
    return projects


def list_prepared_configs(root: Path | str | None = None) -> list[str]:
    prepared = Path(root) if root is not None else discover_prepared_root()
    if prepared is None:
        return []
    meta = prepared / METADATA_SUBDIR
    if not meta.is_dir():
        return []
    configs: list[str] = []
    for path in sorted(meta.glob(f"*-{SPLIT_NAME}.jsonl")):
        name = path.stem[: -(len(SPLIT_NAME) + 1)]
        if name:
            configs.append(name)
    return configs


def prepare_project(
    project_dir: Path | str,
    dest_root: Path | str,
    *,
    overwrite: bool = False,
) -> Path | None:
    """Crop person tracklets from ``reid_index.json`` into ``dest_root/prepared/``."""

    project_dir = Path(project_dir)
    dest_root = Path(dest_root)
    reid_path = project_dir / "reid_index.json"
    if not reid_path.is_file():
        raise FileNotFoundError(f"Missing {reid_path}")

    payload = json.loads(reid_path.read_text(encoding="utf-8"))
    identities = payload.get("identities", {})
    if not isinstance(identities, dict):
        raise RuntimeError(f"Invalid identities in {reid_path}")

    slug = slugify(project_dir.name)
    out_project = dest_root / PREPARED_SUBDIR / slug
    rows: list[dict[str, object]] = []

    from PIL import Image

    for reid_id, identity in identities.items():
        if not isinstance(identity, dict):
            continue
        instances = identity.get("instances", [])
        if not isinstance(instances, list) or not instances:
            continue
        caption = str(identity.get("caption", "")).strip()
        track_dir = out_project / str(reid_id)
        track_dir.mkdir(parents=True, exist_ok=True)
        crop_paths: list[str] = []

        for index, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            frame_rel = str(inst.get("frameFile", "")).replace("\\", "/")
            bbox = inst.get("bbox")
            if not frame_rel or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            frame_path = project_dir / frame_rel
            if not frame_path.is_file():
                continue
            out_name = f"crop_{index:06d}.jpg"
            out_path = track_dir / out_name
            if overwrite or not out_path.is_file():
                _save_crop(frame_path, bbox, out_path, Image=Image)
            if out_path.is_file():
                rel = out_path.relative_to(dest_root).as_posix()
                crop_paths.append(rel)

        if not crop_paths:
            continue
        tracklet_id = f"{slug}/{reid_id}"
        row: dict[str, object] = {
            "subset": slug,
            "tracklet_id": tracklet_id,
            "project": project_dir.name,
            "reid_id": reid_id,
            "crops": crop_paths,
            "captions": [caption] if caption else [],
        }
        rows.append(row)

    if not rows:
        print(f"Skip {project_dir.name}: no tracklets with frames/bboxes.", flush=True)
        return None

    meta_dir = dest_root / METADATA_SUBDIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = meta_dir / f"{slug}-{SPLIT_NAME}.jsonl"
    _write_jsonl(jsonl_path, rows)
    return jsonl_path


def prepare_all_projects(
    source: Path | str | None = None,
    dest_root: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> list[Path]:
    src = Path(source) if source is not None else discover_annotations_source()
    if src is None:
        raise FileNotFoundError(
            "Annotations folder not found. Set SHAWAF_ANNOTATIONS_ROOT or add Annotations/ at repo root."
        )
    dest = Path(dest_root) if dest_root is not None else default_prepared_root()
    dest.mkdir(parents=True, exist_ok=True)

    jsonl_files: list[Path] = []
    all_rows: list[dict[str, object]] = []
    for project in list_annotation_projects(src):
        path = prepare_project(project, dest, overwrite=overwrite)
        if path is None:
            continue
        jsonl_files.append(path)
        all_rows.extend(read_jsonl(path))

    if not all_rows:
        raise RuntimeError(f"No tracklets prepared from any project under {src}")

    combined = dest / METADATA_SUBDIR / f"all-{SPLIT_NAME}.jsonl"
    _write_jsonl(combined, all_rows)
    jsonl_files.append(combined)
    return jsonl_files


def load_local_annotations(
    config: str = "all",
    split: str = SPLIT_NAME,
    root: Path | str | None = None,
) -> TVMarsSplits:
    prepared = Path(root) if root is not None else discover_prepared_root()
    if prepared is None:
        raise FileNotFoundError(
            "Prepared local annotations not found. Run scripts/prepare_local_annotations.py first."
        )
    jsonl = prepared / METADATA_SUBDIR / f"{config}-{split}.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"Missing split table {jsonl}")
    rows = read_jsonl(jsonl)
    queries, gallery = rows_to_local_splits(rows, prepared)
    if not gallery:
        raise RuntimeError(f"No gallery tracklets with crops under {prepared}.")
    return TVMarsSplits(
        query=queries,
        gallery=gallery,
        source=f"local_annotations:{config}/{split}",
    )


def rows_to_local_splits(
    rows: list[dict[str, object]],
    root: Path,
) -> tuple[list[CaptionQuery], list[GalleryTracklet]]:
    id_map: dict[str, int] = {}
    queries: list[CaptionQuery] = []
    gallery: list[GalleryTracklet] = []

    for row in rows:
        tracklet_id = str(row.get("tracklet_id", ""))
        if not tracklet_id:
            continue
        crops = row.get("crops", [])
        if not isinstance(crops, list):
            continue
        crop_paths: list[Path] = []
        for rel in crops:
            path = root / str(rel).replace("\\", "/")
            if path.is_file():
                crop_paths.append(path)
        if not crop_paths:
            continue
        if tracklet_id not in id_map:
            id_map[tracklet_id] = len(id_map) + 1
        person_id = id_map[tracklet_id]
        subset = str(row.get("subset", ""))
        camera_id = hash(subset) % 10_000 if subset else 0
        track_paths = tuple(crop_paths)
        gallery.append(
            GalleryTracklet(
                crop_paths=track_paths,
                person_id=person_id,
                camera_id=camera_id,
                track_id=tracklet_id,
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
                    camera_id=camera_id,
                    track_id=tracklet_id,
                    crop_paths=track_paths,
                )
            )
    return queries, gallery


def local_annotations_status() -> str:
    src = discover_annotations_source()
    prepared = discover_prepared_root()
    if prepared is not None:
        configs = list_prepared_configs(prepared)
        return f"ready ({len(configs)} configs)"
    if src is not None:
        n = len(list_annotation_projects(src))
        return f"source_only ({n} projects — run prepare_local_annotations.py)"
    return "missing"


def _save_crop(
    frame_path: Path,
    bbox: list[float],
    out_path: Path,
    *,
    Image,
) -> None:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    with Image.open(frame_path) as img:
        w, h = img.size
        left = max(0, min(w - 1, int(round(x1))))
        top = max(0, min(h - 1, int(round(y1))))
        right = max(left + 1, min(w, int(round(x2))))
        bottom = max(top + 1, min(h, int(round(y2))))
        crop = img.crop((left, top, right, bottom))
        crop.save(out_path, quality=92)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _looks_like_annotations_root(path: Path) -> bool:
    return path.is_dir() and any(
        (path / child / "reid_index.json").is_file() for child in path.iterdir() if child.is_dir()
    )


def _looks_like_prepared_root(path: Path) -> bool:
    meta = path / METADATA_SUBDIR
    if not meta.is_dir():
        return False
    return any(meta.glob(f"*-{SPLIT_NAME}.jsonl"))
