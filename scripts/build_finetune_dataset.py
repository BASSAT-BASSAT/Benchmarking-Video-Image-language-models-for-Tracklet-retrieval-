#!/usr/bin/env python3
"""Package public person-description data for Kaggle fine-tuning.

Writes::

    finetune_data/<source>/images/...
    finetune_data/<source>/train.jsonl
    finetune_data/<source>/val.jsonl

Never includes Alexandria, GroOT, or any *test* split.

Examples::

    python scripts/build_finetune_dataset.py --source rstpreid
    python scripts/build_finetune_dataset.py --source tvpreid
    python scripts/build_finetune_dataset.py --source cuhk_pedes --hf MaulikMadhavi/CUHK-PEDES-processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.data.rstpreid import download_rstpreid, load_rstpreid  # noqa: E402
from shawaf_vlm.data.tvpreid import download_tvpreid  # noqa: E402
from shawaf_vlm.data.hub_video import read_jsonl  # noqa: E402
from shawaf_vlm.sampling import _read_uniform_rgb_frames  # noqa: E402

TARGET_HEIGHT = 384
TVPREID_CONFIGS = ("prid", "ilids", "duke")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        required=True,
        choices=("rstpreid", "tvpreid", "cuhk_pedes", "icfg_pedes"),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "finetune_data")
    parser.add_argument("--hf", default=None, help="Hugging Face dataset id for CUHK / ICFG")
    parser.add_argument("--max-images", type=int, default=None, help="Cap for smoke tests")
    parser.add_argument("--height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--frames-per-video", type=int, default=4, help="TVPReid: frames kept per train/val video")
    args = parser.parse_args()

    dest = args.out / args.source
    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "rstpreid":
        train_rows, val_rows = _package_rstpreid(images_dir, args.height, args.max_images)
    elif args.source == "tvpreid":
        train_rows, val_rows = _package_tvpreid(
            images_dir, args.height, args.max_images, args.frames_per_video
        )
    elif args.source == "cuhk_pedes":
        hf_id = args.hf or "MaulikMadhavi/CUHK-PEDES-processed"
        train_rows, val_rows = _package_cuhk(images_dir, hf_id, args.height, args.max_images)
    else:
        hf_id = args.hf or "cjc/ICFG-PEDES"
        train_rows, val_rows = _package_icfg(images_dir, hf_id, args.height, args.max_images)

    _write_jsonl(dest / "train.jsonl", train_rows)
    _write_jsonl(dest / "val.jsonl", val_rows)
    print(
        f"wrote {dest}: train={len(train_rows)} val={len(val_rows)} images under {images_dir}",
        flush=True,
    )
    print(
        "Upload as a private Kaggle dataset:\n"
        f"  kaggle datasets create -p {dest}  # after adding dataset-metadata.json",
        flush=True,
    )


def _package_rstpreid(images_dir: Path, height: int, max_images: int | None):
    root = download_rstpreid()
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    # Prefer raw captions file so we keep multi-caption rows intact.
    captions_path = root / "data_captions.json"
    records = json.loads(captions_path.read_text(encoding="utf-8"))
    count = 0
    for record in records:
        split = str(record.get("split", "")).lower()
        if split == "validation":
            split = "val"
        if split not in ("train", "val"):
            continue  # never package test
        img_rel = str(record.get("img_path", "")).replace("\\", "/").lstrip("/")
        src = root / img_rel if img_rel.startswith("imgs/") else root / "imgs" / Path(img_rel).name
        if not src.is_file():
            # load_rstpreid-compatible fallback
            from shawaf_vlm.data.rstpreid import _image_path

            src = _image_path(root, str(record.get("img_path", "")))
        if not src.is_file():
            continue
        person_id = f"rstpreid_{int(record['id'])}"
        captions = record.get("captions") or []
        if isinstance(captions, str):
            captions = [captions]
        captions = [str(c).strip() for c in captions if str(c).strip()]
        if not captions:
            continue
        rel_name = f"{person_id}_{src.stem}.jpg"
        _resize_copy(src, images_dir / rel_name, height)
        row = {
            "image": f"images/{rel_name}",
            "person_id": person_id,
            "captions": captions,
            "split": split,
            "source": "rstpreid",
        }
        (train_rows if split == "train" else val_rows).append(row)
        count += 1
        if max_images is not None and count >= max_images:
            break
    if not train_rows:
        # Sanity: ensure loader sees train via public API
        _ = load_rstpreid(root, split="train")
        raise RuntimeError("No RSTPReid train rows packaged")
    return train_rows, val_rows


def _package_tvpreid(
    images_dir: Path,
    height: int,
    max_images: int | None,
    frames_per_video: int,
):
    """Download TVPReid train+val (not test); extract uniform frames as stills."""

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    count = 0
    for split in ("train", "val"):
        root = download_tvpreid(configs=TVPREID_CONFIGS, split=split)
        for config in TVPREID_CONFIGS:
            jsonl = root / "metadata" / f"{config}-{split}.jsonl"
            if not jsonl.is_file():
                print(f"skip missing {jsonl}", flush=True)
                continue
            for row in read_jsonl(jsonl):
                video_rel = str(row.get("video", "")).replace("\\", "/")
                video_path = root / video_rel
                if not video_path.is_file():
                    continue
                video_id = str(row.get("video_id", video_path.stem))
                person_id = f"tvpreid_{config}_{video_id}"
                captions = row.get("captions") or []
                if isinstance(captions, str):
                    captions = [captions]
                captions = [str(c).strip() for c in captions if str(c).strip()]
                if not captions:
                    continue
                frames = _read_uniform_rgb_frames(video_path, max(1, frames_per_video))
                if not frames:
                    continue
                for fi, frame in enumerate(frames):
                    rel_name = f"{person_id}_f{fi:02d}.jpg"
                    _save_array(frame, images_dir / rel_name, height)
                    record = {
                        "image": f"images/{rel_name}",
                        "person_id": person_id,
                        "captions": captions,
                        "split": split,
                        "source": "tvpreid",
                    }
                    (train_rows if split == "train" else val_rows).append(record)
                    count += 1
                    if max_images is not None and count >= max_images:
                        return train_rows, val_rows
    if not train_rows:
        raise RuntimeError("No TVPReid train frames packaged")
    return train_rows, val_rows


def _save_array(frame: np.ndarray, dest: Path, height: int) -> None:
    _save_pil(Image.fromarray(frame.astype(np.uint8)), dest, height)


def _package_cuhk(images_dir: Path, hf_id: str, height: int, max_images: int | None):
    """HF ``MaulikMadhavi/CUHK-PEDES-processed``: train split only (filename[0:4] = id)."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Install datasets: pip install "datasets"') from exc

    ds = load_dataset(hf_id, split="train")
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    # Hold out 5% of person ids for val (deterministic).
    person_ids = sorted({_cuhk_pid(row) for row in ds})
    val_ids = set(person_ids[::20])  # ~5%
    count = 0
    for row in ds:
        pid = _cuhk_pid(row)
        split = "val" if pid in val_ids else "train"
        image = row.get("image")
        if image is None:
            continue
        captions = _captions_from_row(row)
        if not captions:
            continue
        rel_name = f"cuhk_{pid}_{count:06d}.jpg"
        _save_pil(image, images_dir / rel_name, height)
        record = {
            "image": f"images/{rel_name}",
            "person_id": f"cuhk_{pid}",
            "captions": captions,
            "split": split,
            "source": "cuhk_pedes",
        }
        (train_rows if split == "train" else val_rows).append(record)
        count += 1
        if max_images is not None and count >= max_images:
            break
    return train_rows, val_rows


def _package_icfg(images_dir: Path, hf_id: str, height: int, max_images: int | None):
    """Best-effort ICFG packaging. Prefer JSON/CSV members; avoid untrusted pickle."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit('Install huggingface_hub') from exc

    # Many mirrors ship a zip. We only convert image+caption tables, never pickle.load.
    print(
        f"ICFG-PEDES: downloading from {hf_id}. "
        "If the archive only contains a pickle, convert offline with a trusted tool "
        "and pass a local folder of images + captions.jsonl via a future --local flag.",
        flush=True,
    )
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_id, split="train")
    except Exception as exc:
        raise SystemExit(
            f"Could not load {hf_id} as a datasets Hub table ({exc}). "
            "Request official ICFG-PEDES access and place a safe JSONL export under "
            "finetune_data/icfg_pedes/ manually."
        ) from exc

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    person_ids = sorted({str(row.get("id") or row.get("person_id") or i) for i, row in enumerate(ds)})
    val_ids = set(person_ids[::20])
    count = 0
    for i, row in enumerate(ds):
        pid = str(row.get("id") or row.get("person_id") or i)
        split = "val" if pid in val_ids else "train"
        image = row.get("image")
        if image is None:
            continue
        captions = _captions_from_row(row)
        if not captions:
            continue
        rel_name = f"icfg_{pid}_{count:06d}.jpg"
        _save_pil(image, images_dir / rel_name, height)
        record = {
            "image": f"images/{rel_name}",
            "person_id": f"icfg_{pid}",
            "captions": captions,
            "split": split,
            "source": "icfg_pedes",
        }
        (train_rows if split == "train" else val_rows).append(record)
        count += 1
        if max_images is not None and count >= max_images:
            break
    if not train_rows:
        # Keep hub download path referenced so the message stays actionable.
        _ = hf_hub_download
        raise SystemExit("No ICFG rows could be converted")
    return train_rows, val_rows


def _cuhk_pid(row) -> str:
    name = str(row.get("file_path") or row.get("img_path") or row.get("image_id") or "")
    stem = Path(name).name
    digits = "".join(ch for ch in stem[:4] if ch.isdigit())
    if len(digits) == 4:
        return digits
    return str(row.get("id") or row.get("person_id") or stem)


def _captions_from_row(row) -> list[str]:
    for key in ("captions", "caption", "text", "descriptions"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
    return []


def _resize_copy(src: Path, dest: Path, height: int) -> None:
    if dest.is_file():
        return
    with Image.open(src) as image:
        _save_pil(image.convert("RGB"), dest, height)


def _save_pil(image, dest: Path, height: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "convert"):
        rgb = image.convert("RGB")
    else:
        rgb = Image.open(image).convert("RGB")
    width, height0 = rgb.size
    if height0 != height:
        new_w = max(1, int(round(width * (height / height0))))
        rgb = rgb.resize((new_w, height), Image.BICUBIC)
    rgb.save(dest, quality=95)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
