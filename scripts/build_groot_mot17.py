"""Build the GroOT-MOT17 tracklet mirror and optionally upload it to the Hugging Face Hub.

    python scripts/build_groot_mot17.py --mot17 /path/to/MOT17.zip --out build/GroOT-MOT17
    python scripts/build_groot_mot17.py --mot17 /path/to/MOT17 --out build/GroOT-MOT17 --upload

``--mot17`` accepts MOT17.zip (read in place, no extraction) or an extracted
MOT17 folder. Without ``--mot17`` the script downloads MOT17.zip (~5.5 GB).
The upload token comes from ``HF_TOKEN`` or ``hf auth login`` (write scope).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shawaf_vlm.data.groot import HF_REPO_ID  # noqa: E402
from shawaf_vlm.data.groot_build import (  # noqa: E402
    GROOT_JSON_URL,
    MOT17_ZIP_URL,
    FrameSource,
    collect_tracks,
    write_dataset_card,
    write_metadata,
    write_track_videos,
)


def _download(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}", flush=True)
    from tqdm.auto import tqdm

    with urllib.request.urlopen(url) as response, dest.with_suffix(".part").open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    dest.with_suffix(".part").replace(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("build/GroOT-MOT17"))
    parser.add_argument("--mot17", type=Path, default=None, help="MOT17.zip or extracted MOT17 folder")
    parser.add_argument("--groot-json", type=Path, default=None, help="local mot17_train_coco.json")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "shawaf_vlm" / "groot_build")
    parser.add_argument("--min-height", type=float, default=50.0)
    parser.add_argument("--min-visibility", type=float, default=0.25)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    groot_json = args.groot_json or _download(GROOT_JSON_URL, args.cache / "mot17_train_coco.json")
    mot17 = args.mot17 or _download(MOT17_ZIP_URL, args.cache / "MOT17.zip")

    with Path(groot_json).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    tracks = collect_tracks(
        coco,
        min_height=args.min_height,
        min_visibility=args.min_visibility,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    print(f"Tracks kept: {len(tracks)}", flush=True)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tracks = write_track_videos(tracks, FrameSource(mot17), out, height=args.height, overwrite=args.overwrite)
    write_metadata(tracks, out)
    write_dataset_card(
        tracks,
        out,
        min_height=int(args.min_height),
        min_visibility=args.min_visibility,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        height=args.height,
    )
    size_mb = sum(p.stat().st_size for p in (out / "videos").glob("*.mp4")) / 1e6
    print(f"Wrote {len(tracks)} mp4s ({size_mb:.0f} MB) + metadata to {out}", flush=True)

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo_id,
            repo_type="dataset",
            ignore_patterns=["**/.frames_*/**", ".frames_*/**"],
            commit_message="Add GroOT-MOT17 tracklet mirror",
        )
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
