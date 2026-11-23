#!/usr/bin/env python3
"""Pack a Hub video dataset into one tar per ``<config>-<split>`` and upload it.

Thousands of per-file mp4 requests hit Hugging Face's ~1000 requests / 5 min
limit; ``download_hub_video_dataset`` prefers ``archives/<config>-<split>.tar``
when it exists, so a split downloads in one request.

Example::

    python scripts/pack_hub_video_archives.py --root datasets --repo bassatbassat/TVPReid
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.data.hub_video import video_rels_from_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Local copy with metadata/ and video folders")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", type=Path, default=None, help="Where to write tars (default <root>/archives)")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    out = args.out or args.root / "archives"
    out.mkdir(parents=True, exist_ok=True)
    for table in sorted((args.root / "metadata").glob("*.jsonl")):
        rels = sorted(set(video_rels_from_jsonl(table)))
        missing = [rel for rel in rels if not (args.root / rel).is_file()]
        if missing:
            raise SystemExit(f"{table.name}: {len(missing)} videos missing locally, e.g. {missing[0]}")
        tar_path = out / f"{table.stem}.tar"
        if not tar_path.is_file():
            tmp = tar_path.with_suffix(".tar.tmp")
            with tarfile.open(tmp, "w") as tar:
                for rel in rels:
                    tar.add(args.root / rel, arcname=rel)
            tmp.replace(tar_path)
        print(f"{tar_path.name}: {len(rels)} videos, {tar_path.stat().st_size / 2**20:.0f} MB", flush=True)

    if args.no_upload:
        return
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo="archives",
        allow_patterns=["*.tar"],
        commit_message="Add per-split video archives (avoid per-file rate limits)",
    )
    print(f"uploaded {out} → {args.repo}/archives", flush=True)


if __name__ == "__main__":
    main()
