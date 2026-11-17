#!/usr/bin/env python3
"""Unpack the Alexandria day zips and crop tracklets into datasets/alexandria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shawaf_vlm.data.alexandria import prepare_alexandria, unpack_alexandria


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", type=Path, default=None, help="Folder with dayN-*.zip (default: Annotations/)")
    parser.add_argument("--source", type=Path, default=None, help="Unpack target (default: Annotations/alexandria)")
    parser.add_argument("--videos", type=Path, default=None, help="Raw videos (default: Videos-taken_by_us)")
    parser.add_argument("--dest", type=Path, default=None, help="Prepared root (default: datasets/alexandria)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract and re-crop")
    parser.add_argument("--unpack-only", action="store_true")
    args = parser.parse_args()

    entries = unpack_alexandria(args.zips, args.source, args.videos, overwrite=args.overwrite)
    summary: dict[str, object] = {
        "clips": len(entries),
        "identities": sum(int(e["identities"]) for e in entries),
        "captioned": sum(int(e["captioned"]) for e in entries),
        "instances": sum(int(e["instances"]) for e in entries),
        "clips_without_video": [e["project"] for e in entries if not e["video"]],
        "clips_without_captions": [e["project"] for e in entries if not e["captioned"]],
    }
    if not args.unpack_only:
        files = prepare_alexandria(args.source, args.dest, overwrite=args.overwrite)
        summary["split_tables"] = len(files)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
