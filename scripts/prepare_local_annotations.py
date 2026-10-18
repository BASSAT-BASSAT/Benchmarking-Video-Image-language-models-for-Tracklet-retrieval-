#!/usr/bin/env python3
"""Crop tracklets from Annotations/ into datasets/shawaf_annotations (TVPReid-style metadata)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shawaf_vlm.data.local_annotations import (
    default_prepared_root,
    discover_annotations_source,
    list_annotation_projects,
    prepare_all_projects,
    prepare_project,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local VRush annotation exports for indexing.")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Annotations root (default: repo Annotations/ or SHAWAF_ANNOTATIONS_ROOT)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output root (default: datasets/shawaf_annotations)",
    )
    parser.add_argument("--project", type=str, default=None, help="Only prepare one project folder name")
    parser.add_argument("--overwrite", action="store_true", help="Re-crop even if JPEGs exist")
    args = parser.parse_args()

    source = args.source or discover_annotations_source()
    if source is None:
        raise SystemExit("Annotations source not found.")
    dest = args.dest or default_prepared_root()

    if args.project:
        matches = [p for p in list_annotation_projects(source) if p.name == args.project]
        if not matches:
            raise SystemExit(f"No project named {args.project!r} under {source}")
        path = prepare_project(matches[0], dest, overwrite=args.overwrite)
        if path is None:
            raise SystemExit("Nothing prepared for that project.")
        print(json.dumps({"prepared": str(path), "dest": str(dest)}, indent=2))
        return

    files = prepare_all_projects(source, dest, overwrite=args.overwrite)
    summary = {
        "source": str(source),
        "dest": str(dest),
        "projects": len(list_annotation_projects(source)),
        "jsonl_files": [str(p) for p in files],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
