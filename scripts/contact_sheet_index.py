#!/usr/bin/env python3
"""Build index.html next to failure-analysis contact sheets so they open as one page."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?",
                        default=ROOT / "results" / "failure_analysis" / "pe_core_l14_letterbox_n64_alexandria_all")
    args = parser.parse_args()
    sheets = sorted(args.folder.glob("contact_sheet_*.png"))
    summary = args.folder / "summary.json"
    head = ""
    if summary.is_file():
        s = json.loads(summary.read_text(encoding="utf-8"))
        head = (f"<p>{s['queries']} queries · Rank-1 {s['rank1']} · {s['misses']} misses · "
                f"buckets {html.escape(json.dumps(s['miss_buckets']))}</p>")
    imgs = "".join(
        f'<figure><img src="{p.name}" alt="{p.stem}" loading="lazy"><figcaption>{p.stem}</figcaption></figure>'
        for p in sheets
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Failure analysis sheets</title>
<style>body {{ font: 15px/1.5 system-ui, sans-serif; margin: 24px auto; max-width: 1200px; padding: 0 16px; }}
img {{ width: 100%; border: 1px solid #ccc; }} figure {{ margin: 0 0 24px; }} figcaption {{ color: #555; font-size: 13px; }}</style>
<h1>Failure analysis: {html.escape(args.folder.name)}</h1>{head}
<p>Each row: the query, then 3 crops of the true person (left) and 3 of what ranked first (right).</p>{imgs}</html>"""
    out = args.folder / "index.html"
    out.write_text(page, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
