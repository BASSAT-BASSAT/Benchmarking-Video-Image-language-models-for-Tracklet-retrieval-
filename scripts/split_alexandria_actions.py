#!/usr/bin/env python3
"""Write datasets/alexandria/actions.csv and a local review page.

Each row: the original caption, a fixed caption (typos and odd phrasing
corrected), its appearance part, its action part, an action class, a flag for
rows a human should check, and paths to the person's crops and preview video.

Manual review: edit ``caption_fixed`` / ``appearance`` / ``action`` /
``action_class`` in the CSV (Excel or VS Code) and set ``reviewed`` to ``yes``.
Rows you edited are kept on later runs; ``--overwrite`` discards them.

``datasets/alexandria/review.html`` shows every row with three crops of that
person. It stays on this machine (``datasets/`` is gitignored): it shows faces.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.action_split import _clean, classify_action, split_caption  # noqa: E402
from shawaf_vlm.retrieval.datasets import load_splits  # noqa: E402

FIELDS = (
    "track_id", "caption", "caption_fixed", "appearance", "action", "action_class",
    "flag", "reviewed", "note", "crops_folder", "preview",
)

# First-pass fixes for captions the rules cannot read (typos and odd grammar).
# Keyed by track id; values override the automatic split. The person's original
# caption is never modified.
FIXES: dict[str, dict[str, str]] = {
    "day4_35/person_4": {
        "caption_fixed": "the man in pink chemise is looking at the doctor's phone and talking with her",
        "appearance": "the man in pink chemise",
        "action": "looking at the doctor's phone and talking with her",
        "note": "'is pink chemise' read as 'in pink chemise'",
    },
    # day4_34: reviewed by the annotator from the footage. Nobody visibly
    # "rescues" anyone, and "fainting" cannot be seen, only falling.
    "day4_34/person_1": {
        "caption_fixed": "the girl in red is falling",
        "appearance": "the girl in red",
        "action": "falling",
        "reviewed": "yes",
        "note": "annotator: she is falling; fainting is not visible",
    },
    "day4_34/person_2": {
        "caption_fixed": "a girl in hijab, yellow blouse is standing",
        "appearance": "a girl in hijab, yellow blouse",
        "action": "standing",
        "reviewed": "yes",
        "note": "annotator: just standing; 'rescued' removed",
    },
    "day4_34/person_3": {
        "caption_fixed": "a girl in hijab, brown blouse is on her phone",
        "appearance": "a girl in hijab, brown blouse",
        "action": "on her phone",
        "reviewed": "yes",
        "note": "annotator: on her phone; 'rescued' removed",
    },
    "day4_34/person_4": {
        "caption_fixed": "a girl in hijab, blue blouse is standing",
        "appearance": "a girl in hijab, blue blouse",
        "action": "standing",
        "reviewed": "yes",
        "note": "annotator: just standing; 'rescued' removed",
    },
    "day5_52/person_2": {
        "caption_fixed": "a boy in red shirt and black jeans is talking",
        "action": "talking",
        "reviewed": "yes",
        "note": "annotator: 'taking' is a misspelling of 'talking'",
    },
    "day5_52/person_3": {
        "caption_fixed": "a boy in grey shirt and white jeans is talking",
        "action": "talking",
        "reviewed": "yes",
        "note": "annotator: 'taking' is a misspelling of 'talking'",
    },
    "day5_51/person_3": {
        "caption_fixed": "a girl with beige hijab and white blouse is talking to her friend",
        "action": "talking to her friend",
        "note": "'is on talking' read as 'is talking'",
    },
    "day5_43/person_4": {
        "caption_fixed": "a boy with blue jacket and grey sweatpants is on his phone",
        "action": "on his phone",
        "note": "'her phone' for a boy read as 'his phone'",
    },
    "day2_18/person_2": {
        "caption_fixed": "a curly hair girl wears white shirt and blue pants is on her phone",
        "appearance": "a curly hair girl wears white shirt and blue pants",
        "action": "on her phone",
        "reviewed": "yes",
        "note": "annotator: second item is blue pants, not a blue shirt",
    },
}

# Annotator decision: captions with no action describe someone standing.
STANDING = (
    "day2_16/person_2", "day2_17/person_2", "day4_36/person_1",
    "day4_40/person_1", "day5_43/person_5", "day5_43/person_6",
)
for _track in STANDING:
    FIXES[_track] = {"action": "standing", "reviewed": "yes", "note": "annotator: no action in caption, so standing"}

NO_ACTION_NOTE = "no action in caption: watch the clip and add one if visible"


def build_row(track_id: str, caption: str, crops_folder: Path, preview: Path | None) -> dict[str, str]:
    s = split_caption(caption)
    row = {
        "track_id": track_id,
        "caption": caption,
        "caption_fixed": _clean(caption),
        "appearance": s.appearance,
        "action": s.action,
        "action_class": s.action_class,
        "flag": s.flag,
        "reviewed": "",
        "note": "",
        "crops_folder": str(crops_folder),
        "preview": str(preview) if preview else "",
    }
    fix = FIXES.get(track_id)
    if fix:
        row.update(fix)
        if "caption_fixed" not in fix and fix.get("action") and not s.action:
            # Action added for a caption that had none: append it to the fixed caption.
            row["caption_fixed"] = f"{row['caption_fixed']} is {fix['action']}"
        row["action_class"] = classify_action(row["action"])
        row["flag"] = "" if fix.get("reviewed") == "yes" else "auto-fixed: please confirm"
    if not row["action"]:
        row["flag"] = "no action found"
        row["note"] = row["note"] or NO_ACTION_NOTE
    return row


def write_review_page(rows: list[dict[str, str]], gallery: dict[str, list[Path]], out: Path) -> None:
    flagged_first = sorted(rows, key=lambda r: (r["flag"] == "", r["track_id"]))
    items = []
    for r in flagged_first:
        crops = gallery.get(r["track_id"], [])
        picks = [crops[int(i)] for i in (0, len(crops) // 2, len(crops) - 1)] if crops else []
        imgs = "".join(
            f'<img src="{html.escape(p.resolve().as_uri())}" alt="" loading="lazy">' for p in picks
        )
        flag = f'<span class="flag">{html.escape(r["flag"])}</span>' if r["flag"] else ""
        note = f'<p class="note">{html.escape(r["note"])}</p>' if r["note"] else ""
        # Browsers refuse to open local folders from a page, so show every crop inline.
        all_imgs = "".join(
            f'<img src="{html.escape(p.resolve().as_uri())}" alt="" loading="lazy">' for p in crops
        )
        everything = (
            f'<details><summary>Show all {len(crops)} crops</summary><div class="all">{all_imgs}</div></details>'
            if crops else ""
        )
        items.append(
            f'<article><div class="crops">{imgs}</div><div class="text">'
            f'<h2>{html.escape(r["track_id"])} {flag}</h2>'
            f'<p><b>Original</b> {html.escape(r["caption"])}</p>'
            f'<p><b>Fixed</b> {html.escape(r["caption_fixed"])}</p>'
            f'<p><b>Appearance</b> {html.escape(r["appearance"])}</p>'
            f'<p><b>Action</b> {html.escape(r["action"]) or "<i>none</i>"} '
            f'<span class="cls">{html.escape(r["action_class"])}</span></p>{note}'
            f"{everything}"
            f"</div></article>"
        )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Alexandria caption review</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; margin: 24px auto; max-width: 1100px; padding: 0 16px; background: #f6f7f8; color: #12181f; }}
article {{ display: grid; grid-template-columns: 300px 1fr; gap: 16px; background: #fff; border: 1px solid #d9dee3; border-radius: 10px; padding: 12px; margin: 12px 0; }}
.crops {{ display: flex; gap: 4px; }} .crops img {{ width: 96px; height: 200px; object-fit: contain; background: #111; border-radius: 4px; }}
h2 {{ font-size: 16px; margin: 0 0 6px; }} p {{ margin: 3px 0; }} b {{ display: inline-block; width: 92px; color: #47515d; font-weight: 600; }}
.flag {{ font-size: 12px; background: #fff4d6; color: #7a5200; padding: 2px 8px; border-radius: 99px; margin-left: 6px; }}
.cls {{ font-size: 12px; background: #e6f4f1; color: #0f7a6c; padding: 2px 8px; border-radius: 99px; }}
.note {{ color: #7a5200; }}
details {{ margin-top: 8px; font-size: 13px; }} summary {{ cursor: pointer; color: #0f7a6c; }}
.all {{ display: flex; flex-wrap: wrap; gap: 3px; margin-top: 6px; }} .all img {{ width: 64px; height: 132px; object-fit: contain; background: #111; border-radius: 3px; }}
@media (max-width: 640px) {{ article {{ grid-template-columns: 1fr; }} }}
</style>
<h1>Alexandria caption review</h1>
<p>{len(rows)} people. Flagged rows first. Edit <code>datasets/alexandria/actions.csv</code> and set <code>reviewed</code> to <code>yes</code> on rows you checked; then re-run <code>python scripts/split_alexandria_actions.py</code> to refresh this page.</p>
{''.join(items)}
</html>"""
    out.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="alexandria:all")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "alexandria" / "actions.csv")
    parser.add_argument("--overwrite", action="store_true", help="Discard manual edits and rebuild every row")
    args = parser.parse_args()

    existing: dict[str, dict[str, str]] = {}
    if args.out.is_file() and not args.overwrite:
        with args.out.open(encoding="utf-8", newline="") as handle:
            existing = {row["track_id"]: row for row in csv.DictReader(handle) if row.get("reviewed") == "yes"}

    splits = load_splits(args.dataset, download=False)
    gallery = {g.track_id: list(g.crop_paths) for g in splits.gallery}
    rows = []
    for q in splits.query:
        if q.track_id in existing:
            rows.append({k: existing[q.track_id].get(k, "") for k in FIELDS})
            continue
        crops = gallery.get(q.track_id, [])
        folder = crops[0].parent if crops else Path()
        preview = folder / "preview.mp4" if crops and (folder / "preview.mp4").is_file() else None
        rows.append(build_row(q.track_id, q.text, folder, preview))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    page = args.out.with_name("review.html")
    write_review_page(rows, gallery, page)
    classes = Counter(r["action_class"] for r in rows)
    flagged = [r for r in rows if r["flag"]]
    print(f"wrote {args.out} ({len(rows)} rows, {len(flagged)} flagged, {len(existing)} kept from manual review)")
    print(f"review page: {page}")
    print("action classes:", dict(classes.most_common()))


if __name__ == "__main__":
    main()
