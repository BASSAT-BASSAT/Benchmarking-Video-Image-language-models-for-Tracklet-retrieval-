#!/usr/bin/env python3
"""Action vs scene disentanglement probe for Alexandria.

Uses cached ``query_max`` similarities from ``scripts/rescore.py`` by default
(no GPU). The action-gap on Alexandria is confounded: many actions co-occur
with a room. This script reports:

1. **full_gallery_rank1** — identity Rank-1 on the full gallery.
2. **same_scene_rank1** — identity Rank-1 when the gallery is restricted to the
   query's clip (room known).
3. **top1_action_precision** — whether the full-gallery top-1 shares the query
   action class.
4. **diff_scene_same_vs_other_action_margin** — mean similarity to same-action
   people in *other* clips minus mean similarity to other-action people in
   other clips. Margin > 0 is evidence of action signal beyond scene.

Writes ``results/action_scene/<tag>.json`` and appends ``summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "action_scene"


def clip_of(track_id: str) -> str:
    return track_id.split("/", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sims", type=Path, default=None, help="rescore sims npz (default: pe_core_l14 letterbox n16)")
    parser.add_argument("--actions", type=Path, default=ROOT / "datasets" / "alexandria" / "actions.csv")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    sims_path = args.sims or (ROOT / "results" / "rescore" / "sims" / "pe_core_l14_letterbox_n16.npz")
    if not sims_path.is_file():
        raise SystemExit(f"Missing sims cache {sims_path}. Run scripts/rescore.py once, or pass --sims.")

    data = np.load(sims_path, allow_pickle=True)
    tracks = [str(t) for t in data["q_tracks"]]
    g_tracks = [str(t) for t in data["g_tracks"]]
    if tracks != g_tracks:
        raise SystemExit("Expected square Alexandria sims (query tracks == gallery tracks)")
    sim = np.asarray(data["query_max"], dtype=np.float32)

    with args.actions.open(encoding="utf-8", newline="") as handle:
        caps = {r["track_id"]: r for r in csv.DictReader(handle)}
    actions = [caps[t]["action_class"] for t in tracks]
    clips = [clip_of(t) for t in tracks]
    n = len(tracks)

    full_r1 = []
    same_scene = []
    action_p1 = []
    margins = []
    per_query = []
    by_action: dict[str, list] = defaultdict(list)

    for i in range(n):
        order = np.argsort(-sim[i], kind="stable")
        top = int(order[0])
        full_r1.append(1.0 if top == i else 0.0)
        action_p1.append(1.0 if actions[top] == actions[i] else 0.0)

        same_idx = np.array([j for j in range(n) if clips[j] == clips[i]], dtype=np.int64)
        same_order = same_idx[np.argsort(-sim[i, same_idx], kind="stable")]
        same_rank = int(np.where(same_order == i)[0][0]) + 1
        same_scene.append(1.0 if same_rank == 1 else 0.0)

        diff_same = np.array(
            [j for j in range(n) if actions[j] == actions[i] and clips[j] != clips[i]],
            dtype=np.int64,
        )
        other_act = np.array(
            [j for j in range(n) if actions[j] != actions[i] and clips[j] != clips[i]],
            dtype=np.int64,
        )
        if len(diff_same) and len(other_act):
            margin = float(sim[i, diff_same].mean() - sim[i, other_act].mean())
            margins.append(margin)
        else:
            margin = None

        row = {
            "track_id": tracks[i],
            "action": actions[i],
            "clip": clips[i],
            "full_rank1": int(full_r1[-1]),
            "same_scene_rank1": int(same_scene[-1]),
            "top1_action_match": int(action_p1[-1]),
            "diff_scene_action_margin": margin,
        }
        per_query.append(row)
        by_action[actions[i]].append(row)

    report = {
        "sims": str(sims_path),
        "n": n,
        "full_gallery_rank1": float(np.mean(full_r1)),
        "same_scene_rank1": float(np.mean(same_scene)),
        "top1_action_precision": float(np.mean(action_p1)),
        "diff_scene_same_vs_other_action_margin": float(np.mean(margins)) if margins else None,
        "by_action": {
            act: {
                "n": len(rows),
                "full_rank1": float(np.mean([r["full_rank1"] for r in rows])),
                "top1_action_precision": float(np.mean([r["top1_action_match"] for r in rows])),
                "same_scene_rank1": float(np.mean([r["same_scene_rank1"] for r in rows])),
            }
            for act, rows in sorted(by_action.items())
        },
        "interpretation": {
            "scene_reliance": (
                "If same_scene_rank1 >> full_gallery_rank1, identity is easy once the "
                "room is known; action words may be proxying scene."
            ),
            "action_signal": (
                "diff_scene_same_vs_other_action_margin > 0 means embeddings score "
                "same-action people in other rooms higher than other-action people."
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or sims_path.stem
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / f"{tag}_per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)

    summary_path = OUT / "summary.csv"
    summary_row = {
        "tag": tag,
        "full_r1": f"{report['full_gallery_rank1']:.4f}",
        "same_scene_r1": f"{report['same_scene_rank1']:.4f}",
        "action_p1": f"{report['top1_action_precision']:.4f}",
        "diff_scene_margin": (
            f"{report['diff_scene_same_vs_other_action_margin']:.4f}"
            if report["diff_scene_same_vs_other_action_margin"] is not None
            else ""
        ),
    }
    write_header = not summary_path.is_file()
    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
