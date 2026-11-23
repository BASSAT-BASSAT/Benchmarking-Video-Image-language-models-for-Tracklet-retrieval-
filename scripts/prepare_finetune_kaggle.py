#!/usr/bin/env python3
"""Download + package RSTPReid and TVPReid train/val for fine-tuning (Kaggle-friendly).

Needs Internet (gdown + Hugging Face). Never downloads Alexandria/GroOT/test splits.

TVPReid packaging matches eval ``reid_8fps_n64`` + sliding ``window=8/stride=4``.

Example::

    python scripts/prepare_finetune_kaggle.py --out /kaggle/working/finetune_data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "finetune_data")
    parser.add_argument("--skip-rstpreid", action="store_true")
    parser.add_argument("--skip-tvpreid", action="store_true")
    parser.add_argument("--max-images", type=int, default=None, help="Smoke cap per source")
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()

    build = ROOT / "scripts" / "build_finetune_dataset.py"
    common = [
        sys.executable,
        str(build),
        "--out",
        str(args.out),
        "--height",
        str(args.height),
    ]
    if args.max_images is not None:
        common += ["--max-images", str(args.max_images)]

    if not args.skip_rstpreid:
        cmd = common + ["--source", "rstpreid"]
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)

    if not args.skip_tvpreid:
        cmd = common + [
            "--source",
            "tvpreid",
            "--sample-fps",
            str(args.sample_fps),
            "--max-frames",
            str(args.max_frames),
            "--window",
            str(args.window),
            "--stride",
            str(args.stride),
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
        recipe = (
            f"fps{args.sample_fps:g}_n{args.max_frames}_w{args.window}_s{args.stride}\n"
        )
        marker = args.out / "tvpreid" / ".frame_recipe"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(recipe, encoding="utf-8")

    print("Done. Train with:", flush=True)
    print(
        f"  python scripts/train_pe_finetune.py --run A "
        f"--train {args.out}/rstpreid/train.jsonl {args.out}/tvpreid/train.jsonl "
        f"--val {args.out}/rstpreid/val.jsonl {args.out}/tvpreid/val.jsonl "
        f"--image-root {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
