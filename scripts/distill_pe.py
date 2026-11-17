#!/usr/bin/env python3
"""PE-Core distillation entry point.

``--dry-run`` checks the crop folder and builds a stub student. It does not
download a teacher or train. Omit ``--dry-run`` and this script refuses to
start a run; call ``shawaf_vlm.distill.fit`` from a notebook when you mean to train.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shawaf_vlm.distill.run import dry_run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Distill PE-Core vision into a smaller student.")
    parser.add_argument("--teacher", default="pe_core_l14")
    parser.add_argument("--student", default="pe_core_s16")
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate crops and build a stub student. Does not train.",
    )
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise SystemExit(
            "Refusing to train from this entry point. Pass --dry-run to validate, "
            "or call shawaf_vlm.distill.fit when you are ready to train."
        )
    report = dry_run(args.crops, args.out, args.teacher, args.student)
    print(report)


if __name__ == "__main__":
    main()
