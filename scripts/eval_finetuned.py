#!/usr/bin/env python3
"""Evaluate a fine-tuned PE checkpoint on Alexandria + GroOT (+ optional diagnostics).

Runs the same paths as the paper gate in ``docs/finetuning_plan.md``:

1. ``ingestion_sweep``-style Rank-1 / mAP (hard) via ``scripts/rescore.py`` when
   Alexandria labels exist, else ``scripts/ingestion_sweep.py``.
2. ``scripts/action_gap.py`` (appearance vs action).
3. ``scripts/failure_analysis.py`` (miss buckets).

Example::

    set SHAWAF_PE_B16_FT=results/finetune/pe_b16_run_a/best.pt
    python scripts/eval_finetuned.py --model pe_core_b16_ft --fit letterbox --max-frames 16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="pe_core_b16_ft", choices=("pe_core_b16_ft", "pe_core_l14_ft", "pe_core_b16", "pe_core_l14"))
    parser.add_argument("--fit", default="letterbox")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-action-gap", action="store_true")
    parser.add_argument("--skip-failure", action="store_true")
    parser.add_argument("--skip-groot", action="store_true")
    parser.add_argument(
        "--check-load",
        action="store_true",
        help="Only build the fine-tuned encoder (verify SHAWAF_PE_*_FT) and exit",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "finetune" / "eval")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.check_load:
        from shawaf_vlm.models.registry import build_encoder

        encoder = build_encoder(args.model, device=args.device, fit=args.fit)
        path = getattr(encoder, "ft_checkpoint", None)
        print(json.dumps({"ok": True, "model": args.model, "ft_checkpoint": path}))
        return

    py = sys.executable
    summary: dict = {"model": args.model, "fit": args.fit, "max_frames": args.max_frames, "steps": []}

    # 1) Hard + meaning-based rescore on Alexandria (if prepared).
    actions = ROOT / "datasets" / "alexandria" / "actions.csv"
    if actions.is_file():
        cmd = [
            py, str(ROOT / "scripts" / "rescore.py"),
            "--model", args.model, "--fit", args.fit,
            "--max-frames", str(args.max_frames), "--device", args.device,
        ]
        summary["steps"].append({"name": "rescore_alexandria", "cmd": cmd, "returncode": _run(cmd)})
    else:
        cmd = [
            py, str(ROOT / "scripts" / "ingestion_sweep.py"),
            f"{args.model}:{args.fit}/n{args.max_frames}",
            "--dataset", "alexandria:all",
            "--out", str(args.out / "alexandria_sweep.csv"),
        ]
        summary["steps"].append({"name": "ingestion_alexandria", "cmd": cmd, "returncode": _run(cmd)})

    if not args.skip_groot:
        cmd = [
            py, str(ROOT / "scripts" / "ingestion_sweep.py"),
            f"{args.model}:{args.fit}/n{args.max_frames}",
            "--dataset", "groot:combined",
            "--out", str(args.out / "groot_sweep.csv"),
        ]
        summary["steps"].append({"name": "ingestion_groot", "cmd": cmd, "returncode": _run(cmd)})

    if not args.skip_action_gap and actions.is_file():
        cmd = [
            py, str(ROOT / "scripts" / "action_gap.py"),
            "--model", args.model, "--fit", args.fit,
            "--max-frames", str(args.max_frames),
        ]
        summary["steps"].append({"name": "action_gap", "cmd": cmd, "returncode": _run(cmd)})

    if not args.skip_failure and actions.is_file():
        fail_out = args.out / f"failure_{args.model}"
        cmd = [
            py, str(ROOT / "scripts" / "failure_analysis.py"),
            "--model", args.model, "--fit", args.fit,
            "--max-frames", str(args.max_frames),
            "--out", str(fail_out),
        ]
        summary["steps"].append({"name": "failure_analysis", "cmd": cmd, "returncode": _run(cmd)})

    (args.out / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    failed = [s for s in summary["steps"] if s["returncode"] not in (0, None)]
    if failed:
        raise SystemExit(f"{len(failed)} eval step(s) failed — see {args.out / 'eval_summary.json'}")


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    main()
