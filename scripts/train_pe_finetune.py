#!/usr/bin/env python3
"""Fine-tune PE-B (Run A) or PE-L (Run B) on packaged person-description data.

Examples::

    # Run A — PE-B last-6 vision blocks + text (frozen logit_scale)
    python scripts/train_pe_finetune.py --run A \\
        --train finetune_data/rstpreid/train.jsonl \\
        --val finetune_data/rstpreid/val.jsonl \\
        --image-root finetune_data/rstpreid

    # Run B — last 4 vision blocks + text on PE-L
    python scripts/train_pe_finetune.py --run B \\
        --train finetune_data/rstpreid/train.jsonl finetune_data/cuhk_pedes/train.jsonl \\
        --val finetune_data/rstpreid/val.jsonl \\
        --image-root finetune_data

Never train on Alexandria or GroOT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shawaf_vlm.finetune.train import FinetuneConfig, train_pe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=("A", "B"), default="A")
    parser.add_argument("--train", nargs="+", required=True, help="train.jsonl paths")
    parser.add_argument("--val", nargs="*", default=[], help="val.jsonl paths")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke / CI cap")
    parser.add_argument("--use-lora", action="store_true", help="Run B alternative: LoRA instead of last blocks")
    args = parser.parse_args()

    use_lora = True if args.run == "A" else bool(args.use_lora)
    if args.run == "A":
        config = FinetuneConfig(
            base_key="pe_core_b16",
            output_dir=str(args.output or ROOT / "results" / "finetune" / "pe_b16_run_a"),
            epochs=args.epochs or 5,
            batch_size=args.batch_size or 64,
            lr=args.lr or (1e-4 if use_lora else 2e-6),
            unfreeze_last_blocks=None if use_lora else 6,
            use_lora=use_lora,
            lora_on_text=True,
            freeze_logit_scale=True,
            image_size=224,
        )
    else:
        config = FinetuneConfig(
            base_key="pe_core_l14",
            output_dir=str(args.output or ROOT / "results" / "finetune" / "pe_l14_run_b"),
            epochs=args.epochs or 8,
            batch_size=args.batch_size or 16,
            lr=args.lr or (5e-5 if use_lora else 1e-6),
            unfreeze_last_blocks=None if use_lora else 4,
            use_lora=use_lora,
            lora_on_text=True,
            freeze_logit_scale=True,
            grad_checkpoint=True,
            image_size=336,
        )

    config.train_jsonl = [str(Path(p)) for p in args.train]
    config.val_jsonl = [str(Path(p)) for p in args.val]
    config.image_root = str(args.image_root)
    config.device = args.device
    config.resume = args.resume
    config.max_steps = args.max_steps
    config.fit = "letterbox"

    best = train_pe(config)
    env = "SHAWAF_PE_B16_FT" if config.base_key == "pe_core_b16" else "SHAWAF_PE_L14_FT"
    print(f"best checkpoint: {best}")
    print(f"export {env}={best}")


if __name__ == "__main__":
    main()
