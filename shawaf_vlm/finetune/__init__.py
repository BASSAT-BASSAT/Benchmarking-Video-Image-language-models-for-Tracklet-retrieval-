"""Fine-tune Perception Encoder on public person-description data.

Never train on Alexandria, GroOT, RSTPReid test, or TVPReid test.
See ``docs/finetuning_plan.md``.
"""

from shawaf_vlm.finetune.checkpoint import load_finetuned_pe_state, save_finetune_checkpoint
from shawaf_vlm.finetune.lora import apply_lora
from shawaf_vlm.finetune.losses import identity_infonce
from shawaf_vlm.finetune.train import FinetuneConfig, train_pe

__all__ = [
    "FinetuneConfig",
    "apply_lora",
    "identity_infonce",
    "load_finetuned_pe_state",
    "save_finetune_checkpoint",
    "train_pe",
]
