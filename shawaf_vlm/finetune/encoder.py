"""Eval-time wrappers that load a fine-tuned PE open_clip state dict."""

from __future__ import annotations

import os
from pathlib import Path

from shawaf_vlm.finetune.checkpoint import load_finetuned_pe_state
from shawaf_vlm.models.frame_encoders import PeCoreB16Encoder, PerceptionEncoder


def _checkpoint_from_env(env_name: str, explicit: str | Path | None) -> Path:
    if explicit is not None:
        path = Path(explicit)
    else:
        raw = os.environ.get(env_name, "")
        path = Path(raw) if raw else Path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Fine-tuned weights not found ({env_name}={raw!r} / path={explicit!r}). "
            "Train with shawaf_vlm.finetune.train_pe or set the env var to best.pt."
        )
    return path


class FinetunedPeCoreB16Encoder(PeCoreB16Encoder):
    """PE-B/16 with weights from ``SHAWAF_PE_B16_FT`` or ``checkpoint=``."""

    name = "pe_core_b16_ft"

    def __init__(
        self,
        device: str = "cuda",
        image_size: int | None = None,
        fit: str | None = None,
        checkpoint: str | Path | None = None,
        frame_microbatch: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        # Build the base PE-B first, then overwrite weights.
        super().__init__(
            device=device,
            image_size=image_size,
            fit=fit,
            frame_microbatch=frame_microbatch,
            num_workers=num_workers,
        )
        path = _checkpoint_from_env("SHAWAF_PE_B16_FT", checkpoint)
        load_finetuned_pe_state(path, self.model, strict=False)
        self.ft_checkpoint = str(path)


class FinetunedPeCoreL14Encoder(PerceptionEncoder):
    """PE-L/14 with weights from ``SHAWAF_PE_L14_FT`` or ``checkpoint=``."""

    name = "pe_core_l14_ft"

    def __init__(
        self,
        device: str = "cuda",
        image_size: int | None = None,
        fit: str | None = None,
        checkpoint: str | Path | None = None,
        frame_microbatch: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        super().__init__(
            device=device,
            image_size=image_size,
            fit=fit,
            frame_microbatch=frame_microbatch,
            num_workers=num_workers,
        )
        path = _checkpoint_from_env("SHAWAF_PE_L14_FT", checkpoint)
        load_finetuned_pe_state(path, self.model, strict=False)
        self.ft_checkpoint = str(path)
