"""Save / load fine-tuned open_clip PE checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1


def save_finetune_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    *,
    base_key: str,
    base_checkpoint: str,
    epoch: int,
    metrics: dict[str, float] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a file that ``FinetunedPerceptionEncoder`` / registry loaders accept."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "base_key": base_key,
        "base_checkpoint": base_checkpoint,
        "epoch": int(epoch),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, destination)


def load_finetuned_pe_state(
    path: Path | str,
    model: torch.nn.Module,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Load ``state_dict`` into an already-built PE open_clip model. Returns meta."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} is not a SHAWAF fine-tune checkpoint")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=strict)
    meta = {k: v for k, v in payload.items() if k != "state_dict" and k != "optimizer"}
    meta["missing_keys"] = list(missing)
    meta["unexpected_keys"] = list(unexpected)
    return meta


def resolve_finetune_path(env_or_path: str | None, default_name: str) -> Path | None:
    """Resolve ``SHAWAF_PE_*_FT`` env var or an explicit path."""

    import os

    raw = env_or_path or os.environ.get(default_name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None
