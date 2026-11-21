"""Save / load fine-tuned open_clip PE checkpoints."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 1


def free_disk_gb(path: Path | str) -> float | None:
    """Return free space in GiB for the filesystem containing ``path``, or None."""

    try:
        usage = shutil.disk_usage(Path(path))
        return usage.free / (1024**3)
    except OSError:
        return None


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
    """Write a file that ``FinetunedPerceptionEncoder`` / registry loaders accept.

    Writes atomically via a ``.tmp`` sibling so a failed save does not leave a
    half-written ``.pt``. Optimizer state roughly doubles size — omit it on
    disk-constrained hosts (Kaggle ``/kaggle/working``).
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = free_disk_gb(destination.parent)
    if free is not None and free < 2.0:
        raise OSError(
            f"Only {free:.1f} GiB free under {destination.parent}; "
            "delete old epoch_*.pt / unused datasets before saving again."
        )

    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "base_key": base_key,
        "base_checkpoint": base_checkpoint,
        "epoch": int(epoch),
        "state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra

    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        torch.save(payload, tmp)
        os.replace(tmp, destination)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        free_after = free_disk_gb(destination.parent)
        hint = ""
        if free_after is not None:
            hint = f" Free space now: {free_after:.1f} GiB."
        raise OSError(
            f"Failed to write checkpoint {destination}.{hint} "
            "On Kaggle, remove epoch_*.pt and large download caches, then resume."
        ) from None


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

    import os as _os

    raw = env_or_path or _os.environ.get(default_name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None
