"""One training step and a short AdamW + cosine loop. Does not download models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from shawaf_vlm.distill.losses import feature_cosine_loss


def train_step(
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    teacher_feats: torch.Tensor,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = feature_cosine_loss,
) -> float:
    student.train()
    optimizer.zero_grad(set_to_none=True)
    predicted = student(images)
    loss = loss_fn(predicted, teacher_feats)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def save_distilled_checkpoint(
    path: Path | str,
    student: nn.Module,
    *,
    in_dim: int,
    out_dim: int,
    backbone: str = "linear",
    loss: float | None = None,
) -> None:
    """Write the file ``DistilledPeEncoder`` loads."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "backbone": backbone,
            "state_dict": student.state_dict(),
            "loss": loss,
        },
        destination,
    )


def fit(
    student: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int = 1,
    lr: float = 1e-4,
    out_path: Path | str | None = None,
    in_dim: int | None = None,
    out_dim: int | None = None,
) -> list[float]:
    """Run ``epochs`` passes over ``batches`` of ``(images, teacher_feats)``."""

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    optimizer = AdamW(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[float] = []
    materialised = list(batches)
    for _ in range(epochs):
        for images, teacher_feats in materialised:
            history.append(train_step(student, optimizer, images, teacher_feats))
        scheduler.step()
    if out_path is not None:
        last = history[-1] if history else None
        if in_dim is None or out_dim is None:
            raise ValueError("fit needs in_dim and out_dim to write a distilled checkpoint.")
        save_distilled_checkpoint(
            out_path,
            student,
            in_dim=in_dim,
            out_dim=out_dim,
            loss=last,
        )
    return history
