"""CLIP-style InfoNCE with identity-aware positives (IRRA / TBPS-CLIP style)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_infonce(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    person_ids: torch.Tensor,
    *,
    logit_scale: torch.Tensor | float = 100.0,
) -> torch.Tensor:
    """Symmetric InfoNCE where any same-``person_id`` pair is a positive.

    ``image_features`` and ``text_features`` are ``(B, D)`` (not necessarily
    normalised). ``person_ids`` is ``(B,)`` integer or string-mapped ids.
    """

    if image_features.shape != text_features.shape:
        raise ValueError(
            f"image/text feature shapes differ: {tuple(image_features.shape)} vs "
            f"{tuple(text_features.shape)}"
        )
    if person_ids.shape[0] != image_features.shape[0]:
        raise ValueError("person_ids length must match the batch")

    image = F.normalize(image_features.float(), dim=-1)
    text = F.normalize(text_features.float(), dim=-1)
    if isinstance(logit_scale, torch.Tensor):
        scale = logit_scale.float().exp().clamp(max=100.0)
    else:
        scale = torch.tensor(float(logit_scale), device=image.device, dtype=image.dtype)

    logits_i2t = scale * (image @ text.T)
    logits_t2i = logits_i2t.T
    # Positive mask: same identity, including the diagonal.
    ids = person_ids.view(-1, 1)
    positives = ids.eq(ids.T).to(dtype=logits_i2t.dtype)
    # Soft targets: uniform over positives in the batch.
    targets = positives / positives.sum(dim=1, keepdim=True).clamp_min(1.0)
    loss_i2t = _soft_cross_entropy(logits_i2t, targets)
    loss_t2i = _soft_cross_entropy(logits_t2i, targets)
    return 0.5 * (loss_i2t + loss_t2i)


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()
