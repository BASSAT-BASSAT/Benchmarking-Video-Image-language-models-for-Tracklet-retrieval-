"""Feature, affinity, and text-contrastive losses for PE-Core distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_cosine_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance. Identical rows score 0. Teacher is detached."""

    student_n = F.normalize(student.float(), dim=-1)
    teacher_n = F.normalize(teacher.detach().float(), dim=-1)
    return (1.0 - (student_n * teacher_n).sum(dim=-1)).mean()


def affinity_mimic_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match in-batch image–image cosine matrices (TinyCLIP affinity mimicking)."""

    student_n = F.normalize(student.float(), dim=-1)
    teacher_n = F.normalize(teacher.detach().float(), dim=-1)
    return F.mse_loss(student_n @ student_n.transpose(0, 1), teacher_n @ teacher_n.transpose(0, 1))


def text_contrastive_loss(
    image: torch.Tensor,
    text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE against frozen teacher text embeddings. Diagonal pairs are positives."""

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    image_n = F.normalize(image.float(), dim=-1)
    text_n = F.normalize(text.detach().float(), dim=-1)
    logits = image_n @ text_n.transpose(0, 1) / temperature
    labels = torch.arange(image_n.shape[0], device=image_n.device)
    return F.cross_entropy(logits, labels)
