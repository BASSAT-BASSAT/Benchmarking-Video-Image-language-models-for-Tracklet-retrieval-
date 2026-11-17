"""Distill a PE-Core vision tower into a smaller student. Training is opt-in.

EXPERIMENTAL, NOT WIRED INTO INGESTION. Known bug: ``save_distilled_checkpoint``
always records ``backbone="linear"``, so ``DistilledPeEncoder`` loads a trained
student as its 1x1 mean-colour stub, and ``_VisionForward`` is not an
``nn.Module`` (backbone weights are not saved). Fix before trusting any number;
the planned Kaggle fine-tuning (docs/finetuning_plan.md) supersedes this path.
"""

from shawaf_vlm.distill.data import CropFolderDataset, refuse_tier_a_test
from shawaf_vlm.distill.losses import affinity_mimic_loss, feature_cosine_loss, text_contrastive_loss
from shawaf_vlm.distill.run import dry_run
from shawaf_vlm.distill.student import StudentWithProjection
from shawaf_vlm.distill.teacher_cache import precompute_teacher_cache, read_teacher_cache, write_teacher_cache
from shawaf_vlm.distill.train import fit, save_distilled_checkpoint, train_step

__all__ = [
    "CropFolderDataset",
    "StudentWithProjection",
    "affinity_mimic_loss",
    "dry_run",
    "feature_cosine_loss",
    "fit",
    "precompute_teacher_cache",
    "read_teacher_cache",
    "save_distilled_checkpoint",
    "refuse_tier_a_test",
    "text_contrastive_loss",
    "train_step",
    "write_teacher_cache",
]
