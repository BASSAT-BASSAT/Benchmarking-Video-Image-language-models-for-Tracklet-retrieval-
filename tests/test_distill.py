"""Distillation losses, one CPU step, cache, crop guard, and --dry-run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from shawaf_vlm.distill.data import CropFolderDataset
from shawaf_vlm.distill.losses import affinity_mimic_loss, feature_cosine_loss, text_contrastive_loss
from shawaf_vlm.distill.student import StudentWithProjection
from shawaf_vlm.distill.teacher_cache import precompute_teacher_cache, read_teacher_cache, write_teacher_cache
from shawaf_vlm.distill.train import fit, train_step
from shawaf_vlm.models.frame_encoders import _FrameEncoder
from shawaf_vlm.models.frame_encoders import DistilledPeEncoder
from shawaf_vlm.models.registry import build_encoder


def test_identical_features_have_zero_loss_and_grads_flow() -> None:
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    assert feature_cosine_loss(rows, rows.detach()).item() == pytest.approx(0.0, abs=1e-6)
    assert affinity_mimic_loss(rows, rows.detach()).item() == pytest.approx(0.0, abs=1e-6)
    student = torch.randn(4, 8, requires_grad=True)
    teacher = torch.randn(4, 8)
    text = torch.randn(4, 8)
    loss = feature_cosine_loss(student, teacher) + affinity_mimic_loss(student, teacher)
    loss = loss + text_contrastive_loss(student, text)
    loss.backward()
    assert student.grad is not None
    assert float(student.grad.abs().sum()) > 0


def test_projection_output_shape() -> None:
    student = StudentWithProjection(nn.Linear(5, 5), 5, 9)
    output = student(torch.randn(3, 5))
    assert tuple(output.shape) == (3, 9)


def test_one_train_step_reduces_mse() -> None:
    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(1, 1)
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        def forward(self, inputs):
            return self.fc(inputs)

    student = _Tiny()
    images = torch.ones(4, 1)
    teacher = torch.ones(4, 1)
    before = float(F.mse_loss(student(images), teacher).detach())
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.1, weight_decay=0.0)
    train_step(student, optimizer, images, teacher, loss_fn=F.mse_loss)
    after = float(F.mse_loss(student(images), teacher).detach())
    assert after < before


class _OnesEncoder(_FrameEncoder):
    def __init__(self) -> None:
        super().__init__(device="cpu", frame_microbatch=8, num_workers=0)

    def _encode_images(self, images):
        return torch.ones(len(images), 3)


def test_precompute_teacher_cache(tmp_path) -> None:
    crops = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in crops:
        Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path)
    precompute_teacher_cache(_OnesEncoder(), crops, tmp_path / "cache")
    loaded_paths, embeddings = read_teacher_cache(tmp_path / "cache")
    assert loaded_paths == crops
    assert embeddings.shape == (2, 3)


def test_fit_writes_a_loadable_checkpoint(tmp_path) -> None:
    student = StudentWithProjection(nn.Linear(3, 3), 3, 3)
    images = torch.ones(2, 3)
    teacher = torch.ones(2, 3)
    destination = tmp_path / "student.pt"
    fit(
        student,
        [(images, teacher)],
        epochs=1,
        lr=1e-3,
        out_path=destination,
        in_dim=3,
        out_dim=3,
    )
    encoder = DistilledPeEncoder(device="cpu", checkpoint=str(destination))
    assert encoder.in_dim == 3
    assert encoder.out_dim == 3


def test_teacher_cache_round_trip(tmp_path) -> None:
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    embeddings = np.arange(8, dtype=np.float32).reshape(2, 4)
    write_teacher_cache(tmp_path / "cache", paths, embeddings)
    loaded_paths, loaded = read_teacher_cache(tmp_path / "cache")
    assert loaded_paths == paths
    np.testing.assert_allclose(loaded, embeddings)


def test_crop_folder_refuses_tier_a_test_split(tmp_path) -> None:
    root = tmp_path / "tvpreid" / "test"
    root.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(root / "a.jpg")
    with pytest.raises(ValueError, match="Tier-A"):
        CropFolderDataset(root)


def test_distilled_checkpoint_loads_and_registry_requires_a_file(tmp_path, monkeypatch) -> None:
    torch.manual_seed(0)
    student = StudentWithProjection(nn.Linear(4, 4), 4, 6)
    blob = {
        "in_dim": 4,
        "out_dim": 6,
        "backbone": "linear",
        "state_dict": student.state_dict(),
    }
    path = tmp_path / "student.pt"
    torch.save(blob, path)
    encoder = DistilledPeEncoder(device="cpu", checkpoint=str(path))
    assert encoder.name == "pe_core_distilled"
    assert encoder.out_dim == 6
    image = Image.new("RGB", (6, 6), color=(3, 4, 5))
    image.save(tmp_path / "crop.jpg")
    vectors = encoder.encode_videos([[tmp_path / "crop.jpg"]])
    assert vectors.shape == (1, 6)
    monkeypatch.delenv("SHAWAF_PE_DISTILLED", raising=False)
    with pytest.raises(FileNotFoundError, match="SHAWAF_PE_DISTILLED"):
        build_encoder("pe_core_distilled", device="cpu")


def test_distill_script_dry_run(tmp_path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(crops / "person.jpg")
    script = Path(__file__).resolve().parents[1] / "scripts" / "distill_pe.py"
    spec = importlib.util.spec_from_file_location("distill_pe_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    out = tmp_path / "run"
    module.main(
        [
            "--teacher",
            "pe_core_l14",
            "--student",
            "pe_core_s16",
            "--crops",
            str(crops),
            "--out",
            str(out),
            "--dry-run",
        ]
    )
    report = (out / "dry_run.json").read_text(encoding="utf-8")
    assert '"trained": false' in report
    assert "pe_core_s16" in report
