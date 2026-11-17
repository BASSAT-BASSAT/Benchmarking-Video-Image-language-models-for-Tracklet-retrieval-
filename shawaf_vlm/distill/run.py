"""Validate a distillation job and build a stub student. Does not train."""

from __future__ import annotations

import json
from pathlib import Path

from torch import nn

from shawaf_vlm.distill.data import CropFolderDataset
from shawaf_vlm.distill.student import StudentWithProjection


def dry_run(crops: Path | str, out: Path | str, teacher: str, student_name: str) -> dict:
    dataset = CropFolderDataset(crops)
    if len(dataset) < 1:
        raise FileNotFoundError(f"No crop images under {crops}.")
    student = StudentWithProjection(nn.Linear(8, 8), 8, 16)
    report = {
        "teacher": teacher,
        "student": student_name,
        "crops": len(dataset),
        "projection_in": int(student.projection.in_features),
        "projection_out": int(student.projection.out_features),
        "trained": False,
    }
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "dry_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
