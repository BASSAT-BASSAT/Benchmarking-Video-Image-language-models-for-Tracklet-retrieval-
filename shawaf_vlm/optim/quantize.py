"""NVIDIA ModelOpt INT8 PTQ and QAT-with-distillation wrappers.

Importing this module does not import ModelOpt. The helpers raise a clear
``ImportError`` until ``pip install -e ".[optim]"`` is available.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def _load_modelopt():
    try:
        import modelopt.torch.distill as mtd
        import modelopt.torch.quantization as mtq
    except ImportError as exc:
        raise ImportError(
            'NVIDIA ModelOpt is required. Install with: pip install -e ".[optim]"'
        ) from exc
    return mtq, mtd


def _batch_images(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def int8_ptq(model: nn.Module, calib_loader: Iterable) -> nn.Module:
    """Post-training INT8 quantization. ``calib_loader`` yields image batches."""

    mtq, _ = _load_modelopt()
    config = getattr(mtq, "INT8_DEFAULT_CFG", {"quant_algo": "int8"})

    def forward_loop(module: nn.Module) -> None:
        for batch in calib_loader:
            module(_batch_images(batch))

    return mtq.quantize(model, config, forward_loop)


def qat_kd(student: nn.Module, teacher: nn.Module, loader: Iterable, steps: int) -> nn.Module:
    """Quantization-aware training with the fp16 teacher, as in the ModelOpt distill guide."""

    if steps < 0:
        raise ValueError("steps must be >= 0")
    _, mtd = _load_modelopt()
    converted = mtd.convert(student, mode={"teacher_model": teacher})
    if steps == 0:
        return converted
    stream = iter(loader)
    for _ in range(steps):
        try:
            batch = next(stream)
        except StopIteration:
            break
        converted(_batch_images(batch))
    return converted
