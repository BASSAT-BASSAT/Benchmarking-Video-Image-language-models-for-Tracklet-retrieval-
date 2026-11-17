"""Export a vision module to ONNX. TensorRT itself is not invoked."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def trtexec_command(onnx_path: Path | str, image_size: int) -> str:
    path = Path(onnx_path)
    engine = path.with_suffix(".plan")
    shape = f"3x{image_size}x{image_size}"
    return (
        f"trtexec --onnx={path} --fp16 --saveEngine={engine} "
        f"--minShapes=images:1x{shape} "
        f"--optShapes=images:16x{shape} "
        f"--maxShapes=images:64x{shape}"
    )


def export_module_onnx(module: nn.Module, path: Path | str, image_size: int, opset: int = 17) -> Path:
    """Export ``module(images)`` with a dynamic batch axis. ``images`` is NCHW."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size)
    module.eval()
    try:
        torch.onnx.export(
            module,
            dummy,
            str(destination),
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            module,
            dummy,
            str(destination),
            input_names=["images"],
            output_names=["embeddings"],
            dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
            opset_version=opset,
        )
    return destination
