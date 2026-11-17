"""ONNX export of a tiny module, and ModelOpt wrappers with a fake package."""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from shawaf_vlm.optim.export import export_module_onnx, trtexec_command
from shawaf_vlm.optim.quantize import int8_ptq, qat_kd


class _Tiny(nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))


def test_trtexec_command_is_printed_not_run() -> None:
    command = trtexec_command("vision.onnx", 224)
    assert command.startswith("trtexec ")
    assert "--fp16" in command
    assert "vision.plan" in command
    assert "images:16x3x224x224" in command


def test_export_tiny_module(tmp_path) -> None:
    pytest.importorskip("onnx")
    import onnx

    destination = tmp_path / "tiny.onnx"
    export_module_onnx(_Tiny(), destination, image_size=16)
    model = onnx.load(destination)
    onnx.checker.check_model(model)


def test_quantize_helpers_explain_missing_modelopt(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "modelopt" or name.startswith("modelopt."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "modelopt", None)
    with pytest.raises(ImportError, match="NVIDIA ModelOpt"):
        int8_ptq(_Tiny(), [torch.zeros(1, 3, 4, 4)])


def test_quantize_wrappers_call_modelopt(monkeypatch) -> None:
    calls: dict = {}
    quant = types.ModuleType("modelopt.torch.quantization")
    quant.INT8_DEFAULT_CFG = {"bits": 8}

    def quantize(model, config, loop):
        calls["config"] = config
        loop(model)
        calls["looped"] = True
        return "quantized"

    quant.quantize = quantize
    distill = types.ModuleType("modelopt.torch.distill")

    def convert(student, mode):
        calls["mode"] = mode
        return student

    distill.convert = convert
    monkeypatch.setitem(sys.modules, "modelopt", types.ModuleType("modelopt"))
    monkeypatch.setitem(sys.modules, "modelopt.torch", types.ModuleType("modelopt.torch"))
    monkeypatch.setitem(sys.modules, "modelopt.torch.quantization", quant)
    monkeypatch.setitem(sys.modules, "modelopt.torch.distill", distill)

    model = _Tiny()
    assert int8_ptq(model, [torch.zeros(1, 3, 2, 2)]) == "quantized"
    assert calls["looped"] is True
    assert calls["config"]["bits"] == 8
    teacher = _Tiny()
    converted = qat_kd(model, teacher, [torch.zeros(1, 3, 2, 2)], steps=1)
    assert converted is model
    assert calls["mode"]["teacher_model"] is teacher
