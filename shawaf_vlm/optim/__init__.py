"""Optional ONNX export and NVIDIA ModelOpt wrappers.

EXPERIMENTAL, NOT WIRED INTO INGESTION: there is no ONNX / TensorRT runtime
encoder yet, and onnx / tensorrt / modelopt are not installed in the crowd-gpu
env. Planned for the deployment phase (FP16 / FP8 / INT8 on the RTX 5070).
"""

from shawaf_vlm.optim.export import export_module_onnx, trtexec_command
from shawaf_vlm.optim.quantize import int8_ptq, qat_kd

__all__ = ["export_module_onnx", "int8_ptq", "qat_kd", "trtexec_command"]
