from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from shawaf_vlm.models.runtime import (
    frames_to_uint8_tchw,
    l2_normalize_torch,
    place_model,
    resolve_device,
    to_numpy,
    unwrap_features,
)

CLIP_S = "OpenGVLab/InternVideo2_CLIP_S"
CLIP_1B = "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"

_ONE_B_HELP = (
    f"{CLIP_1B} is a gated LoRA add-on, not a full Hugging Face AutoModel. "
    f"Use --model internvideo2 ({CLIP_S}) for the Stage 1 InternVideo2 baseline, "
    "or load the official InternVideo2 CLIP-1B stack from OpenGVLab/InternVideo "
    "if you specifically need the 1B checkpoint."
)


def _stub_module(name: str, is_package: bool = False):
    import importlib.machinery
    import types

    module = types.ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    module.__spec__ = spec
    module.__package__ = name if is_package else name.rpartition(".")[0]
    if is_package:
        module.__path__ = []
    return module


def install_flash_attn_stub() -> None:
    """Satisfy InternVideo2 remote-code imports without compiling flash-attn.

    CLIP-S already sets use_flash_attn/use_fused_mlp/use_fused_rmsnorm to False
    and uses naive attention. transformers still scans `from flash_attn...`
    and refuses to load the Hub files unless the package imports. The stub must
    expose a ModuleSpec: a bare types.ModuleType makes find_spec raise
    ValueError: flash_attn.__spec__ is None.
    """

    import sys

    existing = sys.modules.get("flash_attn")
    if existing is not None and getattr(existing, "__file__", None):
        return

    from torch import nn

    flash_attn = _stub_module("flash_attn", is_package=True)
    flash_attn.__version__ = "0.0.0"
    modules = _stub_module("flash_attn.modules", is_package=True)
    mlp = _stub_module("flash_attn.modules.mlp")
    ops = _stub_module("flash_attn.ops", is_package=True)
    rms = _stub_module("flash_attn.ops.rms_norm")
    interface = _stub_module("flash_attn.flash_attn_interface")
    padding = _stub_module("flash_attn.bert_padding")

    class FusedMLP(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            raise RuntimeError(
                "flash_attn FusedMLP stub used; set use_fused_mlp=False"
            )

    class DropoutAddRMSNorm(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            raise RuntimeError(
                "flash_attn DropoutAddRMSNorm stub used; set use_fused_rmsnorm=False"
            )

    def _missing(*args, **kwargs):
        raise RuntimeError(
            "flash_attn is stubbed; InternVideo2 should use naive attention"
        )

    mlp.FusedMLP = FusedMLP
    rms.DropoutAddRMSNorm = DropoutAddRMSNorm
    interface.flash_attn_varlen_qkvpacked_func = _missing
    padding.unpad_input = _missing
    padding.pad_input = _missing
    flash_attn.modules = modules
    flash_attn.ops = ops
    modules.mlp = mlp
    ops.rms_norm = rms

    sys.modules.update(
        {
            "flash_attn": flash_attn,
            "flash_attn.modules": modules,
            "flash_attn.modules.mlp": mlp,
            "flash_attn.ops": ops,
            "flash_attn.ops.rms_norm": rms,
            "flash_attn.flash_attn_interface": interface,
            "flash_attn.bert_padding": padding,
        }
    )


def _purge_broken_flash_attn() -> None:
    """Remove spec-less flash_attn stubs left in a reused Kaggle kernel."""

    import sys

    existing = sys.modules.get("flash_attn")
    if existing is None or getattr(existing, "__spec__", None) is not None:
        return
    for name in list(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)


_purge_broken_flash_attn()


def _resolve_internvideo2_class(checkpoint: str, config):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = None
    if isinstance(auto_map, dict):
        class_ref = auto_map.get("AutoModel")
    else:
        class_ref = getattr(auto_map, "AutoModel", None)
    if not class_ref:
        class_ref = "modeling_internvideo2encoder.InternVideo2_CLIP_small"
    return get_class_from_dynamic_module(class_ref, checkpoint)


def _load_internvideo2_state_dict(checkpoint: str) -> dict:
    from huggingface_hub import hf_hub_download
    import torch

    try:
        from safetensors.torch import load_file

        path = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
        return load_file(path, device="cpu")
    except Exception:
        path = hf_hub_download(repo_id=checkpoint, filename="pytorch_model.bin")
        return torch.load(path, map_location="cpu", weights_only=True)


def _build_internvideo2_model(checkpoint: str, config, dtype):
    """Build InternVideo2 without transformers 5 from_pretrained finalize hooks.

    CLIP-S never calls post_init(), so from_pretrained dies on
    all_tied_weights_keys after the weights are already materialized.
    """

    import torch

    model_cls = _resolve_internvideo2_class(checkpoint, config)
    with torch.device("cpu"):
        model = model_cls(config)
    state = _load_internvideo2_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(incompatible.unexpected_keys)
    print(
        f"InternVideo2 loaded {loaded}/{len(state)} tensors "
        f"(missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)})",
        flush=True,
    )
    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


def _force_naive_attention(config) -> None:
    model_cfg = getattr(config, "model", None)
    if model_cfg is None:
        return
    if isinstance(model_cfg, dict):
        vision = model_cfg.get("vision_encoder")
    else:
        vision = getattr(model_cfg, "vision_encoder", None)
    if vision is None:
        return
    for key in ("use_flash_attn", "use_fused_mlp", "use_fused_rmsnorm"):
        try:
            vision[key] = False
        except Exception:
            pass
        if hasattr(vision, key):
            try:
                setattr(vision, key, False)
            except Exception:
                pass


class InternVideo2Encoder:
    """Frozen InternVideo2 CLIP encoder via Hugging Face AutoModel."""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str = CLIP_S,
        name: str = "internvideo2",
    ) -> None:
        # Leftover 0.1.7 stubs have no __spec__; importing transformers first
        # makes find_spec raise. Purge, import transformers, then reinstall.
        _purge_broken_flash_attn()
        from transformers import AutoConfig
        import torch

        self.name = name
        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        install_flash_attn_stub()
        try:
            config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
            _force_naive_attention(config)
            self.model = _build_internvideo2_model(checkpoint, config, dtype)
        except Exception as exc:
            if checkpoint == CLIP_1B:
                raise RuntimeError(_ONE_B_HELP) from exc
            raise
        if not hasattr(self.model, "encode_vision") or not hasattr(
            self.model, "encode_text"
        ):
            if checkpoint == CLIP_1B:
                raise RuntimeError(_ONE_B_HELP)
            raise RuntimeError(
                f"{checkpoint} loaded but has no encode_vision/encode_text API."
            )
        place_model(self.model, self.device, dtype=dtype)
        if hasattr(self.model, "device"):
            try:
                self.model.device = self.device
            except Exception:
                pass

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 2,
    ) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(videos), batch_size),
            desc=f"{self.name} videos",
            unit="batch",
        ):
            batch = videos[start : start + batch_size]
            transformed = []
            for paths in batch:
                frames = frames_to_uint8_tchw(paths)
                video = self.model.transform(frames)
                transformed.append(video)

            pixel_values = torch.stack(transformed, dim=0).to(self.device)
            features = self._forward_videos(pixel_values)
            chunks.append(to_numpy(features))
        return np.concatenate(chunks, axis=0)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"{self.name} texts",
            unit="batch",
        ):
            batch = texts[start : start + batch_size]
            text_input = self._tokenize(batch)
            features = self._forward_texts(text_input)
            chunks.append(to_numpy(features))
        return np.concatenate(chunks, axis=0)

    def _tokenize(self, texts: list[str]) -> object:
        encoded = self.model.tokenizer(texts)
        if hasattr(encoded, "to"):
            return encoded.to(self.device)
        if isinstance(encoded, dict):
            return {
                key: value.to(self.device)
                for key, value in encoded.items()
                if hasattr(value, "to")
            }
        return encoded

    def _forward_videos(self, pixel_values: object) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_vision(pixel_values, test=True)
            else:
                features = self.model.encode_vision(pixel_values, test=True)
        return l2_normalize_torch(unwrap_features(features))

    def _forward_texts(self, text_input: object) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_text(text_input)
            else:
                features = self.model.encode_text(text_input)
        return l2_normalize_torch(unwrap_features(features))


def build_internvideo2_clip_s(device: str = "cuda") -> InternVideo2Encoder:
    return InternVideo2Encoder(
        device=device,
        checkpoint=CLIP_S,
        name="internvideo2",
    )


def build_internvideo2_clip_1b(device: str = "cuda") -> InternVideo2Encoder:
    return InternVideo2Encoder(
        device=device,
        checkpoint=CLIP_1B,
        name="internvideo2_clip_1b",
    )


def build_internvideo2_s2_1b(device: str = "cuda"):
    """InternVideo2-1B stage-2 retrieval model (BERT text encoder, 4 frames)."""

    from shawaf_vlm.models.internvideo2_s2 import InternVideo2S2Encoder

    return InternVideo2S2Encoder(device=device)
