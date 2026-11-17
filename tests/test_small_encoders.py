"""Registry wiring for smaller PE-Core and MobileCLIP2 checkpoints. Nothing is downloaded."""

from __future__ import annotations

import sys
import types

import torch
from PIL import Image
from torch import nn

from shawaf_vlm.models.frame_encoders import PerceptionEncoder
from shawaf_vlm.models.groups import MODEL_GROUPS, OPT_IN_MODELS
from shawaf_vlm.models.registry import build_encoder


class _StubClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.patch_size = 14
        self.visual.positional_embedding = nn.Parameter(torch.zeros(577, 8))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        return images.mean(dim=(2, 3))

    def encode_text(self, tokens):
        return torch.ones(tokens.shape[0], 8)


def _install_open_clip(monkeypatch, seen: dict) -> None:
    def create(checkpoint):
        seen["checkpoint"] = checkpoint
        return _StubClip(), (lambda image: torch.zeros(3, 32, 32))

    def tokenizer(checkpoint):
        def _tokens(texts):
            return torch.zeros(len(texts), 4, dtype=torch.long)

        return _tokens

    module = types.ModuleType("open_clip")
    module.create_model_from_pretrained = create
    module.get_tokenizer = tokenizer
    monkeypatch.setitem(sys.modules, "open_clip", module)


def test_ingestion_group_and_opt_in() -> None:
    assert MODEL_GROUPS["ingestion_fast"] == (
        "pe_core_b16",
        "pe_core_s16",
        "pe_core_t16",
        "mobileclip2_b",
    )
    # Broken upstream (chance-level zero-shot); kept in the registry only.
    assert "mobileclip2_s2" not in MODEL_GROUPS["ingestion_fast"]
    assert "pe_core_b16" not in MODEL_GROUPS["base"]
    assert "pe_core_distilled" in OPT_IN_MODELS
    assert "pe_core_distilled" not in MODEL_GROUPS["base"]
    assert "pe_core_distilled" not in MODEL_GROUPS["ingestion_fast"]


def test_build_encoder_passes_checkpoint_without_download(monkeypatch) -> None:
    seen: dict = {}
    _install_open_clip(monkeypatch, seen)
    encoder = build_encoder("pe_core_s16", device="cpu")
    assert seen["checkpoint"] == "hf-hub:timm/PE-Core-S-16-384"
    assert encoder.name == "pe_core_s16"
    assert encoder.image_size is None
    assert encoder._pos_embed_calls == 1
    assert encoder.model.visual.positional_embedding.shape[0] == 577


def test_image_size_rebuilds_preprocess_and_pos_embed(monkeypatch) -> None:
    _install_open_clip(monkeypatch, {})
    encoder = PerceptionEncoder(device="cpu", image_size=224)
    assert encoder._pos_embed_calls == 1
    assert encoder.model.visual.positional_embedding.shape[0] == 16 * 16 + 1
    image = Image.new("RGB", (30, 40), color=(8, 9, 10))
    tensor = encoder.preprocess(image)
    assert tuple(tensor.shape) == (3, 224, 224)


def test_build_encoder_forwards_image_size(monkeypatch) -> None:
    _install_open_clip(monkeypatch, {})
    encoder = build_encoder("pe_core_b16", device="cpu", image_size=224)
    assert encoder.image_size == 224
    assert encoder.model.visual.positional_embedding.shape[0] == 16 * 16 + 1


def test_image_size_none_leaves_pos_embed(monkeypatch) -> None:
    _install_open_clip(monkeypatch, {})
    encoder = PerceptionEncoder(device="cpu", image_size=None)
    assert encoder._pos_embed_calls == 1
    assert tuple(encoder.model.visual.positional_embedding.shape) == (577, 8)
