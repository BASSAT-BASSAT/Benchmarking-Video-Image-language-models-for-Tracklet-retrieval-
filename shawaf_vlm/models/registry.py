from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shawaf_vlm.models.protocol import VideoTextEncoder


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    label: str
    checkpoint: str
    builder: str
    notes: str = ""


_SPECS: dict[str, EncoderSpec] = {
    "xclip": EncoderSpec(
        key="xclip",
        label="X-CLIP Base/32",
        checkpoint="microsoft/xclip-base-patch32",
        builder="xclip",
        notes="Hugging Face native, 8-frame Kinetics CLIP extension.",
    ),
    "languagebind": EncoderSpec(
        key="languagebind",
        label="LanguageBind Video",
        checkpoint="LanguageBind/LanguageBind_Video",
        builder="languagebind",
        notes="Video-language encoder with temporal attention, 8 frames.",
    ),
    "internvideo2": EncoderSpec(
        key="internvideo2",
        label="InternVideo2 CLIP-S",
        checkpoint="OpenGVLab/InternVideo2_CLIP_S",
        builder="internvideo2",
        notes="Hugging Face AutoModel CLIP-S (~373M). Default InternVideo2.",
    ),
    "internvideo2_clip_1b": EncoderSpec(
        key="internvideo2_clip_1b",
        label="InternVideo2 CLIP-1B",
        checkpoint="OpenGVLab/InternVideo2-CLIP-1B-224p-f8",
        builder="internvideo2_clip_1b",
        notes=(
            "Optional. HF repo is a gated LoRA add-on; AutoModel load may fail. "
            "Prefer internvideo2 on Kaggle T4."
        ),
    ),
}


def all_specs() -> dict[str, EncoderSpec]:
    return dict(_SPECS)


def build_encoder(name: str, device: str = "cuda") -> VideoTextEncoder:
    spec = _SPECS.get(name)
    if spec is None:
        known = ", ".join(sorted(_SPECS))
        raise KeyError(f"Unknown encoder {name!r}. Choose one of: {known}")
    builder = _BUILDERS[spec.builder]
    return builder(device=device)


def _build_xclip(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.xclip import XCLIPEncoder

    return XCLIPEncoder(device=device)


def _build_languagebind(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.languagebind import LanguageBindEncoder

    return LanguageBindEncoder(device=device)


def _build_internvideo2(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.internvideo2 import build_internvideo2_clip_s

    return build_internvideo2_clip_s(device=device)


def _build_internvideo2_clip_1b(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.internvideo2 import build_internvideo2_clip_1b

    return build_internvideo2_clip_1b(device=device)


_BUILDERS: dict[str, Callable[..., VideoTextEncoder]] = {
    "xclip": _build_xclip,
    "languagebind": _build_languagebind,
    "internvideo2": _build_internvideo2,
    "internvideo2_clip_1b": _build_internvideo2_clip_1b,
}
