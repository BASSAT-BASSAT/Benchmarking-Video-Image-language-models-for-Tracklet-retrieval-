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
            "Not in the T4 notebook. The Hub file is a 14 MB add-on; the text "
            "tower is the 25 GB InternVL-C checkpoint."
        ),
    ),
    "siglip2": EncoderSpec(
        key="siglip2",
        label="SigLIP 2 So400m",
        checkpoint="google/siglip2-so400m-patch14-384",
        builder="siglip2",
        notes="Frame encoder, 384px, mean-pooled inside each clip. Public.",
    ),
    "pe_core_l14": EncoderSpec(
        key="pe_core_l14",
        label="Perception Encoder L/14",
        checkpoint="timm/PE-Core-L-14-336",
        builder="pe_core_l14",
        notes="Meta PE-Core, 336px, 32-token text, mean-pooled frames. Public.",
    ),
    "irra": EncoderSpec(
        key="irra",
        label="IRRA ViT-B/16",
        checkpoint="IRRA CUHK-PEDES",
        builder="irra",
        notes=(
            "CVPR 2023 person-description CLIP. Frames are resized to 384x128. "
            "Zero-shot on TVPReid, trained on CUHK-PEDES."
        ),
    ),
    "internvideo2_s2_1b": EncoderSpec(
        key="internvideo2_s2_1b",
        label="InternVideo2-1B-s2",
        checkpoint="OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        builder="internvideo2_s2_1b",
        notes=(
            "BERT-large text encoder plus the 1B stage-2 video encoder. "
            "4 middle frames, 512-d cosine. Gated Hugging Face weights."
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


def _build_internvideo2_s2_1b(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.internvideo2 import build_internvideo2_s2_1b

    return build_internvideo2_s2_1b(device=device)


def _build_siglip2(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import Siglip2Encoder

    return Siglip2Encoder(device=device)


def _build_pe_core_l14(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import PerceptionEncoder

    return PerceptionEncoder(device=device)


def _build_irra(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import IrraEncoder

    return IrraEncoder(device=device)


_BUILDERS: dict[str, Callable[..., VideoTextEncoder]] = {
    "xclip": _build_xclip,
    "languagebind": _build_languagebind,
    "internvideo2": _build_internvideo2,
    "internvideo2_clip_1b": _build_internvideo2_clip_1b,
    "internvideo2_s2_1b": _build_internvideo2_s2_1b,
    "siglip2": _build_siglip2,
    "pe_core_l14": _build_pe_core_l14,
    "irra": _build_irra,
}
