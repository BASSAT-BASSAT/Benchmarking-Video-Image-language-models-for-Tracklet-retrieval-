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
    "pe_core_b16": EncoderSpec(
        key="pe_core_b16",
        label="Perception Encoder B/16",
        checkpoint="timm/PE-Core-B-16",
        builder="pe_core_b16",
        notes="Smaller PE-Core distilled by Meta. 224px. Public.",
    ),
    "pe_core_s16": EncoderSpec(
        key="pe_core_s16",
        label="Perception Encoder S/16",
        checkpoint="timm/PE-Core-S-16-384",
        builder="pe_core_s16",
        notes="Smaller PE-Core distilled by Meta. 384px. Public.",
    ),
    "pe_core_t16": EncoderSpec(
        key="pe_core_t16",
        label="Perception Encoder T/16",
        checkpoint="timm/PE-Core-T-16-384",
        builder="pe_core_t16",
        notes="Smallest PE-Core distilled by Meta. 384px. Public.",
    ),
    "mobileclip2_s2": EncoderSpec(
        key="mobileclip2_s2",
        label="MobileCLIP2-S2",
        checkpoint="timm/MobileCLIP2-S2-OpenCLIP",
        builder="mobileclip2_s2",
        notes=(
            "BROKEN under open_clip 3.3 / timm 1.0.27: fails a zero-shot sanity check "
            "(chance-level retrieval). Do not benchmark. apple-amlr licence."
        ),
    ),
    "mobileclip2_s4": EncoderSpec(
        key="mobileclip2_s4",
        label="MobileCLIP2-S4",
        checkpoint="timm/MobileCLIP2-S4-OpenCLIP",
        builder="mobileclip2_s4",
        notes=(
            "BROKEN under open_clip 3.3 / timm 1.0.27: fails a zero-shot sanity check "
            "(chance-level retrieval). Do not benchmark. apple-amlr licence."
        ),
    ),
    "mobileclip2_b": EncoderSpec(
        key="mobileclip2_b",
        label="MobileCLIP2-B",
        checkpoint="timm/MobileCLIP2-B-OpenCLIP",
        builder="mobileclip2_b",
        notes="Apple MobileCLIP2. apple-amlr licence; check before redistribution.",
    ),
    "eva02_clip_l14": EncoderSpec(
        key="eva02_clip_l14",
        label="EVA02-CLIP-L/14-336",
        checkpoint="timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k",
        builder="eva02_clip_l14",
        notes="General CLIP, same size class as PE-L. MIT licence.",
    ),
    "dfn_clip_l14": EncoderSpec(
        key="dfn_clip_l14",
        label="DFN2B-CLIP-L/14",
        checkpoint="apple/DFN2B-CLIP-ViT-L-14",
        builder="dfn_clip_l14",
        notes="Apple data-filtering-network CLIP, 224px. apple-amlr licence.",
    ),
    "pe_core_distilled": EncoderSpec(
        key="pe_core_distilled",
        label="PE-Core distilled student",
        checkpoint="SHAWAF_PE_DISTILLED",
        builder="pe_core_distilled",
        notes=(
            "Local student checkpoint (env SHAWAF_PE_DISTILLED). Not in the default "
            "benchmark groups. Text queries stay on pe_core_l14 when the file has no text tower."
        ),
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
    "openai_clip_vit_l14": EncoderSpec(
        key="openai_clip_vit_l14",
        label="OpenAI CLIP ViT-L/14",
        checkpoint="OpenAI ViT-L/14",
        builder="openai_clip_vit_l14",
        notes="Official OpenAI frame encoder; normalized frame mean per clip.",
    ),
    "jina_clip_v2": EncoderSpec(
        key="jina_clip_v2",
        label="Jina CLIP v2",
        checkpoint="jinaai/jina-clip-v2",
        builder="jina_clip_v2",
        notes="Frame encoder; normalized frame mean per clip.",
    ),
    "gme_qwen2_vl_2b": EncoderSpec(
        key="gme_qwen2_vl_2b",
        label="GME-Qwen2-VL-2B",
        checkpoint="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        builder="gme_qwen2_vl_2b",
        notes=(
            "Official Sentence Transformers path; image-only visual input is "
            "mean-pooled across normalized frame embeddings."
        ),
    ),
    "qwen3_vl_embed_2b": EncoderSpec(
        key="qwen3_vl_embed_2b",
        label="Qwen3-VL-Embedding-2B",
        checkpoint="Qwen/Qwen3-VL-Embedding-2B",
        builder="qwen3_vl_embed_2b",
        notes="Official native-video embedding path, fp16 and batch 1 on T4.",
    ),
    "embeddinggemma2": EncoderSpec(
        key="embeddinggemma2",
        label="EmbeddingGemma 2",
        checkpoint="google/embeddinggemma-2",
        builder="embeddinggemma2",
        notes=(
            "Hub google/embeddinggemma-2: ST>=6.1, transformers>=5, SearchQuery + image frames; "
            "audio_config=None. Separate env from base Kaggle (pip uninstall torchcodec on Kaggle)."
        ),
    ),
}


def all_specs() -> dict[str, EncoderSpec]:
    return dict(_SPECS)


def build_encoder(
    name: str,
    device: str = "cuda",
    image_size: int | None = None,
    fit: str | None = None,
) -> VideoTextEncoder:
    """``image_size`` / ``fit`` (native, squash, letterbox) apply to PE-Core and MobileCLIP2."""

    spec = _SPECS.get(name)
    if spec is None:
        known = ", ".join(sorted(_SPECS))
        raise KeyError(f"Unknown encoder {name!r}. Choose one of: {known}")
    builder = _BUILDERS[spec.builder]
    kwargs = {key: value for key, value in (("image_size", image_size), ("fit", fit)) if value is not None}
    if not kwargs:
        return builder(device=device)
    try:
        return builder(device=device, **kwargs)
    except TypeError as exc:
        if image_size is not None:
            raise ValueError(f"{name} does not take {', '.join(kwargs)}.") from exc
    # Encoders with their own image processor: pad crops before that processor.
    from shawaf_vlm.models.frame_encoders import make_prefit

    prefit = make_prefit(fit)
    encoder = builder(device=device)
    encoder.prefit = prefit
    encoder.fit = fit
    return encoder


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


def _build_pe_core_l14(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import PerceptionEncoder

    return PerceptionEncoder(device=device, image_size=image_size, fit=fit)


def _build_pe_core_b16(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import PeCoreB16Encoder

    return PeCoreB16Encoder(device=device, image_size=image_size, fit=fit)


def _build_pe_core_s16(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import PeCoreS16Encoder

    return PeCoreS16Encoder(device=device, image_size=image_size, fit=fit)


def _build_pe_core_t16(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import PeCoreT16Encoder

    return PeCoreT16Encoder(device=device, image_size=image_size, fit=fit)


def _build_mobileclip2_s2(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import MobileClip2S2Encoder

    return MobileClip2S2Encoder(device=device, image_size=image_size, fit=fit)


def _build_mobileclip2_s4(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import MobileClip2S4Encoder

    return MobileClip2S4Encoder(device=device, image_size=image_size, fit=fit)


def _build_mobileclip2_b(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import MobileClip2BEncoder

    return MobileClip2BEncoder(device=device, image_size=image_size, fit=fit)


def _build_eva02_clip_l14(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import Eva02ClipL14Encoder

    return Eva02ClipL14Encoder(device=device, image_size=image_size, fit=fit)


def _build_dfn_clip_l14(device: str, image_size: int | None = None, fit: str | None = None) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import DfnClipL14Encoder

    return DfnClipL14Encoder(device=device, image_size=image_size, fit=fit)


def _build_pe_core_distilled(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import DistilledPeEncoder

    return DistilledPeEncoder(device=device)


def _build_irra(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.frame_encoders import IrraEncoder

    return IrraEncoder(device=device)


def _build_openai_clip_vit_l14(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.openai_clip import OpenAIClipEncoder

    return OpenAIClipEncoder(device=device)


def _build_jina_clip_v2(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.jina_clip_v2 import JinaClipV2Encoder

    return JinaClipV2Encoder(device=device)


def _build_gme_qwen2_vl_2b(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.gme_qwen2_vl import GmeQwen2VLEncoder

    return GmeQwen2VLEncoder(device=device)


def _build_qwen3_vl_embed_2b(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.qwen3_vl_embedding import Qwen3VLEmbeddingEncoder

    return Qwen3VLEmbeddingEncoder(device=device)


def _build_embeddinggemma2(device: str) -> VideoTextEncoder:
    from shawaf_vlm.models.embeddinggemma2 import EmbeddingGemma2Encoder

    return EmbeddingGemma2Encoder(device=device)


_BUILDERS: dict[str, Callable[..., VideoTextEncoder]] = {
    "xclip": _build_xclip,
    "languagebind": _build_languagebind,
    "internvideo2": _build_internvideo2,
    "internvideo2_clip_1b": _build_internvideo2_clip_1b,
    "internvideo2_s2_1b": _build_internvideo2_s2_1b,
    "siglip2": _build_siglip2,
    "pe_core_l14": _build_pe_core_l14,
    "pe_core_b16": _build_pe_core_b16,
    "pe_core_s16": _build_pe_core_s16,
    "pe_core_t16": _build_pe_core_t16,
    "mobileclip2_s2": _build_mobileclip2_s2,
    "mobileclip2_s4": _build_mobileclip2_s4,
    "mobileclip2_b": _build_mobileclip2_b,
    "pe_core_distilled": _build_pe_core_distilled,
    "eva02_clip_l14": _build_eva02_clip_l14,
    "dfn_clip_l14": _build_dfn_clip_l14,
    "irra": _build_irra,
    "openai_clip_vit_l14": _build_openai_clip_vit_l14,
    "jina_clip_v2": _build_jina_clip_v2,
    "gme_qwen2_vl_2b": _build_gme_qwen2_vl_2b,
    "qwen3_vl_embed_2b": _build_qwen3_vl_embed_2b,
    "embeddinggemma2": _build_embeddinggemma2,
}
