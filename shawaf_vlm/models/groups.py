"""Encoder groups for Kaggle notebooks, Colab, and the Gradio demo."""

from __future__ import annotations

# Tier-A zero-shot matrix (one Kaggle session per group; incompatible transformers pins).
MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "base": (
        "siglip2",
        "pe_core_l14",
        "irra",
        "internvideo2",
        "internvideo2_s2_1b",
        "xclip",
        "languagebind",
    ),
    "extended_modern": (
        "openai_clip_vit_l14",
        "jina_clip_v2",
        "qwen3_vl_embed_2b",
    ),
    "extended_gme": ("gme_qwen2_vl_2b",),
    # Separate env: pip install -e ".[embeddinggemma2]" (Sentence Transformers + recent transformers).
    "embeddinggemma2": ("embeddinggemma2",),
    # Off-the-shelf smaller encoders for the ingestion-speed sweep. Not part of Tier-A headlines.
    # mobileclip2_s2 / mobileclip2_s4 are left out: under open_clip 3.3 / timm 1.0.27
    # their fastvit_mci towers fail a zero-shot sanity check (chance-level retrieval).
    "ingestion_fast": (
        "pe_core_b16",
        "pe_core_s16",
        "pe_core_t16",
        "mobileclip2_b",
    ),
    # General CLIP models in PE-L's size class (open_clip).
    "clip_large": (
        "eva02_clip_l14",
        "dfn_clip_l14",
    ),
}

# Registry keys kept for manual runs but excluded from standard benchmark groups.
OPT_IN_MODELS: tuple[str, ...] = ("internvideo2_clip_1b", "pe_core_distilled")


def models_for_group(group: str | None) -> tuple[str, ...] | None:
    if group is None or group == "all":
        return None
    if group not in MODEL_GROUPS:
        raise KeyError(f"Unknown model group {group!r}")
    return MODEL_GROUPS[group]
