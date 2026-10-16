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
}

# Registry keys kept for manual runs but excluded from standard benchmark groups.
OPT_IN_MODELS: tuple[str, ...] = ("internvideo2_clip_1b",)


def models_for_group(group: str | None) -> tuple[str, ...] | None:
    if group is None or group == "all":
        return None
    if group not in MODEL_GROUPS:
        raise KeyError(f"Unknown model group {group!r}")
    return MODEL_GROUPS[group]
