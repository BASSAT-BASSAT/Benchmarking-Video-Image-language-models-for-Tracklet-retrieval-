"""Kaggle-style encoder environment groups for the demo UI."""

from __future__ import annotations

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
}


def models_for_group(group: str | None) -> tuple[str, ...] | None:
    if group is None or group == "all":
        return None
    if group not in MODEL_GROUPS:
        raise KeyError(f"Unknown model group {group!r}")
    return MODEL_GROUPS[group]
