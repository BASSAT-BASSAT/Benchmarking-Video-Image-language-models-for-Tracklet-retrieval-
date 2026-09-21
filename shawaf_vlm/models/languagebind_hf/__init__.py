"""Official LanguageBind video classes, vendored for transformers 5."""

from shawaf_vlm.models.languagebind_hf.configuration_video import LanguageBindVideoConfig

__all__ = ["LanguageBindVideo", "LanguageBindVideoConfig"]


def __getattr__(name: str):
    if name == "LanguageBindVideo":
        from shawaf_vlm.models.languagebind_hf.modeling_video import LanguageBindVideo

        return LanguageBindVideo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
