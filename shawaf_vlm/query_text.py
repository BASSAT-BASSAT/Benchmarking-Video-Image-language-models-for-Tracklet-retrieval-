"""Query-side text handling: normalization and prompt templates.

Modes (fixed before looking at Alexandria results, so they are not tuned on it):

- ``raw``       the caption exactly as typed (original behaviour)
- ``clean``     whitespace / punctuation normalized, lower-cased
- ``template``  ``clean`` wrapped as "a photo of {caption}."
- ``ensemble``  mean of L2-normalized embeddings over ``TEMPLATES``

Gallery embeddings are untouched, so switching modes never needs a re-index.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

QUERY_MODES = ("raw", "clean", "template", "ensemble")

TEMPLATES = (
    "a photo of {}.",
    "a surveillance camera image of {}.",
    "{}.",
)

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_PUNCT_NO_SPACE = re.compile(r"([,;:])(?=\S)")
_REPEATED_PUNCT = re.compile(r"([,.;:!?])\1+")
_MULTI_SPACE = re.compile(r"\s+")
_LEADING_ARTICLE = re.compile(r"^(a|an|the|one)\s", re.IGNORECASE)


def normalize_query(text: str) -> str:
    """Tidy a typed caption without changing its words.

    ``"girl in hijab ,red blouse is drinking"`` ->
    ``"girl in hijab, red blouse is drinking"``
    """

    out = str(text).strip().lower()
    out = _MULTI_SPACE.sub(" ", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _REPEATED_PUNCT.sub(r"\1", out)
    out = _PUNCT_NO_SPACE.sub(r"\1 ", out)
    return out.strip(" ,.;:")


def _with_article(caption: str) -> str:
    """'girl in red' -> 'a girl in red' so templates read naturally."""

    if not caption or _LEADING_ARTICLE.match(caption):
        return caption
    return f"a {caption}"


def render_queries(texts: Sequence[str], mode: str) -> list[list[str]]:
    """Return, per query, the list of strings to embed (one, or one per template)."""

    if mode not in QUERY_MODES:
        raise ValueError(f"query mode must be one of {QUERY_MODES}, got {mode!r}")
    if mode == "raw":
        return [[str(t)] for t in texts]
    cleaned = [normalize_query(t) for t in texts]
    if mode == "clean":
        return [[c] for c in cleaned]
    if mode == "template":
        return [[TEMPLATES[0].format(_with_article(c))] for c in cleaned]
    return [[tpl.format(_with_article(c)) for tpl in TEMPLATES] for c in cleaned]


def encode_queries(encoder, texts: Sequence[str], mode: str = "raw", batch_size: int = 32) -> np.ndarray:
    """Embed queries under ``mode``; template ensembles are averaged then L2-normalized."""

    groups = render_queries(texts, mode)
    flat = [s for group in groups for s in group]
    feats = np.asarray(encoder.encode_texts(flat, batch_size=batch_size), dtype=np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)
    out = np.empty((len(groups), feats.shape[1]), dtype=np.float32)
    row = 0
    for i, group in enumerate(groups):
        mean = feats[row : row + len(group)].mean(axis=0)
        out[i] = mean / max(float(np.linalg.norm(mean)), 1e-12)
        row += len(group)
    return out
