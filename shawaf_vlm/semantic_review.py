"""Conservative proposals for Alexandria ``maybe`` semantic pairs."""

from __future__ import annotations

import re

from shawaf_vlm.semantic_labels import _lower_type_ok, colour_match

SOFT = frozenset({"bag", "cap", "curly", "glasses"})
# Caption evidence the candidate's hair is uncovered (so a hijab query should not match).
_UNCOVERED_HAIR = re.compile(
    r"\b(long hair|ponytail|curly|cap|hat|bald)\b",
    re.I,
)


def unknown_attrs(q, g) -> list[str]:
    unknown = []
    for name in q.stated():
        if name == "gender":
            ok = None if g.gender is None else (g.gender == q.gender)
        elif name.startswith("upper:"):
            ok = None if not g.upper else colour_match(name.split(":")[1], g.upper)
        elif name.startswith("lower:"):
            ok = None if not g.lower else colour_match(name.split(":")[1], g.lower)
        elif name.startswith("lower_type:"):
            ok = _lower_type_ok(name.split(":")[1], g.lower_types)
        elif name.startswith("hijab:"):
            ok = None if not g.hijab_colour else colour_match(name.split(":")[1], {g.hijab_colour})
        elif name == "action":
            ok = None if not g.action or g.action == "none" else (g.action == q.action)
        else:
            ok = True if getattr(g, name) else None
        if ok is None:
            unknown.append(name)
    return unknown


def propose(score: float, unknown: list[str], candidate_caption: str | None = None) -> str:
    """Return ``yes``, ``no``, or ``\"\"`` under the soft review policy."""

    soft_only = bool(unknown) and all(u in SOFT for u in unknown)
    hard_unknown = [u for u in unknown if u not in SOFT]
    hijab_unknown = [u for u in unknown if u == "hijab" or u.startswith("hijab:")]
    if (
        candidate_caption
        and hijab_unknown
        and set(unknown) <= set(hijab_unknown)
        and _UNCOVERED_HAIR.search(candidate_caption)
        and "hijab" not in candidate_caption.lower()
    ):
        return "no"
    if soft_only and score >= 0.75:
        return "yes"
    if score >= 0.85 and not hard_unknown:
        return "yes"
    if score < 0.35 and hard_unknown:
        return "no"
    return ""
