"""Split a person caption into its appearance part and its action part.

Alexandria captions mostly read "<appearance> is <action>". This module splits
them with a few rules, maps the action to a coarse class, and flags rows a
human should check. The output feeds the action-gap evaluation
(``scripts/action_gap.py``): full caption vs appearance only vs action only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Fixed before looking at any retrieval result.
ACTION_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("falling", ("fall", "faint")),
    ("watching fight", ("watch", "watching the fight")),
    ("fighting", ("fight",)),
    ("phone", ("phone",)),
    ("laptop / reading", ("laptop", "reading", "studying", "working on")),
    ("drinking", ("drink",)),
    ("eating", ("eat", "chips")),
    ("dancing", ("danc",)),
    ("walking", ("walk",)),
    ("playing", ("playing",)),
    ("presenting / teaching", ("present", "teach", "listening to his teacher")),
    ("talking", ("talk", "taking", "gossip", "shaking hands", "chatting")),
    ("standing / looking", ("standing", "looking", "decorating", "mirror", "mirrir")),
)

_TYPOS = {"phonew": "phone", "mirrir": "mirror", "waers": "wears", "greet": "green", "iN": "in"}
_ING = r"(?:drinking|talking|taking|eating|reading|studying|fighting|dancing|walking|playing|standing|gossiping|presenting|teaching|listening|looking|decorating|working|watching|shaking)"
_SPLITS = (
    # "<appearance> is <action>" (the common case; last ' is ' wins)
    re.compile(r"^(?P<app>.+?)\s+is\s+(?P<act>(?:on |an? )?" + _ING + r".*|on (?:his|her) (?:phone|laptop).*|on talking.*)$", re.I),
    # "<appearance> and/, <-ing verb ...>"
    re.compile(r"^(?P<app>.+?)\s*(?:,|\band\b)\s*(?P<act>" + _ING + r"\b.*)$", re.I),
    # "<appearance> on his/her phone|laptop"
    re.compile(r"^(?P<app>.+?)\s+(?P<act>on (?:his|her) (?:phone|laptop).*)$", re.I),
    # "<appearance> <-ing verb ...>" with no connector
    re.compile(r"^(?P<app>.+?)\s+(?P<act>" + _ING + r"\b.*)$", re.I),
    # "<appearance> watches|fainted|rescued ..."
    re.compile(r"^(?P<app>.+?)\s+(?P<act>(?:watches|fainted|rescued)\b.*)$", re.I),
)


@dataclass(frozen=True)
class SplitCaption:
    full: str
    appearance: str
    action: str
    action_class: str
    flag: str  # "" when the rules are confident


def _clean(text: str) -> str:
    out = " ".join(str(text).replace("_", " ").split())
    for wrong, right in _TYPOS.items():
        out = re.sub(rf"\b{wrong}\b", right, out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",(?=\S)", ", ", out)
    return out.strip(" ,.")


def classify_action(action: str) -> str:
    low = action.lower()
    for name, keys in ACTION_CLASSES:
        if any(key in low for key in keys):
            return name
    return "other" if low else "none"


def split_caption(caption: str) -> SplitCaption:
    text = _clean(caption)
    for pattern in _SPLITS:
        match = pattern.match(text)
        if match:
            app = match.group("app").strip(" ,")
            act = match.group("act").strip(" ,")
            cls = classify_action(act)
            flag = "" if cls not in ("other", "none") else "check action class"
            if len(app.split()) < 2:
                flag = "check appearance"
            return SplitCaption(caption, app, act, cls, flag)
    return SplitCaption(caption, text, "", "none", "no action found")


def action_query(action: str) -> str:
    """Turn an action phrase into a stand-alone query: 'drinking' -> 'a person drinking'."""

    act = action.strip()
    if not act:
        return ""
    act = re.sub(r"^is\s+", "", act)
    return f"a person {act}"
