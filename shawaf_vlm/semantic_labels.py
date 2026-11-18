"""Meaning-based relevance between a text query and every gallery person.

Hard scoring accepts only the exact labelled person. A caption such as "a boy
wearing glasses drinking" can describe several people, so we also score
retrieval against *semantic* relevance (Wray et al., CVPR 2021; Chun et al.,
ECCV Caption, 2022).

Each caption is parsed into attributes: gender, upper and lower garments with
colour, hijab (+ colour), glasses, bag, cap, hair, and the action class. For a
query q and a gallery person g (described by g's own caption):

- any **contradiction** (other gender, another colour for the same garment
  slot, another action class) makes g irrelevant (score 0, label ``no``);
- if every attribute q states is present in g's caption, g is a **match**
  (label ``yes``);
- otherwise, with no contradiction but something q states that g's caption does
  not mention, g is a **maybe**: a human decides.

The graded score is the share of q's attributes g satisfies (0 on any
contradiction). The true person always has score 1 and label ``yes``.
Labels come from captions only, never from a model's ranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COLOURS = {
    "black": "black", "white": "white", "grey": "grey", "gray": "grey",
    "blue": "blue", "navy": "blue", "red": "red", "burgundy": "red", "pink": "pink",
    "green": "green", "yellow": "yellow", "orange": "orange", "brown": "brown",
    "beige": "beige", "purple": "purple",
}
UPPER = ("tshirt", "t shirt", "t-shirt", "shirt", "blouse", "chemise", "jacket", "hoodie", "top", "outfit")
LOWER = ("sweatpants", "jeans", "pants", "skirt", "trousers")
MALE = re.compile(r"\b(boy|man|he|his|him|guy)\b")
FEMALE = re.compile(r"\b(girl|girls|woman|she|her|hijab|hijabi|dr)\b")
_COLOUR_WORD = r"((?:light |dark |baby |navy )?(?:" + "|".join(sorted(COLOURS, key=len, reverse=True)) + r"))"
_LIGHT = {"light", "baby"}
_DARK = {"dark", "navy"}


def _colour(phrase: str) -> str:
    """'light blue' -> 'blue/light', 'navy blue' -> 'blue/dark', 'grey' -> 'grey'."""

    words = phrase.split()
    base = COLOURS[words[-1]]
    if words[-1] == "navy":
        return f"{base}/dark"
    if len(words) > 1 and words[0] in _LIGHT:
        return f"{base}/light"
    if len(words) > 1 and words[0] in _DARK:
        return f"{base}/dark"
    return base


def colour_match(q: str, colours: set[str]) -> bool:
    """Same base colour, and the same shade unless either side gives none."""

    qb, _, qs = q.partition("/")
    for c in colours:
        cb, _, cs = c.partition("/")
        if cb == qb and (not qs or not cs or qs == cs):
            return True
    return False


@dataclass
class Attributes:
    gender: str | None = None
    upper: set[str] = field(default_factory=set)      # colours worn on top (or overall outfit colour)
    lower: set[str] = field(default_factory=set)      # colours on the legs
    lower_types: set[str] = field(default_factory=set)
    hijab: bool = False
    hijab_colour: str | None = None
    glasses: bool = False
    bag: bool = False
    cap: bool = False
    curly: bool = False
    action: str | None = None

    def stated(self) -> list[str]:
        """Attribute names the caption actually states (what a query asks for)."""

        out = []
        if self.gender:
            out.append("gender")
        out += [f"upper:{c}" for c in sorted(self.upper)]
        out += [f"lower:{c}" for c in sorted(self.lower)]
        out += [f"lower_type:{t}" for t in sorted(self.lower_types)]
        for flag in ("hijab", "glasses", "bag", "cap", "curly"):
            if getattr(self, flag):
                out.append(flag)
        if self.hijab_colour:
            out.append(f"hijab:{self.hijab_colour}")
        if self.action and self.action != "none":
            out.append("action")
        return out


def parse_attributes(caption: str, action_class: str | None = None) -> Attributes:
    text = " " + re.sub(r"[^a-z ]+", " ", caption.lower().replace("t-shirt", "tshirt").replace("t_shirt", "tshirt")) + " "
    text = re.sub(r"\s+", " ", text)
    a = Attributes(action=action_class)
    male, female = bool(MALE.search(text)), bool(FEMALE.search(text))
    a.gender = "male" if male and not female else "female" if female and not male else None

    # "jeans jacket" is a denim jacket: blue upper garment, not trousers.
    if "jeans jacket" in text:
        a.upper.add("blue")
        text = text.replace("jeans jacket", " jacket ")

    for m in re.finditer(_COLOUR_WORD + r" (" + "|".join(UPPER) + r")\b", text):
        a.upper.add(_colour(m.group(1)))
    for m in re.finditer(_COLOUR_WORD + r" (" + "|".join(LOWER) + r")\b", text):
        a.lower.add(_colour(m.group(1)))
    for t in LOWER:
        if re.search(rf"\b{t}\b", text):
            a.lower_types.add("pants" if t == "trousers" else t)
    # "in black", "wearing black", "wears pink": overall outfit colour.
    for m in re.finditer(r"\b(?:in|wearing|wears|wear) " + _COLOUR_WORD + r"(?! (?:" + "|".join(UPPER + LOWER) + r"|hijab))\b", text):
        a.upper.add(_colour(m.group(1)))

    a.hijab = bool(re.search(r"\bhijab|hijabi\b", text))
    hc = re.search(_COLOUR_WORD + r" hijab", text)
    a.hijab_colour = _colour(hc.group(1)) if hc else None
    a.glasses = " glasses " in text
    a.bag = " bag " in text
    a.cap = " cap " in text
    a.curly = " curly " in text
    return a


def _lower_type_ok(q: str, g_types: set[str]) -> bool | None:
    if not g_types:
        return None
    if q == "pants" or "pants" in g_types:
        return True  # "pants" is generic
    return q in g_types


def relevance(q: Attributes, g: Attributes) -> tuple[str, float]:
    """Return (label, graded score) for gallery person g against query q."""

    stated = q.stated()
    if not stated:
        return "maybe", 0.0
    matched = 0
    unknown = 0
    for name in stated:
        ok: bool | None
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
        else:  # hijab, glasses, bag, cap, curly: absence in a caption is not proof of absence
            ok = True if getattr(g, name) else None
        if ok is False:
            return "no", 0.0
        if ok is None:
            unknown += 1
        else:
            matched += 1
    score = matched / len(stated)
    return ("yes" if unknown == 0 else "maybe"), score
