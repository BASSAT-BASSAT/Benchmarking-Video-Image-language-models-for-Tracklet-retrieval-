from __future__ import annotations

from shawaf_vlm.semantic_labels import parse_attributes, relevance


def test_parse_basic_attributes() -> None:
    a = parse_attributes("a boy wears black shirt, green pants, glasses is dancing", "dancing")
    assert a.gender == "male" and a.upper == {"black"} and a.lower == {"green"}
    assert a.lower_types == {"pants"} and a.glasses and a.action == "dancing"


def test_hijab_and_denim_jacket() -> None:
    a = parse_attributes("a girl wears white hijab, pink chemise, grey pants")
    assert a.gender == "female" and a.hijab and a.hijab_colour == "white"
    assert a.upper == {"pink"} and a.lower == {"grey"}
    b = parse_attributes("a curly hair boy wears white tshirt and jeans jacket is dancing")
    assert "blue" in b.upper and "jeans" not in b.lower_types and b.curly


def test_relevance_match_maybe_no() -> None:
    q = parse_attributes("a boy wearing glasses drinking", "drinking")
    yes = parse_attributes("a boy in black, glasses is drinking", "drinking")
    maybe = parse_attributes("a boy wearing black and drinking", "drinking")  # glasses not mentioned
    other_action = parse_attributes("a boy in black, glasses is talking", "talking")
    girl = parse_attributes("a girl with glasses is drinking", "drinking")
    assert relevance(q, yes) == ("yes", 1.0)
    assert relevance(q, maybe)[0] == "maybe"
    assert relevance(q, other_action) == ("no", 0.0)
    assert relevance(q, girl) == ("no", 0.0)


def test_shades() -> None:
    light = parse_attributes("a girl wearing black shirt, light blue jeans talking", "talking")
    dark = parse_attributes("a girl wearing black shirt, dark blue jeans talking", "talking")
    plain = parse_attributes("a girl wears black shirt and blue jeans is talking", "talking")
    assert relevance(light, dark) == ("no", 0.0)
    assert relevance(plain, light)[0] == "yes" and relevance(light, plain)[0] == "yes"


def test_colour_contradiction() -> None:
    q = parse_attributes("a girl wears hijab, blue blouse is standing", "standing / looking")
    g = parse_attributes("a girl in hijab, yellow blouse is standing", "standing / looking")
    assert relevance(q, g) == ("no", 0.0)
