from __future__ import annotations

import pytest

from shawaf_vlm.action_split import action_query, classify_action, split_caption


@pytest.mark.parametrize(
    ("caption", "appearance", "action", "cls"),
    [
        ("girl in hijab ,red blouse is drinking", "girl in hijab, red blouse", "drinking", "drinking"),
        ("a boy wearing black and drinking", "a boy wearing black", "drinking", "drinking"),
        ("a curly hair girl wears white shirt and blue shirt on her phone",
         "a curly hair girl wears white shirt and blue shirt", "on her phone", "phone"),
        ("a boy wears black tshirt, glasses  and green pants is talking on his phonew",
         "a boy wears black tshirt, glasses and green pants", "talking on his phone", "phone"),
        ("a girl in red hoodie watches the fight", "a girl in red hoodie", "watches the fight", "watching fight"),
        ("a girl in hijab , yellow blouse rescued the girl who fainted",
         "a girl in hijab, yellow blouse", "rescued the girl who fainted", "fainting / helping"),
    ],
)
def test_split_caption(caption, appearance, action, cls) -> None:
    s = split_caption(caption)
    assert (s.appearance, s.action, s.action_class) == (appearance, action, cls)


def test_no_action_is_flagged() -> None:
    s = split_caption("a boy wears orange tshirt , glasses and black bag")
    assert s.action == "" and s.action_class == "none" and s.flag == "no action found"


def test_action_query_and_classes() -> None:
    assert action_query("drinking") == "a person drinking"
    assert action_query("") == ""
    assert classify_action("is taking") == "talking"
    assert classify_action("staring blankly") == "other"
