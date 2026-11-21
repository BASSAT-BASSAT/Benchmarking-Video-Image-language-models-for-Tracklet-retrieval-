from shawaf_vlm.semantic_labels import parse_attributes
from shawaf_vlm.semantic_review import propose, unknown_attrs


def test_propose_soft_yes() -> None:
    assert propose(0.8, ["glasses"]) == "yes"
    assert propose(0.9, []) == "yes"
    assert propose(0.2, ["upper:red"]) == "no"
    assert propose(0.5, ["upper:red"]) == ""
    assert propose(0.6, ["hijab"], "a girl with long hair wearing black") == "no"


def test_unknown_attrs_detects_omission() -> None:
    q = parse_attributes("a boy wearing glasses drinking", "drinking")
    g = parse_attributes("a boy drinking", "drinking")
    unknown = unknown_attrs(q, g)
    assert "glasses" in unknown
