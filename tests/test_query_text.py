from __future__ import annotations

import numpy as np
import pytest

from shawaf_vlm.query_text import TEMPLATES, encode_queries, normalize_query, render_queries


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("girl in hijab ,red blouse is drinking", "girl in hijab, red blouse is drinking"),
        ("a boy is wearing dark grey shirt , grey sweatpants and drinking", "a boy is wearing dark grey shirt, grey sweatpants and drinking"),
        ("  A Girl   wears black,,blue jeans.  ", "a girl wears black, blue jeans"),
    ],
)
def test_normalize_query(raw: str, clean: str) -> None:
    assert normalize_query(raw) == clean


def test_render_modes() -> None:
    assert render_queries(["Girl ,red"], "raw") == [["Girl ,red"]]
    assert render_queries(["Girl ,red"], "clean") == [["girl, red"]]
    assert render_queries(["girl ,red"], "template") == [["a photo of a girl, red."]]
    assert render_queries(["a boy"], "template") == [["a photo of a boy."]]
    assert len(render_queries(["x"], "ensemble")[0]) == len(TEMPLATES)
    with pytest.raises(ValueError):
        render_queries(["x"], "fancy")


class _Enc:
    def encode_texts(self, texts, batch_size=32):
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_encode_queries_shapes_and_unit_norm() -> None:
    for mode in ("raw", "clean", "template", "ensemble"):
        out = encode_queries(_Enc(), ["a boy in red", "girl ,blue"], mode)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
