from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.needs_transformers
def test_load_siglip2_processor_falls_back_to_gemma_tokenizer() -> None:
    from shawaf_vlm.models.frame_encoders import _Siglip2ProcessorShim, load_siglip2_processor

    fake_image = MagicMock(name="image_processor")
    fake_tokenizer = MagicMock(name="tokenizer")

    with patch(
        "transformers.AutoProcessor.from_pretrained",
        side_effect=TypeError("wrong tokenizer"),
    ):
        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=fake_image,
        ):
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ):
                proc = load_siglip2_processor("google/siglip2-so400m-patch14-384")

    assert isinstance(proc, _Siglip2ProcessorShim)
    assert proc.image_processor is fake_image
    assert proc.tokenizer is fake_tokenizer
