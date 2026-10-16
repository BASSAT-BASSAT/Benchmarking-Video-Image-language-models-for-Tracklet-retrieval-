from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.needs_transformers
def test_xclip_video_forward_uses_return_dict_on_mit() -> None:
    pytest.importorskip("torch")
    import torch

    from shawaf_vlm.models.xclip import XCLIPEncoder

    encoder = XCLIPEncoder.__new__(XCLIPEncoder)
    encoder.device = "cpu"
    encoder.name = "xclip"

    pooler = torch.ones(2, 512)
    vision_out = SimpleNamespace(pooler_output=torch.ones(16, 768))
    mit_out = SimpleNamespace(pooler_output=pooler)

    encoder.model = MagicMock()
    encoder.model.vision_model.return_value = vision_out
    encoder.model.visual_projection.side_effect = lambda x: x
    encoder.model.mit.return_value = mit_out

    pixel_values = torch.zeros(2, 8, 3, 224, 224)
    out = encoder._forward_videos(pixel_values)

    encoder.model.mit.assert_called_once()
    _, kwargs = encoder.model.mit.call_args
    assert kwargs.get("return_dict") is True
    assert out.shape == (2, 512)
