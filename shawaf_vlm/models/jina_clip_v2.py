"""Jina CLIP v2 as a frozen frame/text encoder."""

from __future__ import annotations

from PIL import Image

from shawaf_vlm.models.frame_encoders import _FrameEncoder
from shawaf_vlm.models.runtime import (
    l2_normalize_torch,
    load_pretrained,
    place_model,
    unwrap_features,
)

JINA_CLIP_V2 = "jinaai/jina-clip-v2"


def _to_torch_tensor(features):
    """Accept Jina's tensor or NumPy return types with one stable contract."""

    import torch

    value = unwrap_features(features)
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


class JinaClipV2Encoder(_FrameEncoder):
    """Official Jina CLIP v2 remote-code model, pooled per Stage 1 clip."""

    name = "jina_clip_v2"
    checkpoint = JINA_CLIP_V2
    frame_microbatch = 1

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel

        super().__init__(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = load_pretrained(
            AutoModel.from_pretrained,
            self.checkpoint,
            dtype,
            trust_remote_code=True,
        )
        place_model(self.model, self.device, dtype=dtype)

    def _encode_images(self, images: list[Image.Image]):
        features = self.model.encode_image(images, truncate_dim=None)
        return l2_normalize_torch(_to_torch_tensor(features))

    def _encode_text_batch(self, texts: list[str]):
        # No retrieval prompt: the main benchmark stays zero-shot and neutral.
        features = self.model.encode_text(texts, truncate_dim=None)
        return _to_torch_tensor(features)
