"""Official OpenAI CLIP ViT-L/14 as a frozen frame/text encoder."""

from __future__ import annotations

from PIL import Image

from shawaf_vlm.models.frame_encoders import _FrameEncoder
from shawaf_vlm.models.runtime import l2_normalize_torch, resolve_device, unwrap_features

OPENAI_CLIP_MODEL = "ViT-L/14"


class OpenAIClipEncoder(_FrameEncoder):
    """OpenAI's public ViT-L/14 checkpoint with the official preprocessing."""

    name = "openai_clip_vit_l14"
    checkpoint = OPENAI_CLIP_MODEL

    def __init__(self, device: str = "cuda") -> None:
        import clip

        super().__init__(device)
        self.device = resolve_device(device)
        self.model, self.preprocess = clip.load(
            OPENAI_CLIP_MODEL,
            device=self.device,
            jit=False,
        )
        self.model.eval()

    def _encode_images(self, images: list[Image.Image]):
        import torch

        pixel_values = torch.stack([self.preprocess(image) for image in images])
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(pixel_values)
        return l2_normalize_torch(unwrap_features(features))

    def _encode_text_batch(self, texts: list[str]):
        import clip
        import torch

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            return unwrap_features(self.model.encode_text(tokens))
