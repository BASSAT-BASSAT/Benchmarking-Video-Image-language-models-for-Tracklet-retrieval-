"""GME-Qwen2-VL-2B as a frozen frame/text encoder."""

from __future__ import annotations

from PIL import Image

from shawaf_vlm.models.frame_encoders import _FrameEncoder
from shawaf_vlm.models.runtime import l2_normalize_torch, unwrap_features

GME_QWEN2_VL_2B = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"


class GmeQwen2VLEncoder(_FrameEncoder):
    """Official GME Sentence Transformers path for Transformers 4.52+."""

    name = "gme_qwen2_vl_2b"
    checkpoint = GME_QWEN2_VL_2B
    frame_microbatch = 1

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        super().__init__(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = SentenceTransformer(
            self.checkpoint,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": dtype},
        )
        self.model.eval()

    def _encode_images(self, images: list[Image.Image]):
        items = [{"image": image} for image in images]
        features = self.model.encode(
            items,
            batch_size=1,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return l2_normalize_torch(unwrap_features(features))

    def _encode_text_batch(self, texts: list[str]):
        # The official ST route uses GME's neutral default instruction.
        features = self.model.encode(
            texts,
            batch_size=max(len(texts), 1),
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return unwrap_features(features)
