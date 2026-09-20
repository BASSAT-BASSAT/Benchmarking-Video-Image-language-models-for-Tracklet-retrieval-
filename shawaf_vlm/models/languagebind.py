from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from shawaf_vlm.models.runtime import (
    clip_preprocess_bcthw,
    l2_normalize_torch,
    resolve_device,
    to_numpy,
    unwrap_features,
)


class LanguageBindEncoder:
    """Frozen LanguageBind video encoder (LanguageBind/LanguageBind_Video)."""

    name = "languagebind"
    checkpoint = "LanguageBind/LanguageBind_Video"

    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = resolve_device(device)
        self.model = AutoModel.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model.eval()
        self.model.to(self.device)
        vision = getattr(self.model.config, "vision_config", None)
        self.image_size = int(getattr(vision, "image_size", 224) or 224)

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 2,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(videos), batch_size),
            desc=f"{self.name} videos",
            unit="batch",
        ):
            batch = videos[start : start + batch_size]
            pixel_values = clip_preprocess_bcthw(
                batch,
                image_size=self.image_size,
            ).to(self.device)
            features = self._forward_videos(pixel_values)
            chunks.append(to_numpy(features))
        return np.concatenate(chunks, axis=0)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"{self.name} texts",
            unit="batch",
        ):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if hasattr(value, "to")
            }
            features = self._forward_texts(encoded)
            chunks.append(to_numpy(features))
        return np.concatenate(chunks, axis=0)

    def _forward_videos(self, pixel_values: object) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self._video_features(pixel_values)
            else:
                features = self._video_features(pixel_values)
        return l2_normalize_torch(unwrap_features(features))

    def _forward_texts(self, inputs: dict[str, object]) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self._text_features(inputs)
            else:
                features = self._text_features(inputs)
        return l2_normalize_torch(unwrap_features(features))

    def _video_features(self, pixel_values: object) -> object:
        if not hasattr(self.model, "get_image_features"):
            outputs = self.model(pixel_values=pixel_values)
            if hasattr(outputs, "image_embeds"):
                return outputs.image_embeds
            raise RuntimeError(
                "LanguageBind checkpoint does not expose get_image_features "
                "or image_embeds. Upgrade transformers or install languagebind."
            )
        try:
            return self.model.get_image_features(pixel_values=pixel_values)
        except (RuntimeError, ValueError, TypeError):
            import torch

            if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
                raise
            batch, channels, frames, height, width = pixel_values.shape
            flat = pixel_values.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            )
            features = self.model.get_image_features(pixel_values=flat)
            return features.view(batch, frames, -1).mean(dim=1)

    def _text_features(self, inputs: dict[str, object]) -> object:
        if hasattr(self.model, "get_text_features"):
            return self.model.get_text_features(**inputs)
        outputs = self.model(**inputs)
        if hasattr(outputs, "text_embeds"):
            return outputs.text_embeds
        raise RuntimeError(
            "LanguageBind checkpoint does not expose get_text_features "
            "or text_embeds. Upgrade transformers or install languagebind."
        )
