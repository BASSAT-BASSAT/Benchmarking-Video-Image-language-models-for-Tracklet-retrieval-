from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from shawaf_vlm.models.runtime import (
    imagenet_preprocess_btchw,
    l2_normalize_torch,
    place_model,
    resolve_device,
    to_numpy,
)


class XCLIPEncoder:
    """Frozen microsoft/xclip-base-patch32 video-text encoder."""

    name = "xclip"
    checkpoint = "microsoft/xclip-base-patch32"

    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoModel, AutoProcessor
        import torch

        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.checkpoint,
                use_fast=False,
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        try:
            self.model = AutoModel.from_pretrained(
                self.checkpoint,
                torch_dtype=dtype,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(self.checkpoint)
        place_model(self.model, self.device)

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 4,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(videos), batch_size),
            desc=f"{self.name} videos",
            unit="batch",
        ):
            batch = videos[start : start + batch_size]
            # New transformers XCLIPProcessor maps videos through a processor
            # that no longer exposes pixel_values. Match VideoMAE ImageNet
            # stats ourselves: (B, T, C, H, W).
            pixel_values = imagenet_preprocess_btchw(batch, image_size=224).to(
                self.device
            )
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
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
            features = self._forward_texts(inputs)
            chunks.append(to_numpy(features))
        return np.concatenate(chunks, axis=0)

    def _forward_videos(self, pixel_values: object) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.get_video_features(pixel_values=pixel_values)
            else:
                features = self.model.get_video_features(pixel_values=pixel_values)
        return l2_normalize_torch(features)

    def _forward_texts(self, inputs: dict[str, object]) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.get_text_features(**inputs)
            else:
                features = self.model.get_text_features(**inputs)
        return l2_normalize_torch(features)
