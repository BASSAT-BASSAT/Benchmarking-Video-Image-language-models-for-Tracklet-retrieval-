from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from shawaf_vlm.models.runtime import (
    frames_to_uint8_tchw,
    l2_normalize_torch,
    load_pretrained,
    place_model,
    resolve_device,
    to_numpy,
    unwrap_features,
)

CLIP_S = "OpenGVLab/InternVideo2_CLIP_S"
CLIP_1B = "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"

_ONE_B_HELP = (
    f"{CLIP_1B} is a gated LoRA add-on, not a full Hugging Face AutoModel. "
    f"Use --model internvideo2 ({CLIP_S}) for the Stage 1 InternVideo2 baseline, "
    "or load the official InternVideo2 CLIP-1B stack from OpenGVLab/InternVideo "
    "if you specifically need the 1B checkpoint."
)


class InternVideo2Encoder:
    """Frozen InternVideo2 CLIP encoder via Hugging Face AutoModel."""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str = CLIP_S,
        name: str = "internvideo2",
    ) -> None:
        from transformers import AutoModel
        import torch

        self.name = name
        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        try:
            self.model = load_pretrained(
                AutoModel.from_pretrained,
                checkpoint,
                dtype,
                trust_remote_code=True,
            )
        except Exception as exc:
            if checkpoint == CLIP_1B:
                raise RuntimeError(_ONE_B_HELP) from exc
            raise
        if not hasattr(self.model, "encode_vision") or not hasattr(
            self.model, "encode_text"
        ):
            if checkpoint == CLIP_1B:
                raise RuntimeError(_ONE_B_HELP)
            raise RuntimeError(
                f"{checkpoint} loaded but has no encode_vision/encode_text API."
            )
        place_model(self.model, self.device)
        if hasattr(self.model, "device"):
            try:
                self.model.device = self.device
            except Exception:
                pass

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 2,
    ) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(videos), batch_size),
            desc=f"{self.name} videos",
            unit="batch",
        ):
            batch = videos[start : start + batch_size]
            transformed = []
            for paths in batch:
                frames = frames_to_uint8_tchw(paths)
                video = self.model.transform(frames)
                transformed.append(video)

            pixel_values = torch.stack(transformed, dim=0).to(self.device)
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
            text_input = self._tokenize(batch)
            features = self._forward_texts(text_input)
            chunks.append(to_numpy(features))
        return np.concatenate(chunks, axis=0)

    def _tokenize(self, texts: list[str]) -> object:
        encoded = self.model.tokenizer(texts)
        if hasattr(encoded, "to"):
            return encoded.to(self.device)
        if isinstance(encoded, dict):
            return {
                key: value.to(self.device)
                for key, value in encoded.items()
                if hasattr(value, "to")
            }
        return encoded

    def _forward_videos(self, pixel_values: object) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_vision(pixel_values, test=True)
            else:
                features = self.model.encode_vision(pixel_values, test=True)
        return l2_normalize_torch(unwrap_features(features))

    def _forward_texts(self, text_input: object) -> object:
        import torch

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_text(text_input)
            else:
                features = self.model.encode_text(text_input)
        return l2_normalize_torch(unwrap_features(features))


def build_internvideo2_clip_s(device: str = "cuda") -> InternVideo2Encoder:
    return InternVideo2Encoder(
        device=device,
        checkpoint=CLIP_S,
        name="internvideo2",
    )


def build_internvideo2_clip_1b(device: str = "cuda") -> InternVideo2Encoder:
    return InternVideo2Encoder(
        device=device,
        checkpoint=CLIP_1B,
        name="internvideo2_clip_1b",
    )
