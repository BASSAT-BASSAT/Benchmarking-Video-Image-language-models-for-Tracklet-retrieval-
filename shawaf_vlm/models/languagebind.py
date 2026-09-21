from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from shawaf_vlm.models.runtime import (
    clip_preprocess_bcthw,
    l2_normalize_torch,
    place_model,
    resolve_device,
    to_numpy,
    unwrap_features,
)


def _read_checkpoint_tensors(checkpoint: str) -> tuple[dict, str]:
    from transformers.utils import cached_file

    try:
        path = cached_file(checkpoint, "model.safetensors")
    except OSError:
        path = None
    if path:
        from safetensors.torch import load_file

        return load_file(path), "model.safetensors"

    path = cached_file(checkpoint, "pytorch_model.bin")
    import torch

    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    return raw, "pytorch_model.bin"


def _load_languagebind_weights(model, checkpoint: str) -> None:
    from shawaf_vlm.models.languagebind_hf.compat import remap_peft_state_dict

    raw, source = _read_checkpoint_tensors(checkpoint)
    remapped = remap_peft_state_dict(raw, model)
    missing, _unexpected = model.load_state_dict(remapped, strict=False)
    missing = [key for key in missing if "position_ids" not in key]
    print(
        f"LanguageBind loaded {len(remapped)}/{len(model.state_dict())} "
        f"tensors from {source}",
        flush=True,
    )
    if missing:
        preview = ", ".join(missing[:8])
        print(
            f"LanguageBind still missing {len(missing)} tensors: {preview}",
            flush=True,
        )


class LanguageBindEncoder:
    """Frozen LanguageBind video encoder (LanguageBind/LanguageBind_Video)."""

    name = "languagebind"
    checkpoint = "LanguageBind/LanguageBind_Video"
    context_length = 77

    def __init__(self, device: str = "cuda") -> None:
        from transformers import CLIPTokenizer
        import torch

        from shawaf_vlm.models.languagebind_hf.compat import disable_incompatible_torchao

        disable_incompatible_torchao()
        from shawaf_vlm.models.languagebind_hf.configuration_video import (
            LanguageBindVideoConfig,
        )
        from shawaf_vlm.models.languagebind_hf.modeling_video import LanguageBindVideo

        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        config = LanguageBindVideoConfig.from_pretrained(self.checkpoint)
        # Build, wrap LoRA, then remap old PEFT key names onto current peft.
        model = LanguageBindVideo(config)
        _load_languagebind_weights(model, self.checkpoint)
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
        self.model = model
        self.tokenizer = CLIPTokenizer.from_pretrained(self.checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        place_model(self.model, self.device)
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
                padding="max_length",
                truncation=True,
                max_length=self.context_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"][:, : self.context_length]
            attention_mask = encoded["attention_mask"][:, : self.context_length]
            vocab = self.model.text_model.embeddings.token_embedding.num_embeddings
            input_ids = input_ids.clamp(0, vocab - 1)
            features = self._forward_texts(
                {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }
            )
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
