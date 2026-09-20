from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def resolve_device(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return device
        except ImportError:
            return "cpu"
        return "cpu"
    return device


def load_pil_frames(paths: list[Path]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    if not frames:
        raise FileNotFoundError("No frames could be loaded for a tracklet.")
    return frames


def frames_to_uint8_tchw(paths: list[Path]) -> "object":
    """Stack RGB crops as a uint8 tensor of shape (T, C, H, W)."""

    import torch

    tensors = []
    for path in paths:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return torch.stack(tensors, dim=0)


def l2_normalize_torch(features: "object") -> "object":
    import torch
    import torch.nn.functional as F

    return F.normalize(features.float(), dim=-1)


def to_numpy(features: "object") -> np.ndarray:
    import torch

    if isinstance(features, torch.Tensor):
        return features.detach().cpu().float().numpy()
    return np.asarray(features, dtype=np.float32)


def unwrap_features(output: object) -> "object":
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def clip_preprocess_bcthw(
    paths_batch: list[list[Path]],
    image_size: int = 224,
) -> "object":
    """CLIP-style resize/center-crop/normalize to (B, C, T, H, W)."""

    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    videos = []
    for paths in paths_batch:
        frames = []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                resized = TF.resize(
                    rgb,
                    image_size,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                cropped = TF.center_crop(resized, [image_size, image_size])
                tensor = TF.to_tensor(cropped)
                tensor = TF.normalize(tensor, CLIP_MEAN, CLIP_STD)
            frames.append(tensor)
        videos.append(torch.stack(frames, dim=1))
    return torch.stack(videos, dim=0)
