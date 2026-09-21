from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(device: str) -> str:
    requested = device
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return device
        except ImportError:
            print(
                f"WARNING: {requested} requested but torch is missing; using cpu",
                flush=True,
            )
            return "cpu"
        print(
            f"WARNING: {requested} requested but CUDA is not available; using cpu",
            flush=True,
        )
        return "cpu"
    return device


def describe_device(device: str) -> str:
    if not str(device).startswith("cuda"):
        return str(device)
    try:
        import torch

        index = 0
        if ":" in str(device):
            index = int(str(device).split(":")[-1])
        name = torch.cuda.get_device_name(index)
        return f"{device} ({name})"
    except Exception:
        return str(device)


def place_model(model: object, device: str) -> object:
    """Move a frozen encoder to GPU when CUDA is available and print where it landed."""

    import torch

    model.eval()
    model.to(device)
    param = next(model.parameters())
    print(
        f"Model device: {param.device} dtype={param.dtype} "
        f"requested={describe_device(device)}",
        flush=True,
    )
    if str(device).startswith("cuda") and param.device.type != "cuda":
        raise RuntimeError(
            f"Failed to place model on GPU; weights are on {param.device}"
        )
    return model


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
    import torch.nn.functional as F

    tensor = unwrap_features(features)
    return F.normalize(tensor.float(), dim=-1)


def to_numpy(features: "object") -> np.ndarray:
    import torch

    tensor = unwrap_features(features)
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def unwrap_features(output: object) -> "object":
    """Turn CLIP/X-CLIP ModelOutput objects into a 2-D embedding tensor."""

    if isinstance(output, np.ndarray):
        return output
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return output
    except ImportError:
        pass
    if isinstance(output, (tuple, list)):
        if not output:
            raise TypeError("Encoder returned an empty tuple.")
        return unwrap_features(output[0])
    if isinstance(output, dict):
        for key in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            if output.get(key) is not None:
                return unwrap_features(output[key])
        raise TypeError(f"Could not unwrap encoder dict keys {list(output)}")
    # Newer transformers X-CLIP get_video_features returns BaseModelOutputWithPooling.
    for key in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, key, None)
        if value is not None:
            return unwrap_features(value)
    last = getattr(output, "last_hidden_state", None)
    if last is not None:
        tokens = unwrap_features(last)
        if getattr(tokens, "ndim", 0) >= 2:
            return tokens[:, 0]
        return tokens
    raise TypeError(
        f"Could not unwrap encoder output of type {type(output)!r}"
    )


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


def imagenet_preprocess_btchw(
    paths_batch: list[list[Path]],
    image_size: int = 224,
) -> "object":
    """VideoMAE / X-CLIP resize/center-crop/normalize to (B, T, C, H, W)."""

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
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
                cropped = TF.center_crop(resized, [image_size, image_size])
                tensor = TF.to_tensor(cropped)
                tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
            frames.append(tensor)
        videos.append(torch.stack(frames, dim=0))
    return torch.stack(videos, dim=0)
