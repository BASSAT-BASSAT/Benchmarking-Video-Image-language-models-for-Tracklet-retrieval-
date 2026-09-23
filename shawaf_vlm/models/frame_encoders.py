"""Frame-level text encoders scored with the same clip protocol as the video models.

Each clip is a list of frames. Frame embeddings are L2-normalized, averaged,
then L2-normalized again. Sliding-window pooling stays outside this class.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from shawaf_vlm.models.runtime import (
    CLIP_MEAN,
    CLIP_STD,
    l2_normalize_torch,
    load_pretrained,
    place_model,
    resolve_device,
    to_numpy,
    unwrap_features,
)

SIGLIP2 = "google/siglip2-so400m-patch14-384"
PE_CORE_L14 = "hf-hub:timm/PE-Core-L-14-336"
IRRA_DRIVE_ID = "1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW"
IRRA_IMAGE_SIZE = (384, 128)
_FRAME_MICROBATCH = 4


def mean_pool_clips(frame_features, lengths: list[int]):
    """Average L2-normalized frame embeddings inside each clip, then L2 again."""

    import torch

    if frame_features.shape[0] != sum(lengths):
        raise ValueError(
            f"Got {frame_features.shape[0]} frame embeddings for lengths {lengths}."
        )
    pooled = []
    offset = 0
    for length in lengths:
        if length < 1:
            raise ValueError("A clip has no frames.")
        clip = frame_features[offset : offset + length].mean(dim=0)
        pooled.append(clip)
        offset += length
    return l2_normalize_torch(torch.stack(pooled, dim=0))


def _move_inputs(inputs: dict, device: str, dtype) -> dict:
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            continue
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device)
    return moved


class _FrameEncoder:
    name = "frame"
    checkpoint = ""

    def __init__(self, device: str = "cuda") -> None:
        self.device = resolve_device(device)

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 1,
    ) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        step = max(int(batch_size), 1)
        for start in tqdm(
            range(0, len(videos), step),
            desc=f"{self.name} videos",
            unit="batch",
        ):
            batch = videos[start : start + step]
            images: list[Image.Image] = []
            lengths: list[int] = []
            for clip in batch:
                frames = _load_frames(clip)
                images.extend(frames)
                lengths.append(len(frames))
            frame_features = self._encode_frame_list(images)
            pooled = mean_pool_clips(frame_features, lengths)
            chunks.append(to_numpy(pooled))
            del frame_features, pooled
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return np.concatenate(chunks, axis=0)

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        chunks: list[np.ndarray] = []
        step = max(int(batch_size), 1)
        for start in tqdm(
            range(0, len(texts), step),
            desc=f"{self.name} texts",
            unit="batch",
        ):
            features = self._encode_text_batch(texts[start : start + step])
            chunks.append(to_numpy(l2_normalize_torch(features)))
        return np.concatenate(chunks, axis=0)

    def _encode_frame_list(self, images: list[Image.Image]):
        import torch

        parts = []
        for start in range(0, len(images), _FRAME_MICROBATCH):
            parts.append(self._encode_images(images[start : start + _FRAME_MICROBATCH]))
        return torch.cat(parts, dim=0)

    def _encode_images(self, images: list[Image.Image]):
        raise NotImplementedError

    def _encode_text_batch(self, texts: list[str]):
        raise NotImplementedError


def _load_frames(paths: list[Path]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    if not frames:
        raise FileNotFoundError("No frames could be loaded for a tracklet.")
    return frames


class Siglip2Encoder(_FrameEncoder):
    """google/siglip2-so400m-patch14-384.

    That Hub config says ``model_type=siglip`` and ships ``SiglipProcessor``.
    AutoModel follows the config, so the fixed-resolution checkpoint is not
    forced into the NaFlex ``Siglip2Model`` class.
    """

    name = "siglip2"
    checkpoint = SIGLIP2

    def __init__(self, device: str = "cuda", checkpoint: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        super().__init__(device)
        self.checkpoint = checkpoint or SIGLIP2
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        self.model = load_pretrained(
            AutoModel.from_pretrained,
            self.checkpoint,
            dtype,
        )
        place_model(self.model, self.device, dtype=dtype)
        text_config = self.model.config.text_config
        length = getattr(text_config, "max_position_embeddings", 64)
        # Gemma tokenizer configs publish an unset max length. SigLIP trains at 64.
        if length is None or int(length) > 512:
            length = 64
        self.text_length = int(length)

    def _encode_images(self, images: list[Image.Image]):
        import torch

        inputs = self.processor(images=images, return_tensors="pt")
        dtype = next(self.model.parameters()).dtype
        inputs = _move_inputs(inputs, self.device, dtype)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model.get_image_features(**inputs)
            else:
                output = self.model.get_image_features(**inputs)
        return l2_normalize_torch(unwrap_features(output))

    def _encode_text_batch(self, texts: list[str]):
        import torch

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_length,
        )
        dtype = next(self.model.parameters()).dtype
        inputs = _move_inputs(inputs, self.device, dtype)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model.get_text_features(**inputs)
            else:
                output = self.model.get_text_features(**inputs)
        return unwrap_features(output)


class PerceptionEncoder(_FrameEncoder):
    """Meta PE-Core L/14 at 336px via the open_clip Hub checkpoint.

    Text context is 32 tokens. Longer TVPReid captions are truncated by the
    official tokenizer.
    """

    name = "pe_core_l14"
    checkpoint = PE_CORE_L14

    def __init__(self, device: str = "cuda") -> None:
        import open_clip
        import torch

        super().__init__(device)
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            self.checkpoint
        )
        self.tokenizer = open_clip.get_tokenizer(self.checkpoint)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        place_model(self.model, self.device, dtype=dtype)

    def _encode_images(self, images: list[Image.Image]):
        import torch

        pixel_values = torch.stack([self.preprocess(image) for image in images])
        pixel_values = pixel_values.to(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_image(pixel_values)
            else:
                features = self.model.encode_image(pixel_values)
        return l2_normalize_torch(unwrap_features(features))

    def _encode_text_batch(self, texts: list[str]):
        import torch

        tokens = self.tokenizer(texts)
        if not hasattr(tokens, "to"):
            raise TypeError(f"PE tokenizer returned {type(tokens).__name__}.")
        tokens = tokens.to(self.device)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_text(tokens)
            else:
                features = self.model.encode_text(tokens)
        return unwrap_features(features)


def irra_image_size(positional_embedding_rows: int) -> tuple[int, int]:
    """Map an IRRA vision position table back to the training resolution."""

    patches = positional_embedding_rows - 1
    if patches == (384 // 16) * (128 // 16):
        return IRRA_IMAGE_SIZE
    if patches == (224 // 16) * (224 // 16):
        return (224, 224)
    raise ValueError(
        "IRRA positional embedding has "
        f"{positional_embedding_rows} rows, which is not the 384x128 or 224 grid."
    )


def _irra_cache_dir() -> Path:
    path = Path.home() / ".cache" / "shawaf" / "irra"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_irra_checkpoint() -> Path:
    """Download the official CUHK-PEDES IRRA checkpoint from the paper's Drive link."""

    cache = _irra_cache_dir()
    marker = cache / "best.pth"
    if marker.is_file() and marker.stat().st_size > 1_000_000:
        return marker
    archive = cache / "irra_cuhk.zip"
    if not archive.is_file() or archive.stat().st_size < 1_000_000:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "gdown is required to download IRRA weights. "
                "pip install gdown, then rerun."
            ) from exc
        gdown.download(id=IRRA_DRIVE_ID, output=str(archive), quiet=False)
    extracted = _extract_checkpoint(archive, cache)
    if extracted != marker:
        marker.write_bytes(extracted.read_bytes())
    return marker


def _extract_checkpoint(archive: Path, dest: Path) -> Path:
    payload = archive.read_bytes()
    members: list[tuple[str, bytes]] = []
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                members.append((info.filename, handle.read(info)))
    elif payload[:2] == b"\x1f\x8b" or payload[257:262] == b"ustar":
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as handle:
            for info in handle.getmembers():
                if not info.isfile():
                    continue
                extracted = handle.extractfile(info)
                if extracted is None:
                    continue
                members.append((info.name, extracted.read()))
    else:
        return archive

    weight_names = [
        name
        for name, _ in members
        if name.endswith(".pth") or name.endswith(".pt")
    ]
    if not weight_names:
        raise RuntimeError(f"{archive} did not contain a .pth checkpoint.")
    preferred = sorted(
        weight_names,
        key=lambda name: (0 if "best" in Path(name).name.lower() else 1, len(name)),
    )[0]
    blob = dict(members)[preferred]
    out = dest / "extracted" / Path(preferred).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return out


def load_irra_weights(model, checkpoint: Path) -> tuple[int, int]:
    """Copy IRRA's CLIP tower into an OpenAI ViT-B/16 and return the vision grid."""

    import torch
    from torch import nn

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload
    if isinstance(state, dict):
        for key in ("model", "state_dict", "module"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
    if not isinstance(state, dict):
        raise TypeError(f"IRRA checkpoint is {type(payload).__name__}, expected a state dict.")

    cleaned = {}
    for key, value in state.items():
        name = str(key)
        while name.startswith("module."):
            name = name[len("module.") :]
        while name.startswith("base_model."):
            name = name[len("base_model.") :]
        if not hasattr(value, "shape"):
            continue
        cleaned[name] = value

    pos_key = "visual.positional_embedding"
    if pos_key not in cleaned:
        sample = list(cleaned)[:8]
        raise RuntimeError(
            "IRRA checkpoint has no visual.positional_embedding. "
            f"First keys: {sample}"
        )
    height, width = irra_image_size(int(cleaned[pos_key].shape[0]))
    visual = model.visual
    pos = cleaned[pos_key].detach().to(
        device=visual.positional_embedding.device,
        dtype=visual.positional_embedding.dtype,
    )
    if tuple(visual.positional_embedding.shape) != tuple(pos.shape):
        visual.positional_embedding = nn.Parameter(pos.clone())
    visual.image_size = (height, width)
    current = model.state_dict()
    aligned = {}
    for name, value in cleaned.items():
        target = current.get(name)
        if target is None:
            aligned[name] = value
            continue
        if tuple(value.shape) != tuple(target.shape):
            raise RuntimeError(
                f"IRRA tensor {name} has shape {tuple(value.shape)}, "
                f"ViT-B/16 expects {tuple(target.shape)}."
            )
        aligned[name] = value.detach().to(dtype=target.dtype)
    message = model.load_state_dict(aligned, strict=False)
    required = (
        "visual.conv1.weight",
        "visual.class_embedding",
        "visual.positional_embedding",
        "visual.proj",
        "visual.ln_pre.weight",
        "visual.ln_post.weight",
        "token_embedding.weight",
        "positional_embedding",
        "ln_final.weight",
        "text_projection",
    )
    missing = set(message.missing_keys)
    absent = [key for key in required if key in missing]
    transformer_missing = [
        key for key in message.missing_keys if key.startswith("visual.transformer.resblocks.0.")
    ]
    if absent or transformer_missing:
        raise RuntimeError(
            "IRRA weights did not match ViT-B/16. "
            f"Missing: {(absent + transformer_missing)[:8]}"
        )
    return height, width


class IrraEncoder(_FrameEncoder):
    """IRRA (CVPR 2023) CLIP ViT-B/16 trained on CUHK-PEDES person descriptions.

    Official training resizes person images to 384x128. Tracklet frames use that
    same resize, then the clip is mean-pooled.
    """

    name = "irra"
    checkpoint = "IRRA CUHK-PEDES ViT-B/16"

    def __init__(self, device: str = "cuda", checkpoint: str | Path | None = None) -> None:
        import open_clip
        import torch

        super().__init__(device)
        self.model = open_clip.create_model(
            "ViT-B-16",
            pretrained="openai",
            force_quick_gelu=True,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        weight_path = Path(checkpoint) if checkpoint else download_irra_checkpoint()
        height, width = load_irra_weights(self.model, weight_path)
        self.image_size = (height, width)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        place_model(self.model, self.device, dtype=dtype)

    def _preprocess(self, image: Image.Image):
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        resized = TF.resize(
            image,
            list(self.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        tensor = TF.to_tensor(resized)
        return TF.normalize(tensor, CLIP_MEAN, CLIP_STD)

    def _encode_images(self, images: list[Image.Image]):
        import torch

        pixel_values = torch.stack([self._preprocess(image) for image in images])
        pixel_values = pixel_values.to(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_image(pixel_values)
            else:
                features = self.model.encode_image(pixel_values)
        return l2_normalize_torch(unwrap_features(features))

    def _encode_text_batch(self, texts: list[str]):
        import torch

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_text(tokens)
            else:
                features = self.model.encode_text(tokens)
        return unwrap_features(features)
