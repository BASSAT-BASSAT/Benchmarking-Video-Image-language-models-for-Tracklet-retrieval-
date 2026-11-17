"""Frame-level text encoders scored with the same clip protocol as the video models.

Each clip is a list of frames. Frame embeddings are L2-normalized, averaged,
then L2-normalized again. Sliding-window pooling stays outside this class.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import zipfile
from contextlib import nullcontext
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


class _Siglip2ProcessorShim:
    """Image processor + Gemma tokenizer (SigLIP 2 Hub checkpoints)."""

    def __init__(self, image_processor, tokenizer) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        images=None,
        text=None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        if images is not None:
            return self.image_processor(images=images, return_tensors=return_tensors)
        if text is not None:
            return self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=kwargs.get("padding", "max_length"),
                truncation=kwargs.get("truncation", True),
                max_length=kwargs.get("max_length"),
            )
        raise ValueError("Pass images= or text= to the SigLIP 2 processor shim.")


def load_siglip2_processor(checkpoint: str = SIGLIP2):
    """Load processor; avoid AutoProcessor picking legacy SiglipTokenizer on TF 4.x."""

    from transformers import AutoImageProcessor, AutoTokenizer

    try:
        from transformers import Siglip2Processor

        return Siglip2Processor.from_pretrained(checkpoint)
    except (ImportError, OSError, ValueError):
        pass

    try:
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        return _Siglip2ProcessorShim(image_processor, tokenizer)
    except (OSError, ValueError, TypeError):
        pass

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(checkpoint)


PE_CORE_L14 = "hf-hub:timm/PE-Core-L-14-336"
PE_CORE_B16 = "hf-hub:timm/PE-Core-B-16"
PE_CORE_S16 = "hf-hub:timm/PE-Core-S-16-384"
PE_CORE_T16 = "hf-hub:timm/PE-Core-T-16-384"
# Apple MobileCLIP2 weights are released under the apple-amlr licence.
MOBILECLIP2_S2 = "hf-hub:timm/MobileCLIP2-S2-OpenCLIP"
MOBILECLIP2_S4 = "hf-hub:timm/MobileCLIP2-S4-OpenCLIP"
MOBILECLIP2_B = "hf-hub:timm/MobileCLIP2-B-OpenCLIP"
EVA02_CLIP_L14_336 = "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
DFN2B_CLIP_L14 = "hf-hub:apple/DFN2B-CLIP-ViT-L-14"
IRRA_DRIVE_ID = "1OBhFhpZpltRMZ88K6ceNUv4vZgevsFCW"
IRRA_IMAGE_SIZE = (384, 128)
_FRAME_MICROBATCH = 64


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


def path_key(path: Path | str) -> str:
    return str(Path(path))


def windows_from_frame_cache(windows: list[list[Path]], index: dict[str, int], feats):
    """L2-normalized mean of cached frame rows, one vector per window.

    ``feats`` rows must already be the per-frame embeddings ``_encode_images``
    returns. This matches ``mean_pool_clips`` on the same rows.
    """

    import torch

    rows = []
    for window in windows:
        if not window:
            raise ValueError("A clip has no frames.")
        ids = [index[path_key(path)] for path in window]
        rows.append(feats[ids].mean(dim=0))
    stacked = torch.stack(rows, dim=0)
    return to_numpy(l2_normalize_torch(stacked))


def preprocess_at_size(image_size: int):
    """RGB preprocess that yields ``(3, image_size, image_size)``."""

    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    if image_size < 1:
        raise ValueError("image_size must be >= 1")

    def _apply(image: Image.Image) -> "torch.Tensor":
        resized = TF.resize(
            image,
            [image_size, image_size],
            interpolation=InterpolationMode.BICUBIC,
        )
        tensor = TF.to_tensor(resized)
        return TF.normalize(tensor, CLIP_MEAN, CLIP_STD)

    return _apply


FIT_MODES = ("native", "squash", "letterbox", "pad2")


class _Letterbox:
    """Pad a PIL image toward a square with ``fill``.

    ``max_ratio=1`` pads to a full square (keeps exact proportions, fewest
    pixels on the person). ``max_ratio=2`` (``pad2``) only pads until the long
    side is at most twice the short side; the resize to the square input then
    squashes the rest. That halves the distortion of a typical 1:2.5 person
    crop while keeping more pixels on the person than a full letterbox.
    """

    def __init__(self, fill: tuple[int, int, int], max_ratio: float = 1.0) -> None:
        self.fill = fill
        self.max_ratio = max(float(max_ratio), 1.0)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        new_w = max(width, int(round(height / self.max_ratio)))
        new_h = max(height, int(round(width / self.max_ratio)))
        if (new_w, new_h) == (width, height):
            return image
        canvas = Image.new("RGB", (new_w, new_h), self.fill)
        canvas.paste(image.convert("RGB"), ((new_w - width) // 2, (new_h - height) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"Letterbox(fill={self.fill}, max_ratio={self.max_ratio:g})"


def make_prefit(fit: str | None, fill: tuple[int, int, int] = (127, 127, 127)):
    """PIL pre-transform for encoders that keep their own processor (SigLIP 2,
    EmbeddingGemma 2, IRRA, ...): pad the crop *before* the model's resize.

    Returns ``None`` for ``native`` / ``None`` (no change). ``squash`` is not
    offered here because those processors already decide their own geometry.
    """

    if fit in (None, "native"):
        return None
    if fit == "letterbox":
        return _Letterbox(fill, max_ratio=1.0)
    if fit == "pad2":
        return _Letterbox(fill, max_ratio=2.0)
    raise ValueError(f"fit {fit!r} is not supported for this encoder; use native, letterbox or pad2")


def rebuild_preprocess(preprocess, image_size: int | None = None, fit: str = "native"):
    """Rebuild an open_clip/torchvision ``Compose`` at a new size or fit mode.

    Keeps the model's own interpolation and normalization. ``fit``:

    - ``native``     the checkpoint's own geometry (PE squashes; MobileCLIP2
                     resizes the short side then centre-crops, which cuts the
                     head and feet off tall person crops)
    - ``squash``     resize the whole crop to a square (aspect not kept)
    - ``letterbox``  pad to a square with the mean colour, then resize
    - ``pad2``       pad only to a 1:2 aspect, then resize (half-way)

    Non-``Compose`` callables (tests, custom encoders) fall back to
    ``preprocess_at_size`` when a size is given.
    """

    if fit not in FIT_MODES:
        raise ValueError(f"fit must be one of {FIT_MODES}, got {fit!r}")
    transforms = getattr(preprocess, "transforms", None)
    if transforms is None:
        if image_size is None:
            return preprocess
        return preprocess_at_size(image_size)
    if image_size is None and fit == "native":
        return preprocess

    from torchvision import transforms as T

    size = None
    interpolation = T.InterpolationMode.BICUBIC
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    tail = []
    for step in transforms:
        if isinstance(step, T.Resize):
            interpolation = step.interpolation
            value = step.size
            size = value if isinstance(value, int) else int(value[0])
        elif isinstance(step, T.CenterCrop):
            size = int(step.size[0])
        elif isinstance(step, T.Normalize):
            mean, std = tuple(step.mean), tuple(step.std)
            tail.append(step)
        else:
            tail.append(step)
    target = int(image_size or size or 224)

    if fit == "native":
        rebuilt = []
        for step in transforms:
            if isinstance(step, T.Resize):
                square = not isinstance(step.size, int)
                rebuilt.append(
                    T.Resize(
                        (target, target) if square else target,
                        interpolation=step.interpolation,
                        antialias=True,
                    )
                )
            elif isinstance(step, T.CenterCrop):
                rebuilt.append(T.CenterCrop((target, target)))
            else:
                rebuilt.append(step)
        return T.Compose(rebuilt)

    head = []
    if fit in ("letterbox", "pad2"):
        fill = tuple(int(round(255 * float(m))) for m in mean)
        head.append(_Letterbox(fill, max_ratio=1.0 if fit == "letterbox" else 2.0))
    head.append(T.Resize((target, target), interpolation=interpolation, antialias=True))
    return T.Compose(head + tail)


def interpolate_absolute_pos_embed(positional, image_size: int, patch_size: int):
    """Resize a ViT absolute position table (cls token + square grid).

    Returns ``None`` when the table is not a square grid (RoPE-style tables
    are left untouched by the caller).
    """

    import torch
    import torch.nn.functional as F

    if patch_size < 1 or image_size % patch_size != 0:
        raise ValueError(
            f"image_size {image_size} is not divisible by patch_size {patch_size}."
        )
    cls, grid = positional[:1], positional[1:]
    side = int(round(grid.shape[0] ** 0.5))
    if side * side != grid.shape[0]:
        return None
    new_side = image_size // patch_size
    if new_side == side:
        return positional
    spatial = grid.reshape(1, side, side, -1).permute(0, 3, 1, 2).float()
    spatial = F.interpolate(spatial, size=(new_side, new_side), mode="bicubic", align_corners=False)
    spatial = spatial.permute(0, 2, 3, 1).reshape(new_side * new_side, -1)
    return torch.cat([cls.float(), spatial], dim=0).to(dtype=positional.dtype)


def resize_timm_trunk(visual, image_size: int) -> bool:
    """Resize an open_clip ``TimmModel`` trunk (PE-Core Eva) to ``image_size``.

    timm Eva asserts a fixed input size and caches its RoPE table for the
    pretrained grid. This resamples the absolute position table once and
    rebuilds the (parameter-free) RoPE cache for the new grid, so the model
    runs statically at the new size. Returns ``False`` when ``visual`` is not a
    timm trunk with a patch embed.
    """

    import torch

    trunk = getattr(visual, "trunk", None)
    patch_embed = getattr(trunk, "patch_embed", None)
    if trunk is None or patch_embed is None:
        return False
    patch = patch_embed.patch_size
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)
    if image_size % patch != 0:
        raise ValueError(f"image_size {image_size} is not divisible by patch size {patch}.")
    old_grid = tuple(int(v) for v in patch_embed.grid_size)
    side = image_size // patch
    new_grid = (side, side)
    if new_grid == old_grid:
        return True

    pos = getattr(trunk, "pos_embed", None)
    if pos is not None:
        from timm.layers import resample_abs_pos_embed

        prefix = 0 if getattr(trunk, "no_embed_class", False) else int(getattr(trunk, "num_prefix_tokens", 1))
        resized = resample_abs_pos_embed(
            pos.data.float(),
            new_size=list(new_grid),
            old_size=list(old_grid),
            num_prefix_tokens=prefix,
        )
        trunk.pos_embed = torch.nn.Parameter(resized.to(dtype=pos.dtype, device=pos.device))

    rope = getattr(trunk, "rope", None)
    if rope is not None and getattr(rope, "pos_embed", None) is not None:
        old = rope.pos_embed
        rope.feat_shape = list(new_grid)
        fresh = rope._get_pos_embed_values(rope.feat_shape)
        rope.pos_embed = fresh.to(dtype=old.dtype, device=old.device)

    patch_embed.img_size = (image_size, image_size)
    patch_embed.grid_size = new_grid
    patch_embed.num_patches = side * side
    if hasattr(visual, "image_size"):
        visual.image_size = (image_size, image_size)
    return True


def _patch_size_of(visual) -> int:
    value = getattr(visual, "patch_size", None)
    if isinstance(value, tuple):
        return int(value[0])
    if value is not None:
        return int(value)
    conv = getattr(visual, "conv1", None)
    kernel = getattr(conv, "kernel_size", None)
    if isinstance(kernel, tuple):
        return int(kernel[0])
    if kernel is not None:
        return int(kernel)
    return 14


def _is_oom(exc: BaseException) -> bool:
    import torch

    oom_types = tuple(
        cls
        for cls in (
            getattr(torch.cuda, "OutOfMemoryError", None),
            getattr(torch, "OutOfMemoryError", None),
        )
        if isinstance(cls, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    return "out of memory" in str(exc).lower()


def _stage(timer, name: str):
    if timer is None:
        return nullcontext()
    return timer.stage(name)


def _collate_frames(batch):
    indices = [item[0] for item in batch]
    images = [item[1] for item in batch]
    return indices, images


class FrameDataset:
    """Load RGB frames. Preprocess stays in the parent process so it can be timed."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB")
        return index, rgb


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
    frame_microbatch = _FRAME_MICROBATCH

    def __init__(self, device: str = "cuda", frame_microbatch: int | None = None, num_workers: int | None = None) -> None:
        self.device = resolve_device(device)
        if frame_microbatch is None:
            frame_microbatch = type(self).frame_microbatch
        self.frame_microbatch = max(int(frame_microbatch), 1)
        if num_workers is None:
            num_workers = 0 if os.name == "nt" else 2
        self.num_workers = max(int(num_workers), 0)
        self.pin_memory = self.device.startswith("cuda")
        self.stage_timer = None
        self.encoded_frames = 0

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 1,
    ) -> np.ndarray:
        if not videos:
            raise ValueError("No clips to encode.")
        flat = [path for clip in videos for path in clip]
        unique, feats = self.encode_frames(flat)
        index = {path_key(path): row for row, path in enumerate(unique)}
        chunks: list[np.ndarray] = []
        step = max(int(batch_size), 1)
        with _stage(self.stage_timer, "pool"):
            for start in tqdm(
                range(0, len(videos), step),
                desc=f"{self.name} videos",
                unit="batch",
            ):
                chunks.append(windows_from_frame_cache(videos[start : start + step], index, feats))
        return np.concatenate(chunks, axis=0)

    def encode_frames(self, paths: list[Path]) -> tuple[list[Path], object]:
        """Encode each distinct path once. Returns ``(unique_paths, features)``."""

        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = path_key(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(Path(path))
        timer = self.stage_timer
        if not unique:
            import torch

            return [], torch.zeros(0, 1)
        if self._uses_loader():
            feats = self._encode_unique_loader(unique, timer)
        else:
            with _stage(timer, "load"):
                images = _load_frames(unique)
                prefit = getattr(self, "prefit", None)
                if prefit is not None:
                    images = [prefit(image) for image in images]
            with _stage(timer, "forward"):
                feats = self._encode_frame_list(images)
        if timer is not None:
            timer.frames += len(unique)
        return unique, feats

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

    def _uses_loader(self) -> bool:
        preprocess = getattr(self, "preprocess", None)
        if not callable(preprocess):
            return False
        return type(self)._encode_pixels is not _FrameEncoder._encode_pixels

    def _encode_frame_list(self, images: list[Image.Image]):
        return self._encode_chunks(images, self._encode_images)

    def _encode_chunks(self, items, encode_fn):
        import torch

        if len(items) == 0:
            raise ValueError("No frames to encode.")
        step = max(int(self.frame_microbatch), 1)
        parts = []
        start = 0
        while start < len(items):
            end = min(len(items), start + step)
            try:
                parts.append(encode_fn(items[start:end]))
            except Exception as exc:
                if not _is_oom(exc) or step == 1:
                    raise
                step = max(1, step // 2)
                self.frame_microbatch = step
                continue
            self.encoded_frames += end - start
            start = end
        return torch.cat(parts, dim=0)

    def _encode_unique_loader(self, paths: list[Path], timer):
        """Thread-pool prefetch: decode + preprocess batch i+1 while batch i runs.

        PIL decode/resize and torchvision tensor ops release the GIL, so threads
        scale on Windows where DataLoader workers cannot fork. With profiling
        on, ``load`` is the time the GPU waited for the CPU pipeline (decode
        and preprocess together); ``preprocess`` stays 0 on this path.
        """

        import torch
        from concurrent.futures import ThreadPoolExecutor

        preprocess = self.preprocess

        def prepare(path: Path):
            with Image.open(path) as image:
                return preprocess(image.convert("RGB"))

        pin = self.pin_memory and self.device.startswith("cuda") and torch.cuda.is_available()
        parts = []
        threads = loader_threads()
        with ThreadPoolExecutor(max_workers=threads) as pool:

            def submit(start: int):
                end = min(len(paths), start + max(int(self.frame_microbatch), 1))
                return start, end, [pool.submit(prepare, path) for path in paths[start:end]]

            pending = submit(0)
            while pending is not None:
                start, end, futures = pending
                with _stage(timer, "load"):
                    pixels = torch.stack([future.result() for future in futures])
                    if pin:
                        pixels = pixels.pin_memory()
                pending = submit(end) if end < len(paths) else None
                with _stage(timer, "forward"):
                    parts.append(self._forward_with_backoff(pixels))
                self.encoded_frames += end - start
        return torch.cat(parts, dim=0)

    def _forward_with_backoff(self, pixels):
        """Encode ``pixels``; on OOM split in halves (and shrink later batches)."""

        import torch

        try:
            return self._encode_pixels(pixels)
        except Exception as exc:
            if not _is_oom(exc) or pixels.shape[0] == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            half = max(1, pixels.shape[0] // 2)
            self.frame_microbatch = min(int(self.frame_microbatch), half)
            return torch.cat(
                [self._forward_with_backoff(pixels[:half]), self._forward_with_backoff(pixels[half:])],
                dim=0,
            )

    def _encode_images(self, images: list[Image.Image]):
        raise NotImplementedError

    def _encode_pixels(self, pixels):
        raise NotImplementedError

    def _encode_text_batch(self, texts: list[str]):
        raise NotImplementedError


def loader_threads(num_workers: int | None = None) -> int:
    """Threads for frame decode. ``SHAWAF_LOADER_THREADS`` overrides."""

    env = os.environ.get("SHAWAF_LOADER_THREADS")
    if env:
        return max(int(env), 1)
    if num_workers:
        return max(int(num_workers), 1)
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_frames(paths: list[Path]) -> list[Image.Image]:
    from concurrent.futures import ThreadPoolExecutor

    if len(paths) <= 1:
        return [_open_rgb(path) for path in paths]
    with ThreadPoolExecutor(max_workers=loader_threads()) as pool:
        frames: list[Image.Image] = list(pool.map(_open_rgb, paths))
    if not frames:
        raise FileNotFoundError("No frames could be loaded for a tracklet.")
    return frames


class Siglip2Encoder(_FrameEncoder):
    """google/siglip2-so400m-patch14-384 (Gemma tokenizer + SigLIP image tower)."""

    name = "siglip2"
    checkpoint = SIGLIP2

    def __init__(self, device: str = "cuda", checkpoint: str | None = None) -> None:
        import torch
        from transformers import AutoModel

        super().__init__(device)
        self.checkpoint = checkpoint or SIGLIP2
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = load_siglip2_processor(self.checkpoint)
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
    official tokenizer. ``image_size`` rebuilds the preprocess and interpolates
    absolute position embeddings. RoPE tables are left as they are.
    """

    name = "pe_core_l14"
    checkpoint = PE_CORE_L14

    def __init__(
        self,
        device: str = "cuda",
        image_size: int | None = None,
        checkpoint: str | None = None,
        frame_microbatch: int | None = None,
        num_workers: int | None = None,
        fit: str | None = None,
    ) -> None:
        import open_clip
        import torch

        super().__init__(
            device,
            frame_microbatch=frame_microbatch,
            num_workers=num_workers,
        )
        if checkpoint:
            self.checkpoint = checkpoint
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            self.checkpoint
        )
        self.tokenizer = open_clip.get_tokenizer(self.checkpoint)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        place_model(self.model, self.device, dtype=dtype)
        self._pos_embed_calls = 0
        self.image_size = image_size
        self.fit = fit or os.environ.get("SHAWAF_FIT", "native")
        self._resize_pos_embed(image_size)
        self.preprocess = rebuild_preprocess(self.preprocess, image_size, self.fit)
        self._maybe_compile()

    def _resize_pos_embed(self, image_size: int | None) -> None:
        self._pos_embed_calls += 1
        if image_size is None:
            return
        visual = getattr(self.model, "visual", None)
        if visual is not None and resize_timm_trunk(visual, image_size):
            return
        positional = getattr(visual, "positional_embedding", None) if visual is not None else None
        if positional is None:
            return
        resized = interpolate_absolute_pos_embed(
            positional.data,
            image_size=image_size,
            patch_size=_patch_size_of(visual),
        )
        if resized is None:
            return
        import torch

        visual.positional_embedding = torch.nn.Parameter(resized.to(dtype=positional.dtype))

    def _maybe_compile(self) -> None:
        if os.environ.get("SHAWAF_COMPILE", "1") == "0":
            return
        if "pytest" in sys.modules:
            return
        if not self.device.startswith("cuda"):
            return
        import importlib.util

        if importlib.util.find_spec("triton") is None:
            # Windows wheels ship without Triton; compile would only fail at
            # the first forward pass. Install ``triton-windows`` to enable.
            print("torch.compile skipped: triton not installed", flush=True)
            return
        import torch

        self._eager_encode_image = self.model.encode_image
        try:
            self.model.encode_image = torch.compile(self.model.encode_image, dynamic=False)
            self._compile_pending = True
        except Exception as exc:
            print(f"torch.compile skipped: {exc}", flush=True)
            self.model.encode_image = self._eager_encode_image

    def _run_encode_image(self, pixel_values):
        """First compiled call is checked; on failure revert to eager for good."""

        if not getattr(self, "_compile_pending", False):
            return self.model.encode_image(pixel_values)
        try:
            out = self.model.encode_image(pixel_values)
        except Exception as exc:
            if _is_oom(exc):
                raise
            print(f"torch.compile failed at first forward, using eager: {exc}", flush=True)
            self.model.encode_image = self._eager_encode_image
            out = self.model.encode_image(pixel_values)
        self._compile_pending = False
        return out

    def _encode_images(self, images: list[Image.Image]):
        import torch

        pixel_values = torch.stack([self.preprocess(image) for image in images])
        return self._encode_pixels(pixel_values)

    def _encode_pixels(self, pixel_values):
        import torch

        pixel_values = pixel_values.to(device=self.device, non_blocking=True).to(
            dtype=next(self.model.parameters()).dtype,
        )
        with torch.inference_mode():
            features = self._run_encode_image(pixel_values)
        return l2_normalize_torch(unwrap_features(features))

    def _encode_text_batch(self, texts: list[str]):
        import torch

        tokens = self.tokenizer(texts)
        if not hasattr(tokens, "to"):
            raise TypeError(f"PE tokenizer returned {type(tokens).__name__}.")
        tokens = tokens.to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return unwrap_features(features)


class PeCoreB16Encoder(PerceptionEncoder):
    name = "pe_core_b16"
    checkpoint = PE_CORE_B16


class PeCoreS16Encoder(PerceptionEncoder):
    name = "pe_core_s16"
    checkpoint = PE_CORE_S16


class PeCoreT16Encoder(PerceptionEncoder):
    name = "pe_core_t16"
    checkpoint = PE_CORE_T16


class MobileClip2S2Encoder(PerceptionEncoder):
    """MobileCLIP2-S2. Weights use the apple-amlr licence; check it before redistribution."""

    name = "mobileclip2_s2"
    checkpoint = MOBILECLIP2_S2


class MobileClip2S4Encoder(PerceptionEncoder):
    """MobileCLIP2-S4. Weights use the apple-amlr licence; check it before redistribution."""

    name = "mobileclip2_s4"
    checkpoint = MOBILECLIP2_S4


class MobileClip2BEncoder(PerceptionEncoder):
    """MobileCLIP2-B. Weights use the apple-amlr licence; check it before redistribution."""

    name = "mobileclip2_b"
    checkpoint = MOBILECLIP2_B


class Eva02ClipL14Encoder(PerceptionEncoder):
    """EVA02-CLIP-L/14 at 336 px (open_clip, MIT licence)."""

    name = "eva02_clip_l14"
    checkpoint = EVA02_CLIP_L14_336


class DfnClipL14Encoder(PerceptionEncoder):
    """Apple DFN2B-CLIP ViT-L/14 at 224 px (open_clip). apple-amlr licence."""

    name = "dfn_clip_l14"
    checkpoint = DFN2B_CLIP_L14


class DistilledPeEncoder(_FrameEncoder):
    """Student vision tower mapped into PE-L's embedding space.

    The text tower stays PE-L's. Pass a local checkpoint via ``checkpoint`` or
    ``SHAWAF_PE_DISTILLED``. This class is not part of the default benchmark groups.
    """

    name = "pe_core_distilled"

    def __init__(self, device: str = "cuda", checkpoint: str | None = None) -> None:
        import torch

        super().__init__(device)
        path = checkpoint or os.environ.get("SHAWAF_PE_DISTILLED", "")
        if not path:
            raise FileNotFoundError(
                "pe_core_distilled needs a local checkpoint. "
                "Set SHAWAF_PE_DISTILLED or pass checkpoint=."
            )
        blob = torch.load(Path(path), map_location="cpu")
        self.in_dim = int(blob["in_dim"])
        self.out_dim = int(blob["out_dim"])
        self._vision = None
        if blob.get("backbone") == "open_clip":
            self._vision = PerceptionEncoder(
                device=device,
                checkpoint=blob["backbone_checkpoint"],
                image_size=blob.get("image_size"),
            )
            backbone = _VisionForward(self._vision)
            in_dim = self.in_dim
        else:
            backbone = torch.nn.Linear(self.in_dim, self.in_dim)
            in_dim = self.in_dim
        from shawaf_vlm.distill.student import StudentWithProjection

        self.student = StudentWithProjection(backbone, in_dim, self.out_dim)
        self.student.load_state_dict(blob["state_dict"])
        self.student.eval()
        place_model(self.student, self.device)

    def _encode_images(self, images: list[Image.Image]):
        import torch

        if self._vision is not None:
            feats = self._vision._encode_images(images)
            projected = self.student.projection(feats)
            return l2_normalize_torch(projected)
        rows = []
        for image in images:
            color = image.resize((1, 1)).getpixel((0, 0))
            values = [float(color[0]), float(color[1]), float(color[2])]
            rows.append(torch.tensor(values, dtype=torch.float32).repeat(self.in_dim)[: self.in_dim])
        batch = torch.stack(rows, dim=0)
        return l2_normalize_torch(self.student(batch))

    def _encode_text_batch(self, texts: list[str]):
        if self._vision is None:
            raise RuntimeError(
                "This distilled checkpoint has no text tower. Encode queries with pe_core_l14."
            )
        return self._vision._encode_text_batch(texts)


class _VisionForward:
    """Call a PerceptionEncoder's image tower and return its embedding tensor."""

    def __init__(self, vision: PerceptionEncoder) -> None:
        self.vision = vision

    def __call__(self, images):
        return self.vision._encode_pixels(images)

    def parameters(self):
        return self.vision.model.parameters()

    def state_dict(self):
        return self.vision.model.state_dict()

    def load_state_dict(self, state, strict=True):
        return self.vision.model.load_state_dict(state, strict=strict)


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
