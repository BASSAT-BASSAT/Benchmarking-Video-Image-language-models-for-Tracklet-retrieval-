"""Google EmbeddingGemma 2 via Sentence Transformers (text + image inputs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shawaf_vlm.models.runtime import resolve_device
from shawaf_vlm.models.sentence_transformers_compat import import_sentence_transformer
from shawaf_vlm.sampling import VIDEO_SUFFIXES

CHECKPOINT = "google/embeddinggemma-2"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (matrix / norms).astype(np.float32)


def _mean_pool_clip(frame_embeddings: np.ndarray) -> np.ndarray:
    if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] < 1:
        raise ValueError(f"Expected (K, D) frame embeddings, got {frame_embeddings.shape}")
    pooled = _l2_normalize_rows(frame_embeddings).mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 1e-12:
        pooled = pooled / norm
    return pooled.astype(np.float32)


class EmbeddingGemma2Encoder:
    """Frozen EmbeddingGemma 2 aligned with SHAWAF's clip-window protocol."""

    name = "embeddinggemma2"

    def __init__(
        self,
        device: str = "cuda",
        *,
        load_audio: bool = False,
        native_video_clip: bool = False,
        video_fps: float = 1.0,
    ) -> None:
        import torch

        SentenceTransformer = import_sentence_transformer()

        self._device = resolve_device(device)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        config_kwargs: dict[str, object] = {"audio_config": None} if not load_audio else {}
        self._st = SentenceTransformer(
            CHECKPOINT,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs=config_kwargs,
            device=str(self._device),
        )
        self._native_video_clip = native_video_clip
        self._video_fps = video_fps

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        encode_query = getattr(self._st, "encode_query", None)
        if callable(encode_query):
            vectors = encode_query(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 32,
            )
        else:
            vectors = self._st.encode(
                texts,
                prompt_name="SearchQuery",
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 32,
            )
        return np.asarray(vectors, dtype=np.float32)

    def encode_videos(self, videos: list[list[Path]], batch_size: int = 4) -> np.ndarray:
        if not videos:
            return np.zeros((0, 1), dtype=np.float32)
        out: list[np.ndarray] = []
        for window in videos:
            out.append(self._encode_one_window(window, batch_size=batch_size))
        return np.stack(out, axis=0).astype(np.float32)

    def _encode_one_window(self, window: list[Path], batch_size: int) -> np.ndarray:
        paths = [Path(p) for p in window if Path(p).is_file()]
        if not paths:
            raise FileNotFoundError("Window has no readable frame paths.")

        if self._native_video_clip and len(paths) == 1 and paths[0].suffix.lower() in VIDEO_SUFFIXES:
            vec = self._st.encode(
                [{"video": str(paths[0])}],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vec[0], dtype=np.float32)

        payloads: list[dict[str, str]] = []
        for path in paths:
            suffix = path.suffix.lower()
            if suffix in _IMAGE_SUFFIXES:
                payloads.append({"image": str(path)})
            elif suffix in VIDEO_SUFFIXES:
                raise RuntimeError(
                    f"EmbeddingGemma2 window contains raw video {path.name}; "
                    "use frame paths from SHAWAF sampling (or set native_video_clip=True "
                    "with FFmpeg/torchcodec available)."
                )
            else:
                payloads.append({"image": str(path)})

        frame_embs = self._st.encode(
            payloads,
            batch_size=max(1, batch_size),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _mean_pool_clip(np.asarray(frame_embs, dtype=np.float32))
