from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class VideoTextEncoder(Protocol):
    name: str

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 4,
    ) -> np.ndarray:
        """Return L2-normalized video embeddings of shape (N, D)."""

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return L2-normalized text embeddings of shape (M, D)."""
