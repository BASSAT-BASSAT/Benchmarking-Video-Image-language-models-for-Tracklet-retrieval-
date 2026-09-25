"""GME-Qwen2-VL-2B as a frozen frame/text encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from shawaf_vlm.models.frame_encoders import _FrameEncoder, mean_pool_clips
from shawaf_vlm.models.runtime import l2_normalize_torch, to_numpy, unwrap_features

GME_QWEN2_VL_2B = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"


class GmeQwen2VLEncoder(_FrameEncoder):
    """Official GME Sentence Transformers path for Transformers 4.51.3."""

    name = "gme_qwen2_vl_2b"
    checkpoint = GME_QWEN2_VL_2B
    frame_microbatch = 1

    def __init__(self, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer

        super().__init__(device)
        self.model = SentenceTransformer(
            self.checkpoint,
            device=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 1,
    ) -> np.ndarray:
        """Encode absolute frame paths, then apply the Stage 1 frame mean."""

        import torch

        chunks: list[np.ndarray] = []
        step = max(int(batch_size), 1)
        for start in tqdm(
            range(0, len(videos), step),
            desc=f"{self.name} videos",
            unit="batch",
        ):
            batch = videos[start : start + step]
            items: list[dict[str, str]] = []
            lengths: list[int] = []
            for clip in batch:
                if not clip:
                    raise FileNotFoundError("A GME clip has no frames.")
                paths = [str(Path(path).resolve()) for path in clip]
                items.extend({"image": path} for path in paths)
                lengths.append(len(paths))
            features = self.model.encode(
                items,
                batch_size=1,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            frame_features = l2_normalize_torch(unwrap_features(features))
            pooled = mean_pool_clips(frame_features, lengths)
            chunks.append(to_numpy(pooled))
            del features, frame_features, pooled
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return np.concatenate(chunks, axis=0)

    def _encode_text_batch(self, texts: list[str]):
        # The official ST route uses GME's neutral default instruction.
        features = self.model.encode(
            texts,
            batch_size=max(len(texts), 1),
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return unwrap_features(features)
