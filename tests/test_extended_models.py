from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_extended_encoder_registry() -> None:
    from shawaf_vlm.models.registry import all_specs

    specs = all_specs()
    assert specs["openai_clip_vit_l14"].checkpoint == "OpenAI ViT-L/14"
    assert specs["jina_clip_v2"].checkpoint == "jinaai/jina-clip-v2"
    assert (
        specs["gme_qwen2_vl_2b"].checkpoint
        == "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    )
    assert (
        specs["qwen3_vl_embed_2b"].checkpoint
        == "Qwen/Qwen3-VL-Embedding-2B"
    )


def test_gme_uses_new_transformers_compatible_official_path() -> None:
    from shawaf_vlm.models.gme_qwen2_vl import GmeQwen2VLEncoder

    source = inspect.getsource(GmeQwen2VLEncoder.__init__)
    assert "SentenceTransformer" in source
    assert "AutoModel" not in source
    assert GmeQwen2VLEncoder.frame_microbatch == 1


def test_qwen3_video_input_preserves_order_and_context(tmp_path: Path) -> None:
    from shawaf_vlm.models.qwen3_vl_embedding import (
        NEUTRAL_INSTRUCTION,
        QWEN3_MAX_LENGTH,
        Qwen3VLEmbeddingEncoder,
    )

    assert QWEN3_MAX_LENGTH == 8192
    encoder = Qwen3VLEmbeddingEncoder.__new__(Qwen3VLEmbeddingEncoder)
    encoder.max_frames = 64
    encoder.instruction = NEUTRAL_INSTRUCTION
    paths = [tmp_path / f"frame_{index:03d}.jpg" for index in range(8)]
    conversation = encoder._format_input({"video": paths})
    content = conversation[1]["content"][0]
    assert content["type"] == "video"
    assert [Path(uri).name for uri in content["video"]] == [path.name for path in paths]
    assert conversation[0]["content"][0]["text"] == NEUTRAL_INSTRUCTION
    source = inspect.getsource(Qwen3VLEmbeddingEncoder._preprocess)
    assert "truncation=False" in source
    assert "sequence_length > self.max_length" in source


def test_openai_clip_uses_official_checkpoint() -> None:
    from shawaf_vlm.models.openai_clip import OPENAI_CLIP_MODEL, OpenAIClipEncoder

    assert OPENAI_CLIP_MODEL == "ViT-L/14"
    source = inspect.getsource(OpenAIClipEncoder.__init__)
    assert "clip.load" in source


def test_jina_numpy_outputs_are_converted_to_torch() -> None:
    torch = pytest.importorskip("torch")
    from shawaf_vlm.models.jina_clip_v2 import JinaClipV2Encoder, _to_torch_tensor

    converted = _to_torch_tensor(np.array([[3.0, 4.0]], dtype=np.float32))
    assert isinstance(converted, torch.Tensor)

    encoder = JinaClipV2Encoder.__new__(JinaClipV2Encoder)

    class FakeModel:
        def encode_image(self, images, truncate_dim=None):
            return np.array([[3.0, 4.0]], dtype=np.float32)

        def encode_text(self, texts, truncate_dim=None):
            return np.array([[0.0, 5.0]], dtype=np.float32)

    encoder.model = FakeModel()
    image_features = encoder._encode_images([object()])
    text_features = encoder._encode_text_batch(["person"])
    assert isinstance(image_features, torch.Tensor)
    assert isinstance(text_features, torch.Tensor)
    torch.testing.assert_close(image_features.norm(dim=-1), torch.ones(1))
