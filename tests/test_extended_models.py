from __future__ import annotations

import inspect
import json
from pathlib import Path
import tomllib

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


def test_gme_does_not_override_remote_torch_dtype() -> None:
    from shawaf_vlm.models.gme_qwen2_vl import GmeQwen2VLEncoder

    source = inspect.getsource(GmeQwen2VLEncoder.__init__)
    assert "SentenceTransformer" in source
    assert "AutoModel" not in source
    assert "model_kwargs" not in source
    assert "torch_dtype" not in source
    assert GmeQwen2VLEncoder.frame_microbatch == 1


def test_gme_video_encoding_uses_absolute_paths_and_frame_mean(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from shawaf_vlm.models.gme_qwen2_vl import GmeQwen2VLEncoder

    clips = [
        [tmp_path / "a.jpg", tmp_path / "b.jpg"],
        [tmp_path / "c.jpg"],
    ]
    encoder = GmeQwen2VLEncoder.__new__(GmeQwen2VLEncoder)
    encoder.device = "cpu"
    captured = []

    class FakeModel:
        def encode(self, items, **kwargs):
            captured.extend(items)
            return torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
                dtype=torch.float32,
            )

    encoder.model = FakeModel()
    features = encoder.encode_videos(clips, batch_size=2)
    assert [item["image"] for item in captured] == [
        str(path.resolve()) for clip in clips for path in clip
    ]
    assert all(isinstance(item["image"], str) for item in captured)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        features[0],
        np.array([2**-0.5, 2**-0.5], dtype=np.float32),
        atol=1e-6,
    )


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


def test_extended_dependency_profiles_are_mutually_compatible() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]
    modern = set(extras["extended_modern"])
    gme = set(extras["extended_gme"])
    assert "transformers==4.57.3" in modern
    assert "transformers==4.51.3" in gme
    assert "extended" not in extras

    notebook = json.loads(
        (root / "notebooks" / "colab_extended_zero_shot.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert '"gme_qwen2_vl_2b": "gme"' in source
    assert '"qwen3_vl_embed_2b": "modern"' in source
    assert '"modern": "4.57.3", "gme": "4.51.3"' in source
    assert "RUNTIME RESTART REQUIRED" in source


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
