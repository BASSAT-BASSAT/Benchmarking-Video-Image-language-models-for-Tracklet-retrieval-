from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_embeddinggemma2_mean_pool_text_and_frames() -> None:
    pytest.importorskip("torch")
    fake_st = MagicMock()
    fake_st.encode_query.return_value = np.ones((2, 4), dtype=np.float32)
    frame_embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    fake_st.encode.return_value = frame_embs

    st_cls = MagicMock(return_value=fake_st)
    with patch("shawaf_vlm.models.embeddinggemma2.assert_embeddinggemma2_runtime"):
        with patch(
            "shawaf_vlm.models.embeddinggemma2.import_sentence_transformer",
            return_value=st_cls,
        ):
            with patch("torch.cuda.is_available", return_value=False):
                from shawaf_vlm.models.embeddinggemma2 import EmbeddingGemma2Encoder

                enc = EmbeddingGemma2Encoder(device="cpu")
    texts = enc.encode_texts(["a", "b"], batch_size=2)
    assert texts.shape == (2, 4)
    fake_st.encode_query.assert_called_once()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "f1.jpg"
        p2 = Path(tmp) / "f2.jpg"
        p1.write_bytes(b"x")
        p2.write_bytes(b"y")
        vecs = enc.encode_videos([[p1, p2]], batch_size=2)
    assert vecs.shape == (1, 4)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-5)


def test_encode_window_rejects_raw_video_paths() -> None:
    pytest.importorskip("torch")
    fake_st = MagicMock()
    st_cls = MagicMock(return_value=fake_st)
    with patch("shawaf_vlm.models.embeddinggemma2.assert_embeddinggemma2_runtime"):
        with patch(
            "shawaf_vlm.models.embeddinggemma2.import_sentence_transformer",
            return_value=st_cls,
        ):
            with patch("torch.cuda.is_available", return_value=False):
                from shawaf_vlm.models.embeddinggemma2 import EmbeddingGemma2Encoder

                enc = EmbeddingGemma2Encoder(device="cpu")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "clip.mp4"
        mp4.write_bytes(b"x")
        with pytest.raises(RuntimeError, match="raw video"):
            enc._encode_one_window([mp4], batch_size=1)


def test_build_encoder_registry_key() -> None:
    with patch("shawaf_vlm.models.embeddinggemma2.EmbeddingGemma2Encoder") as mock_cls:
        mock_cls.return_value = MagicMock(name="embeddinggemma2")
        from shawaf_vlm.models.registry import build_encoder

        build_encoder("embeddinggemma2", device="cpu")
    mock_cls.assert_called_once_with(device="cpu")
