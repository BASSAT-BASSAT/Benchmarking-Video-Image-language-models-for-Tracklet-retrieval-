from __future__ import annotations

import builtins
import sys
from unittest.mock import patch

import pytest

from shawaf_vlm.models.sentence_transformers_compat import (
    assert_embeddinggemma2_runtime,
    import_sentence_transformer,
    neutralize_broken_torchcodec,
    purge_torchcodec_modules,
)


def _clear_torchcodec_modules() -> None:
    for key in list(sys.modules):
        if key == "torchcodec" or key.startswith("torchcodec."):
            sys.modules.pop(key, None)


def test_purge_torchcodec_modules_removes_entries() -> None:
    _clear_torchcodec_modules()
    sys.modules["torchcodec"] = object()
    sys.modules["torchcodec.decoders"] = object()
    purge_torchcodec_modules()
    assert "torchcodec" not in sys.modules
    assert not any(k.startswith("torchcodec.") for k in sys.modules)


def test_neutralize_does_not_leave_stub_modules() -> None:
    _clear_torchcodec_modules()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if isinstance(name, str) and (name == "torchcodec" or name.startswith("torchcodec")):
            raise RuntimeError("libtorchcodec")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        neutralize_broken_torchcodec()

    assert "torchcodec" not in sys.modules
    _clear_torchcodec_modules()


def test_assert_embeddinggemma2_runtime_requires_st_61() -> None:
    def fake_version(name: str) -> str:
        if name == "sentence-transformers":
            return "5.1.2"
        return "5.1.0"

    with patch("importlib.metadata.version", side_effect=fake_version):
        with pytest.raises(RuntimeError, match="6.1.0"):
            assert_embeddinggemma2_runtime()


def test_assert_embeddinggemma2_runtime_requires_tf_5() -> None:
    def fake_version(name: str) -> str:
        if name == "sentence-transformers":
            return "6.1.0"
        return "4.57.3"

    with patch("importlib.metadata.version", side_effect=fake_version):
        with pytest.raises(RuntimeError, match="transformers>=5"):
            assert_embeddinggemma2_runtime()
