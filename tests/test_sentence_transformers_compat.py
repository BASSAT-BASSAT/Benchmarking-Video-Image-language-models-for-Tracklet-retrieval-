from __future__ import annotations

import builtins
import sys
from unittest.mock import patch

from shawaf_vlm.models.sentence_transformers_compat import neutralize_broken_torchcodec


def _clear_torchcodec_modules() -> None:
    for key in list(sys.modules):
        if key == "torchcodec" or key.startswith("torchcodec."):
            sys.modules.pop(key, None)


def test_neutralize_broken_torchcodec_stubs_after_runtime_error() -> None:
    _clear_torchcodec_modules()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if isinstance(name, str) and (name == "torchcodec" or name.startswith("torchcodec")):
            raise RuntimeError("libtorchcodec")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        neutralize_broken_torchcodec()

    dec = sys.modules["torchcodec.decoders"]
    try:
        dec.AudioDecoder()
        raised = False
    except ImportError:
        raised = True
    assert raised
    _clear_torchcodec_modules()
