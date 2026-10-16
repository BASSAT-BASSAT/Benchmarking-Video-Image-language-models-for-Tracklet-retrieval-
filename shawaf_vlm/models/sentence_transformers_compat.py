"""Work around broken optional torchcodec on Kaggle/Colab before importing ST."""

from __future__ import annotations

import sys
import types


def neutralize_broken_torchcodec() -> None:
    """If torchcodec is installed but cannot load FFmpeg libs, stub it for ST import.

    sentence-transformers >= 5.4 imports torchcodec at module import time. A broken
    install raises RuntimeError (not ImportError), which aborts loading text/image models.
    """

    prefix = "torchcodec"
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import AudioDecoder, VideoDecoder  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        for key in list(sys.modules):
            if key == prefix or key.startswith(f"{prefix}."):
                sys.modules.pop(key, None)
        stub = types.ModuleType(prefix)
        decoders = types.ModuleType(f"{prefix}.decoders")

        class _Unavailable:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise ImportError(
                    "torchcodec is unavailable (FFmpeg/libtorchcodec missing). "
                    "Use frame images for EmbeddingGemma 2 or install FFmpeg."
                )

        decoders.AudioDecoder = _Unavailable
        decoders.VideoDecoder = _Unavailable
        sys.modules[prefix] = stub
        sys.modules[f"{prefix}.decoders"] = decoders


def import_sentence_transformer():
    """Return SentenceTransformer class after torchcodec neutralization."""

    neutralize_broken_torchcodec()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer
