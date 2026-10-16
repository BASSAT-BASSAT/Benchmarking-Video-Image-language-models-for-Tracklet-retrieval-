"""Work around broken optional torchcodec on Kaggle/Colab before importing ST."""

from __future__ import annotations

import sys


def purge_torchcodec_modules() -> None:
    """Remove torchcodec from sys.modules (partial imports break transformers find_spec)."""

    for key in list(sys.modules):
        if key == "torchcodec" or key.startswith("torchcodec."):
            sys.modules.pop(key, None)


def neutralize_broken_torchcodec() -> None:
    """Drop unloadable torchcodec so transformers and sentence-transformers can import.

    Do not register a fake ``torchcodec`` module: transformers checks
    ``importlib.util.find_spec("torchcodec")`` and raises if ``__spec__`` is None.
    """

    purge_torchcodec_modules()
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import AudioDecoder, VideoDecoder  # noqa: F401
    except (ImportError, OSError, RuntimeError, ValueError):
        purge_torchcodec_modules()


_MIN_SENTENCE_TRANSFORMERS = "6.1.0"


def import_sentence_transformer():
    """Return SentenceTransformer class after torchcodec cleanup."""

    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        st_version = version("sentence-transformers")
    except PackageNotFoundError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            'Use: pip install -e ".[embeddinggemma2]"'
        ) from None
    if Version(st_version) < Version(_MIN_SENTENCE_TRANSFORMERS):
        raise RuntimeError(
            f"google/embeddinggemma-2 requires sentence-transformers>={_MIN_SENTENCE_TRANSFORMERS}, "
            f"found {st_version}. Run: pip uninstall -y torchcodec sentence-transformers && "
            "pip install 'sentence-transformers>=6.1.0,<7'"
        )

    neutralize_broken_torchcodec()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer
