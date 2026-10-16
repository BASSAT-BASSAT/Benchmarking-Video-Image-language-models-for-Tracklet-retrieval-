"""Work around broken optional torchcodec on Kaggle/Colab before importing ST."""

from __future__ import annotations

import sys


def purge_torchcodec_modules() -> None:
    """Remove torchcodec from sys.modules (partial imports break transformers find_spec)."""

    for key in list(sys.modules):
        if key == "torchcodec" or key.startswith("torchcodec."):
            sys.modules.pop(key, None)


def purge_torchaudio_modules() -> None:
    """Drop torchaudio from sys.modules after pip uninstall (Kaggle ABI mismatch with torch)."""

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            sys.modules.pop(key, None)


def prepare_embeddinggemma2_import_env() -> None:
    """Purge optional audio wheels before importing sentence-transformers / transformers 5.19."""

    neutralize_broken_torchcodec()
    purge_torchaudio_modules()


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


# From google/embeddinggemma-2 config_sentence_transformers.json on the Hub.
_MIN_SENTENCE_TRANSFORMERS = "6.1.0"
_MIN_TRANSFORMERS = "5.19.0"  # embedding_gemma2 architecture (Transformers v5.19.0, Oct 2026)
CHECKPOINT = "google/embeddinggemma-2"


def assert_embeddinggemma2_runtime() -> None:
    """Validate versions required by the Hub checkpoint (ST 6.1+, transformers 5.19+)."""

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
            f"{CHECKPOINT} requires sentence-transformers>={_MIN_SENTENCE_TRANSFORMERS} "
            f"(Hub modules use sentence_transformers.base); found {st_version}."
        )

    try:
        tf_version = version("transformers")
    except PackageNotFoundError:
        raise RuntimeError(
            "transformers is not installed. "
            'Use: pip install -e ".[embeddinggemma2]"'
        ) from None
    if Version(tf_version) < Version(_MIN_TRANSFORMERS):
        raise RuntimeError(
            f"{CHECKPOINT} needs transformers>={_MIN_TRANSFORMERS} (embedding_gemma2); "
            f"found {tf_version}."
        )


def import_sentence_transformer():
    """Return SentenceTransformer class after torchcodec cleanup."""

    assert_embeddinggemma2_runtime()
    prepare_embeddinggemma2_import_env()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer
