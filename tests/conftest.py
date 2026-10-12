"""Shared pytest hooks for SHAWAF."""

from __future__ import annotations

import pytest


def _transformers_import_error() -> str | None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        return str(exc)
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_transformers: requires a working transformers + huggingface-hub pair",
    )
    config.addinivalue_line(
        "markers",
        "needs_transformers_v5: InternVideo shims that target transformers 5.x APIs",
    )


def _transformers_major() -> int | None:
    err = _transformers_import_error()
    if err:
        return None
    import transformers

    return int(transformers.__version__.split(".", 1)[0])


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("needs_transformers") is not None:
        err = _transformers_import_error()
        if err:
            pytest.skip(
                f"transformers not importable ({err}). "
                "See docs/ENVIRONMENTS.md — fix huggingface-hub vs transformers versions."
            )
    if item.get_closest_marker("needs_transformers_v5") is not None:
        major = _transformers_major()
        if major is None:
            pytest.skip("transformers not importable")
        if major < 5:
            pytest.skip(f"transformers {major}.x — shim tests require 5.x")
