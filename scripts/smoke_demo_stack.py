#!/usr/bin/env python3
"""Smoke checks for demo stack (no GPU, no Gradio server). Run: python scripts/smoke_demo_stack.py"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"OK  {name}")
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        raise


def check_imports() -> None:
    import shawaf_vlm.retrieval.tier_a_download as td
    import shawaf_vlm.retrieval.search as search
    from shawaf_vlm.retrieval.datasets import dataset_status

    dataset_status()
    assert hasattr(td, "ensure_tier_a_datasets")
    assert search.RetrievalMode


def check_huggingface_hub_compat() -> None:
    import importlib.metadata

    import huggingface_hub

    hub_major = int(huggingface_hub.__version__.split(".", 1)[0])
    try:
        tf_major = int(importlib.metadata.version("transformers").split(".", 1)[0])
    except importlib.metadata.PackageNotFoundError:
        return
    if tf_major < 5 and hub_major >= 1:
        print(
            f"WARN huggingface-hub {huggingface_hub.__version__} breaks transformers 4.x — "
            'pip install "huggingface-hub>=0.23,<1.0"'
        )
        return
    if tf_major >= 5 and hub_major < 1:
        print(
            f"WARN huggingface-hub {huggingface_hub.__version__} is too old for transformers 5.x — "
            'pip install "huggingface-hub>=1.5,<2.0"'
        )


def check_demo_module_structure() -> None:
    path = REPO / "demo" / "app.py"
    text = path.read_text(encoding="utf-8")
    assert "Download TVPReid" not in text
    assert "download_tvpreid_ui" not in text
    assert "download_tier_a.py" in text


def check_build_demo_callable() -> None:
    if importlib.util.find_spec("gradio") is None:
        print("SKIP build_demo (gradio not installed)")
        return
    sys.path.insert(0, str(REPO))
    try:
        from demo.app import build_demo

        demo = build_demo()
        assert demo is not None
    except ModuleNotFoundError as exc:
        if "pyaudioop" in str(exc) or "audioop" in str(exc):
            print("SKIP build_demo (install demo extra: pyaudioop on Python 3.13+)")
            return
        raise


def main() -> None:
    _check("imports", check_imports)
    check_huggingface_hub_compat()
    print("OK  huggingface_hub (or WARN above)")
    _check("demo_no_ui_download", check_demo_module_structure)
    _check("build_demo", check_build_demo_callable)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
