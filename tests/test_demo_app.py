from __future__ import annotations

from pathlib import Path


def test_demo_app_has_no_download_buttons() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("demo", "app.py").read_text(encoding="utf-8")
    assert "Download TVPReid" not in text
    assert "download_tvpreid_ui" not in text
    assert "download_tier_a.py" in text


def test_demo_stack_script_exists() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_demo_stack.py"
    assert script.is_file()


def test_resolve_ffmpeg_binary_prefers_env(tmp_path: Path, monkeypatch) -> None:
    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from demo.app import _resolve_ffmpeg_binary

    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("FFMPEG_BINARY", str(fake))
    assert _resolve_ffmpeg_binary() == str(fake)


def test_gradio_file_url_encodes_spaces() -> None:
    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from demo.app import _gradio_file_url

    url = _gradio_file_url(r"C:\Users\Legion 5\Desktop\clip.mp4")
    assert "%20" in url
    assert "/file=" in url


def test_gallery_html_video_shows_frame_preview(tmp_path: Path) -> None:
    import sys

    import cv2
    import numpy as np

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from demo.app import _gallery_html

    clip = tmp_path / "t.mp4"
    writer = cv2.VideoWriter(
        str(clip),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :] = (40, 120, 200)
    for _ in range(5):
        writer.write(frame)
    writer.release()

    html = _gallery_html([(str(clip), "#1 demo")])
    assert "data:image/jpeg;base64," in html
    assert "#1 demo" in html


def test_build_demo_api_info() -> None:
    import importlib.util

    if importlib.util.find_spec("gradio") is None:
        return
    import sys

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from demo.app import build_demo

    from demo.app import _patch_gradio_client_schema

    _patch_gradio_client_schema()
    demo = build_demo()
    demo.get_api_info()
