"""Gradio demo: text → tracklet retrieval with pooling modes (datasets via CLI download)."""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import numpy as np

from shawaf_vlm.models.groups import MODEL_GROUPS, models_for_group
from shawaf_vlm.models.registry import all_specs, build_encoder
from shawaf_vlm.retrieval.cache import index_dir, load_numpy_index
from shawaf_vlm.retrieval.datasets import TIER_A_DATASETS, dataset_status
from shawaf_vlm.retrieval.search import top_k_tracklets

REPO_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_K = 5
MODE_LABELS = {
    "Average (global mean)": "mean",
    "Max (best clip)": "query_max",
    "Type C (global + local)": "global_local",
}


def _cache_root() -> Path:
    env = os.environ.get("SHAWAF_INDEX_ROOT")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "shawaf_vlm" / "indexes"


def _model_choices() -> list[str]:
    group = os.environ.get("SHAWAF_MODEL_GROUP", "all")
    allowed = models_for_group(group if group != "all" else None)
    specs = all_specs()
    keys = sorted(specs.keys())
    if allowed is not None:
        keys = [k for k in keys if k in allowed]
    return keys


def _dataset_choices() -> list[str]:
    return [d.key for d in TIER_A_DATASETS]


def _has_index(model: str, dataset: str) -> bool:
    return (index_dir(_cache_root(), model, dataset) / "manifest.json").is_file()


def _default_model() -> str:
    choices = _model_choices()
    prefer = os.environ.get("SHAWAF_DEMO_MODEL")
    if prefer and prefer in choices:
        return prefer
    for key in ("pe_core_l14", "irra", "siglip2"):
        if key in choices and any(_has_index(key, d) for d in _dataset_choices()):
            return key
    for model in choices:
        if any(_has_index(model, d) for d in _dataset_choices()):
            return model
    return choices[0]


def _default_dataset(model: str) -> str:
    for dataset in _dataset_choices():
        if _has_index(model, dataset):
            return dataset
    return "tvpreid:prid"


def _status_markdown() -> str:
    st = dataset_status()
    lines = [
        f"- **TVPReid (HF):** {'✅ on disk' if st['tvpreid'] == 'ready' else '⚠️ missing'}",
        f"- **GroOT-MOT17 (HF):** {'✅ on disk' if st['groot'] == 'ready' else '⚠️ missing'}",
        f"- **RSTPReid (Google Drive):** {'✅ on disk' if st['rstpreid'] == 'ready' else '⚠️ missing'}",
        "",
        "Download data from the terminal (not in this UI):",
        "`python scripts/download_tier_a.py`",
        "",
        "Then build an index:",
        "`python scripts/index_qdrant.py --model siglip2 --dataset tvpreid:prid`",
    ]
    return "\n".join(lines)


def _index_status(model: str, dataset: str) -> str:
    path = index_dir(_cache_root(), model, dataset)
    manifest = path / "manifest.json"
    if manifest.is_file():
        return f"✅ Index at `{path}`"
    return f"⚠️ No index — run **Build / refresh index** or `scripts/index_qdrant.py`."


@lru_cache(maxsize=2)
def _get_encoder(model: str):
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    return build_encoder(model, device=device)


def _load_tracklets(model: str, dataset: str):
    path = index_dir(_cache_root(), model, dataset)
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(
            f"No numpy index at {path}. Run scripts/index_qdrant.py for this model and dataset."
        )
    return load_numpy_index(path)


_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _gradio_file_url(path: str) -> str:
    """Path Gradio's /file= route can serve (works on Windows)."""
    resolved = Path(path).resolve().as_posix()
    return "/file=" + quote(resolved, safe="/:")


def _web_preview_cache_dir() -> Path:
    return Path.home() / ".cache" / "shawaf_vlm" / "web_previews"


def _video_jpeg_data_url(path: str, frame_index: int = 0) -> str | None:
    """First-frame JPEG preview (works for mp4v clips browsers cannot decode)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total > 0 and frame_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_index, total - 1))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return None
    ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok_enc:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _resolve_ffmpeg_binary() -> str | None:
    """System ffmpeg, FFMPEG_BINARY, or bundled imageio-ffmpeg (no admin install)."""
    override = os.environ.get("FFMPEG_BINARY")
    if override and Path(override).is_file():
        return override
    if override and shutil.which(override):
        return override
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return bundled
    except ImportError:
        pass
    return None


def _browser_h264_path(source: str) -> str | None:
    """Transcode to H.264 once (cached) when ffmpeg is available; else None."""
    src = Path(source)
    if not src.is_file():
        return None
    cache_dir = _web_preview_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(src.resolve()).encode("utf-8")).hexdigest()[:20]
    out = cache_dir / f"{digest}.mp4"
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        return str(out)
    ffmpeg = _resolve_ffmpeg_binary()
    if not ffmpeg:
        return None
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not out.is_file():
        return None
    return str(out)


def _gallery_html(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return "<p><em>No previews.</em></p>"
    parts = [
        '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;">',
    ]
    for path, caption in entries:
        safe_cap = caption.replace("<", "&lt;").replace(">", "&gt;")
        if not path or not Path(path).is_file():
            parts.append(f'<p style="font-size:0.8rem;">{safe_cap}<br/><code>{path or "—"}</code></p>')
            continue
        suffix = Path(path).suffix.lower()
        src = _gradio_file_url(path)
        if suffix in _VIDEO_SUFFIXES:
            poster = _video_jpeg_data_url(path)
            playable = _browser_h264_path(path)
            media_bits: list[str] = []
            if playable:
                vsrc = _gradio_file_url(playable)
                poster_attr = f' poster="{poster}"' if poster else ""
                media_bits.append(
                    f'<video src="{vsrc}"{poster_attr} controls playsinline preload="metadata" '
                    f'style="max-height:200px;max-width:100%;border-radius:6px;background:#111;">'
                    f"</video>"
                )
            elif poster:
                media_bits.append(
                    f'<img src="{poster}" alt="" '
                    f'style="max-height:200px;max-width:100%;border-radius:6px;background:#111;"/>'
                )
                media_bits.append(
                    '<p style="font-size:0.65rem;color:#888;margin:4px 0 0;">'
                    "Still frame only — install <code>pip install imageio-ffmpeg</code> "
                    "(or system ffmpeg) and search again for H.264 playback.</p>"
                )
            else:
                media_bits.append(
                    f'<video src="{src}" controls playsinline preload="metadata" '
                    f'style="max-height:200px;max-width:100%;border-radius:6px;background:#111;">'
                    f"</video>"
                )
            parts.append(
                f'<figure style="margin:0;text-align:center;max-width:260px;">'
                + "".join(media_bits)
                + f'<figcaption style="font-size:0.75rem;margin-top:4px;">{safe_cap}</figcaption>'
                f"</figure>"
            )
        elif suffix in _IMAGE_SUFFIXES:
            parts.append(
                f'<figure style="margin:0;text-align:center;max-width:140px;">'
                f'<img src="{src}" alt="" style="max-height:120px;max-width:100%;"/>'
                f'<figcaption style="font-size:0.75rem;">{safe_cap}</figcaption></figure>'
            )
        else:
            parts.append(f'<p style="font-size:0.8rem;">{safe_cap}<br/><code>{path}</code></p>')
    parts.append("</div>")
    return "\n".join(parts)


def _allowed_media_paths() -> list[str]:
    """Gradio must allow serving dataset videos and cached assets."""
    roots = [
        REPO_ROOT / "datasets",
        Path.home() / ".cache" / "shawaf_vlm",
    ]
    env = os.environ.get("SHAWAF_DATA_ROOT")
    if env:
        roots.append(Path(env))
    return [str(p.resolve()) for p in roots if p.is_dir()]


def build_index_ui(model: str, dataset: str) -> str:
    script = REPO_ROOT / "scripts" / "index_qdrant.py"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model,
        "--dataset",
        dataset,
        "--skip-qdrant",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return f"Index failed (code {proc.returncode}):\n```\n{log[-4000:]}\n```"
    return f"Index built.\n```\n{log[-2000:]}\n```"


def search_ui(
    query: str,
    mode_label: str,
    model: str,
    dataset: str,
    alpha: float,
) -> tuple:
    status = _index_status(model, dataset)
    if not query.strip():
        return "_Enter a query and click Search._", status, _gallery_html([])
    mode = MODE_LABELS[mode_label]
    try:
        tracklets, _manifest = _load_tracklets(model, dataset)
    except FileNotFoundError as exc:
        hint = _default_dataset(model)
        extra = f" Try dataset `{hint}`." if hint != dataset and _has_index(model, hint) else ""
        return (
            f"**No index for this model + dataset.**\n\n{exc}{extra}",
            status,
            _gallery_html([]),
        )
    encoder = _get_encoder(model)
    qvec = encoder.encode_texts([query.strip()])[0]
    if not isinstance(qvec, np.ndarray):
        qvec = np.asarray(qvec)
    results = top_k_tracklets(qvec, tracklets, mode, k=PREVIEW_K, hybrid_alpha=alpha)
    lines = [
        "| rank | tracklet_id | fused | global | local |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    gallery: list[tuple[str, str]] = []
    for rank, (t, fused, g_score, l_score) in enumerate(results, start=1):
        lines.append(
            f"| {rank} | {t.tracklet_id} | {fused:.4f} | {g_score:.4f} | {l_score:.4f} |"
        )
        gallery.append((t.media_path, f"#{rank} {t.tracklet_id} ({fused:.3f})"))
    table_md = "\n".join(lines) if len(lines) > 2 else "_No results._"
    return table_md, status, _gallery_html(gallery)


def build_demo():
    import gradio as gr

    with gr.Blocks(title="SHAWAF tracklet retrieval") as demo:
        gr.Markdown(
            "# SHAWAF — text → tracklet retrieval\n"
            "Frozen **text ↔ tracklet** search over a pre-built gallery index (Tier-A person/video benchmarks). "
            "This is **not** open-world video QA: queries work best like the benchmark captions "
            "(clothing, build, bag, colors). Action phrases (e.g. “eating lunch”) only help if the "
            "encoder and frames support them.\n\n"
            "Pooling **Type C** uses a global tracklet vector plus **best matching short window** "
            "(weak temporal cue, not full event understanding). Rebuild the index after changing encoder."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Dataset status")
                gr.Markdown(_status_markdown())

                gr.Markdown("### Search settings")
                default_model = _default_model()
                default_dataset = _default_dataset(default_model)
                dataset = gr.Dropdown(
                    _dataset_choices(), value=default_dataset, label="Dataset"
                )
                model = gr.Dropdown(_model_choices(), value=default_model, label="Encoder")
                mode = gr.Dropdown(
                    list(MODE_LABELS.keys()),
                    value="Type C (global + local)",
                    label="Retrieval mode",
                )
                alpha = gr.Slider(0, 1, value=0.5, step=0.05, label="α (global weight for Type C)")
                index_md = gr.Markdown(_index_status(default_model, default_dataset))
                build_btn = gr.Button("Build / refresh index (numpy; optional Qdrant via CLI)")
                build_log = gr.Markdown("")
                build_btn.click(build_index_ui, inputs=[model, dataset], outputs=build_log)

                gr.Markdown(
                    "**Model groups:** set `SHAWAF_MODEL_GROUP` to `base`, `extended_modern`, or `extended_gme` "
                    f"and restart. Groups: {', '.join(MODEL_GROUPS.keys())}."
                )

            with gr.Column(scale=2):
                query = gr.Textbox(label="Text query", lines=2)
                search_btn = gr.Button("Search", variant="primary")
                table = gr.Markdown(label="Top 5")
                gallery = gr.HTML(
                    label="Previews",
                    value=(
                        "<p><em>Search for top-5 tracklets. "
                        "Install <code>imageio-ffmpeg</code> for playable H.264 previews.</em></p>"
                    ),
                )

        def on_mode_change(label: str):
            return gr.update(visible=(label == "Type C (global + local)"))

        mode.change(on_mode_change, inputs=mode, outputs=alpha)

        def refresh_index(model_key: str, dataset_key: str):
            return _index_status(model_key, dataset_key)

        model.change(refresh_index, inputs=[model, dataset], outputs=index_md)
        dataset.change(refresh_index, inputs=[model, dataset], outputs=index_md)

        search_btn.click(
            search_ui,
            inputs=[query, mode, model, dataset, alpha],
            outputs=[table, index_md, gallery],
        )
    return demo


def _patch_gradio_client_schema() -> None:
    """Gradio 4.44 + gradio_client crash when Video/Gallery schema uses bool additionalProperties."""
    try:
        import gradio_client.utils as client_utils

        if getattr(client_utils, "_shawaf_schema_patched", False):
            return
        _orig_get_type = client_utils.get_type

        def get_type(schema):  # type: ignore[no-untyped-def]
            if not isinstance(schema, dict):
                return "Any"
            return _orig_get_type(schema)

        client_utils.get_type = get_type
        client_utils._shawaf_schema_patched = True
    except ImportError:
        pass


def main() -> None:
    _patch_gradio_client_schema()
    if _resolve_ffmpeg_binary() is None:
        print(
            "Note: pip install imageio-ffmpeg for H.264 tracklet previews "
            "(source mp4v files do not play in the browser)."
        )
    demo = build_demo()
    host = os.environ.get("SHAWAF_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("SHAWAF_DEMO_PORT", "7860"))
    demo.launch(
        server_name=host,
        server_port=port,
        show_api=False,
        allowed_paths=_allowed_media_paths(),
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
