"""Gradio demo: text → tracklet retrieval with pooling modes (datasets via CLI download)."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import numpy as np

from shawaf_vlm.models.groups import models_for_group
from shawaf_vlm.models.registry import all_specs, build_encoder
from shawaf_vlm.data.local_annotations import tracklet_preview_media_path
from shawaf_vlm.retrieval.cache import index_dir, load_numpy_index
from shawaf_vlm.retrieval.datasets import tier_a_and_local_datasets, dataset_status
from shawaf_vlm.retrieval.search import top_k_tracklets

REPO_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_K = 5
# Plain-language retrieval modes. "Best moment" (query_max) is the measured best
# on Alexandria with letterboxed crops (docs/ingestion_speed.md).
MODE_LABELS = {
    "Best moment": "query_max",
    "Whole clip": "mean",
    "Blend of both": "global_local",
}
DEFAULT_MODE_LABEL = "Best moment"
BLEND_LABEL = "Blend of both"
# Recommended ingestion settings for the in-app rebuild button.
INDEX_FIT = os.environ.get("SHAWAF_INDEX_FIT", "letterbox")
INDEX_MAX_FRAMES = os.environ.get("SHAWAF_INDEX_MAX_FRAMES", "8")
EXAMPLE_QUERIES = (
    "a girl in hijab, red blouse is drinking",
    "a boy in black shirt and green pants is fighting",
    "a girl wears black shirt and blue jeans, studying on her laptop",
    "a boy in white shirt and grey jeans is on his phone",
)
ENCODER_NAMES = {
    "pe_core_l14": "Perception Encoder L/14",
    "pe_core_b16": "Perception Encoder B/16 (fast)",
    "pe_core_s16": "Perception Encoder S/16",
    "pe_core_t16": "Perception Encoder T/16",
    "siglip2": "SigLIP 2 So400m",
    "irra": "IRRA (person search)",
    "mobileclip2_b": "MobileCLIP2-B",
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
    return [d.key for d in tier_a_and_local_datasets()]


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
    for prefer in ("alexandria:all", "local:all"):
        if prefer in _dataset_choices() and _has_index(model, prefer):
            return prefer
    for dataset in _dataset_choices():
        if _has_index(model, dataset):
            return dataset
    for prefer in ("alexandria:all", "local:all"):
        if prefer in _dataset_choices():
            return prefer
    return "tvpreid:prid"


def _setup_help_markdown() -> str:
    """Terminal-side setup, shown only inside Research settings."""

    st = dataset_status()

    def mark(ok: bool) -> str:
        return "on disk" if ok else "missing"

    return "\n".join(
        [
            "| Dataset | Status |",
            "| --- | --- |",
            f"| Alexandria | {st.get('alexandria', 'missing')} |",
            f"| TVPReid | {mark(st['tvpreid'] == 'ready')} |",
            f"| GroOT-MOT17 | {mark(st['groot'] == 'ready')} |",
            f"| RSTPReid | {mark(st['rstpreid'] == 'ready')} |",
            "",
            "Prepare data from a terminal, then rebuild the index here:",
            "",
            "```",
            "python scripts/prepare_alexandria.py",
            "python scripts/download_tier_a.py",
            "```",
        ]
    )


@lru_cache(maxsize=16)
def _index_size(path_str: str, mtime: float) -> int:
    manifest = json.loads((Path(path_str) / "manifest.json").read_text(encoding="utf-8"))
    return len(manifest.get("tracklets", []))


def _index_count(model: str, dataset: str) -> int | None:
    path = index_dir(_cache_root(), model, dataset)
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    return _index_size(str(path), manifest.stat().st_mtime)


def _dataset_label(dataset: str) -> str:
    for spec in tier_a_and_local_datasets():
        if spec.key == dataset:
            return spec.label
    return dataset


def _encoder_label(model: str) -> str:
    return ENCODER_NAMES.get(model, model)


def _index_status(model: str, dataset: str) -> str:
    """One-line status chip: ready with a count, or what to do next."""

    count = _index_count(model, dataset)
    where = f"{html.escape(_dataset_label(dataset))} · {html.escape(_encoder_label(model))}"
    if count is not None:
        return (
            f'<div class="sh-status" role="status"><span class="sh-dot sh-dot--ok" aria-hidden="true"></span>'
            f"<span><strong>Ready</strong> · {count} people indexed</span>"
            f'<span class="sh-status__where">{where}</span></div>'
        )
    return (
        f'<div class="sh-status sh-status--warn" role="status"><span class="sh-dot" aria-hidden="true"></span>'
        f"<span><strong>No index yet</strong> · open Research settings and press Rebuild index</span>"
        f'<span class="sh-status__where">{where}</span></div>'
    )


_TRACK_RE = re.compile(r"^day(\d+)_(\d+)/person_(\d+)$")


def _person_label(tracklet_id: str) -> tuple[str, str]:
    """('Day 1 · Clip 3', 'Person 1') for Alexandria ids, else (id, '')."""

    match = _TRACK_RE.match(tracklet_id)
    if match:
        day, clip, person = match.groups()
        return f"Day {day} · Clip {clip}", f"Person {person}"
    return tracklet_id, ""


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


def _gallery_media_path(stored_path: str) -> str:
    preview = tracklet_preview_media_path([stored_path])
    return str(preview) if preview is not None else stored_path


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


def _media_html(path: str, name: str) -> str:
    """Video (H.264 when ffmpeg is available, else a still) or image for one card."""

    label = html.escape(name, quote=True)
    if not path or not Path(path).is_file():
        return f'<div class="sh-card__media sh-card__media--missing" role="img" aria-label="{label}">No preview</div>'
    suffix = Path(path).suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        poster = _video_jpeg_data_url(path)
        playable = _browser_h264_path(path)
        poster_attr = f' poster="{poster}"' if poster else ""
        if playable:
            src = _gradio_file_url(playable)
            return (
                f'<video class="sh-card__media" src="{src}"{poster_attr} controls muted loop playsinline '
                f'preload="metadata" aria-label="{label}"></video>'
            )
        if poster:
            return f'<img class="sh-card__media" src="{poster}" alt="{label}"/>'
        return (
            f'<video class="sh-card__media" src="{_gradio_file_url(path)}" controls muted playsinline '
            f'preload="metadata" aria-label="{label}"></video>'
        )
    if suffix in _IMAGE_SUFFIXES:
        return f'<img class="sh-card__media" src="{_gradio_file_url(path)}" alt="{label}"/>'
    return f'<div class="sh-card__media sh-card__media--missing" role="img" aria-label="{label}">No preview</div>'


def _card_html(
    rank: int,
    path: str,
    tracklet_id: str,
    score: float,
    best: float,
    detail: str = "",
) -> str:
    place, person = _person_label(tracklet_id)
    name = f"Match {rank}: {place}" + (f", {person}" if person else "")
    share = max(0.0, min(1.0, score / best)) if best > 0 else 0.0
    top = " sh-card--top" if rank == 1 else ""
    detail_html = f'<p class="sh-card__detail">{html.escape(detail)}</p>' if detail else ""
    return (
        f'<li class="sh-card{top}">'
        f'<div class="sh-card__frame">{_media_html(path, name)}'
        f'<span class="sh-card__rank" aria-hidden="true">{rank}</span></div>'
        f'<div class="sh-card__body">'
        f'<p class="sh-card__place">{html.escape(place)}</p>'
        + (f'<p class="sh-card__person">{html.escape(person)}</p>' if person else "")
        + f'<div class="sh-score" title="Cosine similarity to the query">'
        f'<span class="sh-score__bar"><span style="width:{share * 100:.0f}%"></span></span>'
        f'<span class="sh-score__value">{score:.3f}</span></div>'
        f"{detail_html}"
        f'<p class="sh-card__id">{html.escape(tracklet_id)}</p>'
        f"</div></li>"
    )


def _results_html(cards: list[str], query: str = "") -> str:
    if not cards:
        return _empty_state_html()
    heading = "Top matches" + (f" for “{html.escape(query)}”" if query else "")
    return (
        f'<section class="sh-results" aria-labelledby="sh-results-h">'
        f'<h2 id="sh-results-h" class="sh-results__title">{heading}</h2>'
        f'<p class="sh-results__hint">Ranked by similarity. The right person is in the top five about 85% of the time on Alexandria; compare clothing and actions before deciding.</p>'
        f'<ol class="sh-grid">{"".join(cards)}</ol></section>'
    )


def _empty_state_html(message: str | None = None) -> str:
    body = message or (
        "Describe a person the way you would to a colleague: clothing colours first, then "
        "accessories, then what they are doing. Press Enter or Search."
    )
    return (
        '<section class="sh-empty" aria-labelledby="sh-empty-h">'
        '<h2 id="sh-empty-h" class="sh-results__title">Search the footage</h2>'
        f'<p class="sh-results__hint">{body}</p></section>'
    )


def _gallery_html(entries: list[tuple[str, str]]) -> str:
    """Compatibility wrapper: ``(media_path, caption)`` pairs rendered as cards."""

    if not entries:
        return _empty_state_html()
    cards = [_card_html(rank, path, caption, 0.0, 0.0) for rank, (path, caption) in enumerate(entries, start=1)]
    return _results_html(cards)


def _allowed_media_paths() -> list[str]:
    """Gradio must allow serving dataset videos and cached assets."""
    roots = [
        REPO_ROOT / "datasets",
        REPO_ROOT / "Annotations",
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
        "--no-download",
        "--fit",
        INDEX_FIT,
        "--max-frames",
        str(INDEX_MAX_FRAMES),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-12:])
        return f"**Rebuild failed** (exit {proc.returncode}). Last lines:\n```\n{tail}\n```"
    count = _index_count(model, dataset)
    return f"Rebuilt · {count} people indexed with {_encoder_label(model)}."


def search_ui(
    query: str,
    mode_label: str,
    model: str,
    dataset: str,
    alpha: float,
) -> tuple[str, str]:
    """Return ``(results_html, status_html)``."""

    status = _index_status(model, dataset)
    text = (query or "").strip()
    if not text:
        return _empty_state_html("Type a description first, for example the clothes and what the person is doing."), status
    mode = MODE_LABELS.get(mode_label, "query_max")
    try:
        tracklets, _manifest = _load_tracklets(model, dataset)
    except FileNotFoundError:
        hint = _default_dataset(model)
        extra = (
            f" An index exists for <strong>{html.escape(_dataset_label(hint))}</strong>; switch the dataset in Research settings."
            if hint != dataset and _has_index(model, hint)
            else " Open Research settings and press Rebuild index (about a minute for Alexandria)."
        )
        return _empty_state_html(f"There is no index for this dataset and encoder yet.{extra}"), status
    encoder = _get_encoder(model)
    # SHAWAF_QUERY_MODE=raw|clean|template|ensemble (see shawaf_vlm/query_text.py).
    from shawaf_vlm.query_text import encode_queries

    query_mode = os.environ.get("SHAWAF_QUERY_MODE", "raw")
    qvec = np.asarray(encode_queries(encoder, [text], query_mode)[0])
    results = top_k_tracklets(qvec, tracklets, mode, k=PREVIEW_K, hybrid_alpha=alpha)
    if not results:
        return _empty_state_html("No people in this index."), status
    best = float(results[0][1])
    cards = []
    for rank, (t, fused, g_score, l_score) in enumerate(results, start=1):
        detail = ""
        # Only explain the blend when its two parts actually differ.
        if mode == "global_local" and abs(g_score - l_score) > 1e-4:
            detail = f"whole clip {g_score:.3f} · best moment {l_score:.3f}"
        cards.append(_card_html(rank, _gallery_media_path(t.media_path), t.tracklet_id, float(fused), best, detail))
    return _results_html(cards, text), status


THEME_CSS = """
:root, .dark {
  --sh-bg: #0d1117; --sh-panel: #151b23; --sh-raised: #1c232d; --sh-line: #2a323d;
  --sh-ink: #e6e9ee; --sh-ink-2: #a3adba; --sh-ink-3: #7f8a99;
  --sh-accent: #4cc2b0; --sh-accent-ink: #062a25; --sh-accent-soft: rgba(76,194,176,0.14);
  --sh-warn: #e3b341;
  --sh-mono: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
}
@media (prefers-color-scheme: light) {
  :root:not(.dark) {
    --sh-bg: #f5f6f8; --sh-panel: #ffffff; --sh-raised: #eef1f4; --sh-line: #d6dbe1;
    --sh-ink: #11161d; --sh-ink-2: #46505d; --sh-ink-3: #5b6672;
    --sh-accent: #0f7a6c; --sh-accent-ink: #ffffff; --sh-accent-soft: rgba(15,122,108,0.10);
    --sh-warn: #9a6700;
  }
}
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; padding: 24px 24px 48px !important; }
footer { display: none !important; }
::selection { background: var(--sh-accent-soft); color: var(--sh-ink); }
.sh-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin: 0 0 4px; }
.sh-head h1 { font-size: 1.375rem; font-weight: 650; letter-spacing: -0.01em; margin: 0; color: var(--sh-ink); }
.sh-head h1 span { color: var(--sh-ink-3); font-weight: 450; }
.sh-head p { margin: 0; color: var(--sh-ink-2); font-size: 0.875rem; }
.sh-status { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; font-size: 0.875rem; color: var(--sh-ink-2); }
.sh-status strong { color: var(--sh-ink); font-weight: 600; }
.sh-status__where { color: var(--sh-ink-3); }
.sh-status__where::before { content: "·"; margin-right: 10px; }
.sh-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--sh-warn); flex: none; }
.sh-dot--ok { background: var(--sh-accent); }
#sh-search { gap: 8px !important; align-items: stretch !important; }
#sh-query textarea, #sh-query input { font-size: 1rem !important; padding: 12px 14px !important; }
#sh-search button { min-height: 48px !important; }
.sh-results__title { font-size: 1rem; font-weight: 600; color: var(--sh-ink); margin: 8px 0 4px; }
.sh-results__hint { font-size: 0.875rem; color: var(--sh-ink-2); margin: 0 0 16px; max-width: 72ch; }
.sh-empty { padding: 24px 0 8px; }
.sh-grid { list-style: none; margin: 0; padding: 0; display: grid; gap: 16px;
  grid-template-columns: repeat(5, minmax(0, 1fr)); }
.sh-card { background: var(--sh-panel); border: 1px solid var(--sh-line); border-radius: 10px; overflow: hidden;
  display: flex; flex-direction: column; transition: border-color 160ms ease-out; }
.sh-card:hover { border-color: var(--sh-ink-3); }
.sh-card--top { border-color: var(--sh-accent); box-shadow: 0 6px 18px -10px rgba(76,194,176,0.45); }
.sh-card__frame { position: relative; aspect-ratio: 3 / 5; background: #05070a; overflow: hidden; }
.sh-card__frame .sh-card__media { position: absolute; inset: 0; width: 100% !important; height: 100% !important;
  max-width: none !important; max-height: none !important; margin: 0 !important; object-fit: contain; display: block; background: #05070a; }
.sh-card__media--missing { display: grid; place-items: center; color: var(--sh-ink-3); font-size: 0.8125rem; }
.sh-card__rank { position: absolute; top: 8px; left: 8px; min-width: 28px; height: 28px; padding: 0 8px;
  border-radius: 14px; display: grid; place-items: center; font: 600 0.875rem/1 var(--sh-mono);
  background: rgba(13,17,23,0.82); color: #e6e9ee; font-variant-numeric: tabular-nums; }
.sh-card--top .sh-card__rank { background: var(--sh-accent); color: var(--sh-accent-ink); }
.sh-card__body { padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 4px; }
.sh-card__place { margin: 0; font-size: 0.9375rem; font-weight: 600; color: var(--sh-ink); }
.sh-card__person { margin: 0; font-size: 0.875rem; color: var(--sh-ink-2); }
.sh-card__detail { margin: 2px 0 0; font-size: 0.8125rem; color: var(--sh-ink-2); }
.sh-card__id { margin: 4px 0 0; font: 0.75rem/1.4 var(--sh-mono); color: var(--sh-ink-3); word-break: break-all; }
.sh-score { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
.sh-score__bar { flex: 1; height: 4px; border-radius: 2px; background: var(--sh-raised); overflow: hidden; }
.sh-score__bar > span { display: block; height: 100%; background: var(--sh-accent); border-radius: 2px; }
.sh-score__value { font: 500 0.8125rem/1 var(--sh-mono); color: var(--sh-ink); font-variant-numeric: tabular-nums; }
#sh-settings { margin-top: 28px !important; }
#sh-settings .prose table { font-size: 0.875rem; }
@media (max-width: 1080px) { .sh-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) {
  .gradio-container { padding: 16px 16px 40px !important; }
  .sh-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
  #sh-search { flex-wrap: wrap !important; }
  #sh-search > :first-child { flex: 1 1 100% !important; min-width: 100% !important; }
  .sh-status__where { flex-basis: 100%; }
  .sh-status__where::before { content: none; }
}
@media (prefers-reduced-motion: reduce) { .sh-card { transition: none; } }
"""


def _theme():
    import gradio as gr

    sans = ["ui-sans-serif", "Segoe UI", "system-ui", "-apple-system", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
    mono = ["ui-monospace", "Cascadia Mono", "SFMono-Regular", "Consolas", "monospace"]
    return gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_md,
        font=sans,
        font_mono=mono,
    ).set(
        body_background_fill="#f5f6f8",
        body_background_fill_dark="#0d1117",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#151b23",
        background_fill_secondary="#eef1f4",
        background_fill_secondary_dark="#1c232d",
        block_background_fill="#ffffff",
        block_background_fill_dark="#151b23",
        block_border_color="#d6dbe1",
        block_border_color_dark="#2a323d",
        border_color_primary="#d6dbe1",
        border_color_primary_dark="#2a323d",
        body_text_color="#11161d",
        body_text_color_dark="#e6e9ee",
        body_text_color_subdued="#5b6672",
        body_text_color_subdued_dark="#a3adba",
        block_label_text_color="#46505d",
        block_label_text_color_dark="#a3adba",
        input_background_fill="#ffffff",
        input_background_fill_dark="#0d1117",
        input_border_color="#c3cad3",
        input_border_color_dark="#3a4350",
        input_border_color_focus="#0f7a6c",
        input_border_color_focus_dark="#4cc2b0",
        button_primary_background_fill="#0f7a6c",
        button_primary_background_fill_dark="#4cc2b0",
        button_primary_background_fill_hover="#0c6559",
        button_primary_background_fill_hover_dark="#6fd3c4",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#062a25",
        button_secondary_background_fill="#eef1f4",
        button_secondary_background_fill_dark="#1c232d",
        button_secondary_text_color="#11161d",
        button_secondary_text_color_dark="#e6e9ee",
        color_accent_soft="rgba(15,122,108,0.10)",
        color_accent_soft_dark="rgba(76,194,176,0.14)",
        link_text_color="#0f7a6c",
        link_text_color_dark="#4cc2b0",
        slider_color="#0f7a6c",
        slider_color_dark="#4cc2b0",
    )


def build_demo():
    import gradio as gr

    default_model = _default_model()
    default_dataset = _default_dataset(default_model)
    with gr.Blocks(title="SHAWAF · Person search", theme=_theme(), css=THEME_CSS) as demo:
        gr.HTML(
            '<header class="sh-head"><h1>SHAWAF <span>Person search</span></h1>'
            "<p>Describe someone; get the five closest people from the indexed footage.</p></header>"
        )
        status = gr.HTML(_index_status(default_model, default_dataset))

        with gr.Row(elem_id="sh-search", equal_height=True):
            query = gr.Textbox(
                elem_id="sh-query",
                show_label=False,
                placeholder="e.g. boy in black shirt, green pants and glasses, fighting",
                lines=1,
                max_lines=1,
                scale=8,
                autofocus=True,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1, min_width=120)
            clear_btn = gr.Button("Clear", variant="secondary", scale=0, min_width=88)

        gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=[query], label="Try an example")
        results = gr.HTML(_empty_state_html())

        with gr.Accordion("Research settings", open=False, elem_id="sh-settings"):
            with gr.Row():
                dataset = gr.Dropdown(
                    _dataset_choices(), value=default_dataset, label="Dataset", scale=1
                )
                model = gr.Dropdown(
                    [(_encoder_label(k), k) for k in _model_choices()],
                    value=default_model,
                    label="Encoder",
                    scale=1,
                )
                mode = gr.Radio(
                    list(MODE_LABELS.keys()),
                    value=DEFAULT_MODE_LABEL,
                    label="Match on",
                    info="Best moment scores each person's single best window; whole clip averages all of them.",
                    scale=1,
                )
            alpha = gr.Slider(
                0,
                1,
                value=0.5,
                step=0.05,
                label="Blend: weight on whole clip",
                visible=False,
            )
            with gr.Row():
                build_btn = gr.Button("Rebuild index", variant="secondary", scale=0, min_width=160)
                build_log = gr.Markdown(
                    f"Rebuild uses padded crops and {INDEX_MAX_FRAMES} frames per person "
                    "(the measured best settings). Takes about a minute for Alexandria."
                )
            gr.Markdown(_setup_help_markdown())

        def on_mode_change(label: str):
            return gr.update(visible=(label == BLEND_LABEL))

        def refresh_status(model_key: str, dataset_key: str):
            return _index_status(model_key, dataset_key)

        def clear():
            return "", _empty_state_html()

        mode.change(on_mode_change, inputs=mode, outputs=alpha)
        model.change(refresh_status, inputs=[model, dataset], outputs=status)
        dataset.change(refresh_status, inputs=[model, dataset], outputs=status)
        build_btn.click(build_index_ui, inputs=[model, dataset], outputs=build_log).then(
            refresh_status, inputs=[model, dataset], outputs=status
        )
        search_inputs = [query, mode, model, dataset, alpha]
        search_btn.click(search_ui, inputs=search_inputs, outputs=[results, status], show_progress="minimal")
        query.submit(search_ui, inputs=search_inputs, outputs=[results, status], show_progress="minimal")
        clear_btn.click(clear, outputs=[query, results])
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
