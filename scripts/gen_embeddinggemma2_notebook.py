"""One-off generator for notebooks/embeddinggemma2_tier_a_zero_shot.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "embeddinggemma2_tier_a_zero_shot.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


RUN_EVAL = r'''
import gc
import json
import traceback
from datetime import datetime, timezone

from shawaf_vlm.eval_loop import evaluate_text_to_tracklet, evaluate_text_to_tracklet_windows
from shawaf_vlm.models import build_encoder
from shawaf_vlm.models.runtime import ensure_cuda_healthy

METRIC_KEYS = [
    "Rank-1", "Rank-5", "Rank-10", "Rank-20", "Rank-50",
    "mAP", "MdR", "MnR", "nDCG@10", "mINP",
    "num_valid_queries", "num_gallery", "num_clips",
    "decode_s", "video_s", "text_s", "score_s", "total_s",
    "video_ms_per_item", "peak_gpu_gb", "reserved_gpu_gb",
]


class StopModel(Exception):
    pass


class GalleryCache:
    def __init__(self, encoder):
        self.encoder = encoder
        self._key = None
        self._features = None
        self.last_hit = False

    def encode_videos(self, videos, batch_size=4):
        key = tuple(tuple(str(path) for path in clip) for clip in videos)
        if key == self._key:
            self.last_hit = True
            return self._features
        self.last_hit = False
        self._features = self.encoder.encode_videos(videos, batch_size=batch_size)
        self._key = key
        return self._features

    def encode_texts(self, texts, batch_size=32):
        return self.encoder.encode_texts(texts, batch_size=batch_size)

    def __getattr__(self, name):
        return getattr(self.encoder, name)


def family(model):
    return "s2" if model in NATIVE_FRAMES else "clip8"


def protocols_for(model, dataset):
    kind = DATASETS[dataset]["protocols"]
    specs = []
    for spec in SINGLE_PROTOCOLS:
        if spec["family"] != family(model):
            continue
        if kind == "image" and spec.get("video_only"):
            continue
        specs.append({**spec, "type": "single"})
    if kind == "video":
        specs.extend({**spec, "type": "window"} for spec in WINDOW_PROTOCOLS)
    return specs


def sampling_text(model, spec):
    if spec["type"] == "single":
        return f"{spec['frame_sample']} {spec['num_frames']} frames"
    return (
        f"sliding windows of {NATIVE_FRAMES.get(model, 8)}, stride {spec['stride']}, "
        f"{spec['sample_fps']:g} fps, cap {spec['max_frames']} frames"
    )


def result_path(dataset, model, subset, protocol, pool):
    return RESULTS_DIR / dataset / f"{model}_{subset}_{protocol}_{pool}.json"


def expected_paths(dataset, model, subset, spec):
    pools = ["none"] if spec["type"] == "single" else list(POOLS)
    return [result_path(dataset, model, subset, spec["name"], pool) for pool in pools]


def is_complete(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except Exception:
        return False


def save_row(dataset, model, subset, spec, pool, metrics=None, error=None, cached=False):
    row = {
        "group": ENV_GROUP,
        "dataset": dataset,
        "subset": subset,
        "model": model,
        "protocol": spec["name"],
        "protocol_type": spec["type"],
        "pool": pool,
        "hybrid_alpha": HYBRID_ALPHA if pool == "global_local" else None,
        "sampling": sampling_text(model, spec),
        "gallery_cached": bool(cached),
        "status": "failed" if error else "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metrics is not None:
        row.update({key: float(metrics[key]) for key in METRIC_KEYS if key in metrics})
    if error is not None:
        row["error"] = str(error)[:2000]
    path = result_path(dataset, model, subset, spec["name"], pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")


def clean_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_one(encoder, model, dataset, subset, spec):
    paths = expected_paths(dataset, model, subset, spec)
    if all(is_complete(path) for path in paths):
        print(f"skip {model} {dataset}/{subset} {spec['name']}", flush=True)
        return
    splits = load_split(dataset, subset)
    cache = WORK / "frame_cache" / dataset
    batch = MODEL_BATCH.get(model, BATCH)
    text_batch = MODEL_TEXT_BATCH.get(model, TEXT_BATCH)
    print(
        f"== {PRETTY.get(model, model)} | {dataset}/{subset} | {spec['name']} ({sampling_text(model, spec)})",
        flush=True,
    )
    try:
        if spec["type"] == "single":
            scored = {
                "none": evaluate_text_to_tracklet(
                    encoder,
                    splits,
                    num_frames=spec["num_frames"],
                    batch_size=batch,
                    text_batch_size=text_batch,
                    junk_same_camera=False,
                    frame_cache=cache,
                    frame_sample=spec["frame_sample"],
                )
            }
        else:
            scored = evaluate_text_to_tracklet_windows(
                encoder,
                splits,
                num_frames=NATIVE_FRAMES.get(model, 8),
                stride=spec["stride"],
                sample_fps=spec["sample_fps"],
                max_frames=spec["max_frames"],
                pools=POOLS,
                batch_size=batch,
                text_batch_size=text_batch,
                junk_same_camera=False,
                frame_cache=cache,
                hybrid_alpha=HYBRID_ALPHA,
            )
    except RuntimeError as exc:
        pools = ["none"] if spec["type"] == "single" else list(POOLS)
        for pool in pools:
            save_row(dataset, model, subset, spec, pool, error=exc)
        if "out of memory" in str(exc).lower():
            raise StopModel(str(exc)) from exc
        raise
    for pool, metrics in scored.items():
        save_row(dataset, model, subset, spec, pool, metrics=metrics, cached=encoder.last_hit)
        print(
            f"-- {pool}: R1 {metrics['Rank-1']:.2f}  mAP {metrics['mAP']:.2f}  mINP {metrics.get('mINP', float('nan')):.2f}",
            flush=True,
        )


ensure_cuda_healthy(DEVICE)

for model in MODELS:
    jobs = [
        (dataset, spec, subset)
        for dataset in RUN_DATASETS
        for spec in protocols_for(model, dataset)
        for subset in DATASETS[dataset]["subsets"]
    ]
    todo = [
        job
        for job in jobs
        if not all(is_complete(p) for p in expected_paths(job[0], model, job[2], job[1]))
    ]
    print(f"\n######## {PRETTY.get(model, model)}: {len(todo)} of {len(jobs)} runs to do", flush=True)
    if not todo:
        continue
    try:
        encoder = GalleryCache(build_encoder(model, device=DEVICE))
    except Exception:
        traceback.print_exc()
        print(f"Could not build {model}; skipping it", flush=True)
        clean_cuda()
        continue
    try:
        for dataset, spec, subset in todo:
            run_one(encoder, model, dataset, subset, spec)
    except StopModel as exc:
        print(f"CUDA OOM on {model}; stopping: {exc}", flush=True)
    finally:
        del encoder
        clean_cuda()

print("Done", flush=True)
'''

SUMMARY = r'''
import json
from pathlib import Path

import pandas as pd
from IPython.display import display

rows = []
for path in sorted(RESULTS_DIR.rglob("*.json")):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if row.get("status") != "complete":
        continue
    rows.append(row)

if not rows:
    print("No complete results yet under", RESULTS_DIR)
else:
    df = pd.DataFrame(rows)
    cols = ["dataset", "subset", "protocol", "pool", "Rank-1", "mAP", "mINP"]
    display(df[cols].sort_values(["dataset", "subset", "protocol", "pool"]))
'''

cells = [
    md(
        """# EmbeddingGemma 2 — Tier-A zero-shot benchmark

Same protocol as `kaggle-zero-shot.ipynb`: **TVPReid** (prid, ilids, duke), **RSTPReid** (test), **GroOT-MOT17** (all, appearance, action, combined).

| Item | Value |
|---|---|
| Model | `google/embeddinggemma-2` |
| Text | SearchQuery / encode_query |
| Video | uniform8 (RSTP) + `reid_8fps_n64` sliding windows |
| Pools | mean, mean_s8, max, query_max, global_local |

Use a **separate** Python env from the `base` Kaggle group: `pip install -e ".[embeddinggemma2]"`. Restart the kernel if an older `transformers` was already imported.

Results: `zero_shot_results/embeddinggemma2/`."""
    ),
    md("## 1. Configuration"),
    code(
        """from pathlib import Path

ENV_GROUP = "embeddinggemma2"
MODELS = ["embeddinggemma2"]
RUN_DATASETS = ["tvpreid", "rstpreid", "groot_mot17"]

DATASETS = {
    "tvpreid": {"label": "TVPReid", "subsets": ["prid", "ilids", "duke"], "protocols": "video"},
    "rstpreid": {"label": "RSTPReid", "subsets": ["test"], "protocols": "image"},
    "groot_mot17": {
        "label": "GroOT-MOT17",
        "subsets": ["all", "appearance", "action", "combined"],
        "protocols": "video",
    },
}
SPLIT = "test"
DEVICE = "cuda"
BATCH = 1
TEXT_BATCH = 8
HYBRID_ALPHA = 0.5
POOLS = ("mean", "mean_s8", "max", "query_max", "global_local")

NATIVE_FRAMES = {}
MODEL_BATCH = {"embeddinggemma2": 1}
MODEL_TEXT_BATCH = {"embeddinggemma2": 8}

SINGLE_PROTOCOLS = [
    {"name": "uniform8", "num_frames": 8, "frame_sample": "uniform", "family": "clip8"},
]
WINDOW_PROTOCOLS = [
    {"name": "reid_8fps_n64", "sample_fps": 8.0, "max_frames": 64, "stride": 4},
]

PRETTY = {"embeddinggemma2": "EmbeddingGemma 2"}

ON_KAGGLE = Path("/kaggle/working").is_dir()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("results")
RESULTS_DIR = WORK / "zero_shot_results" / ENV_GROUP
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("Models", MODELS, "datasets", RUN_DATASETS, "results", RESULTS_DIR, flush=True)"""
    ),
    md("## 2. Setup"),
    code(
        """import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

SHAWAF_URL = "https://github.com/BASSAT-BASSAT/Benchmarking-Video-Image-language-models-for-Tracklet-retrieval-.git"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call([str(p) for p in cmd], cwd=str(cwd) if cwd else None)


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


REPO = WORK / "shawaf_vlm_repo"
if (REPO / "pyproject.toml").is_file():
    run(["git", "pull", "--ff-only"], cwd=REPO)
elif REPO.exists():
    run(["git", "pull", "--ff-only"], cwd=REPO)
else:
    run(["git", "clone", "--depth", "1", SHAWAF_URL, str(REPO)])

if not ON_KAGGLE:
    run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "--index-url", TORCH_INDEX])
run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[embeddinggemma2]"])
# Hub: ST>=6.1, transformers>=5.19. Kaggle wheels often break TF/ST import (ABI mismatch).
for _pkg in ("torchcodec", "torchaudio", "sentence-transformers"):
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", _pkg])
run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "sentence-transformers>=6.1.0,<7.0",
        "transformers>=5.19.0,<6.0",
        "accelerate>=0.30",
    ]
)

if "transformers" in sys.modules or "sentence_transformers" in sys.modules:
    raise RuntimeError("Restart the kernel, then run from the top (transformers/ST already imported).")

sys.path.insert(0, str(REPO))
from shawaf_vlm.models.sentence_transformers_compat import (
    assert_embeddinggemma2_runtime,
    prepare_embeddinggemma2_import_env,
)

prepare_embeddinggemma2_import_env()
import shawaf_vlm

assert_embeddinggemma2_runtime()
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

if "embedding_gemma2" not in CONFIG_MAPPING:
    raise RuntimeError(
        "transformers is too old for google/embeddinggemma-2; need >=5.19.0. "
        "Re-run setup: pip install --upgrade 'transformers>=5.19.0,<6'"
    )
from sentence_transformers import SentenceTransformer  # noqa: F401 — smoke test

print(
    "shawaf_vlm",
    shawaf_vlm.__version__,
    "transformers",
    installed("transformers"),
    "sentence-transformers",
    installed("sentence-transformers"),
    flush=True,
)

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if ON_KAGGLE and not token:
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("hugging_face")
    except Exception as exc:
        print("No hugging_face secret:", type(exc).__name__, flush=True)
if token:
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    print("HF token loaded", flush=True)"""
    ),
    md("## 3. Datasets"),
    code(
        """import pandas as pd
from IPython.display import display

from shawaf_vlm.data import (
    discover_local_groot,
    download_groot,
    download_rstpreid,
    download_tvpreid,
    load_groot,
    load_rstpreid,
    load_tvpreid,
)

DATA_ROOTS = {}
if "tvpreid" in RUN_DATASETS:
    DATA_ROOTS["tvpreid"] = download_tvpreid(configs=tuple(DATASETS["tvpreid"]["subsets"]), split=SPLIT)
if "rstpreid" in RUN_DATASETS:
    DATA_ROOTS["rstpreid"] = download_rstpreid()
if "groot_mot17" in RUN_DATASETS:
    DATA_ROOTS["groot_mot17"] = discover_local_groot() or download_groot(
        configs=tuple(DATASETS["groot_mot17"]["subsets"]), split=SPLIT
    )


def load_split(dataset, subset):
    root = DATA_ROOTS[dataset]
    if dataset == "tvpreid":
        return load_tvpreid(subset, split=SPLIT, root=root)
    if dataset == "rstpreid":
        return load_rstpreid(root, split=subset)
    if dataset == "groot_mot17":
        return load_groot(subset, split=SPLIT, root=root)
    raise ValueError(dataset)

summary = []
for dataset in RUN_DATASETS:
    for subset in DATASETS[dataset]["subsets"]:
        splits = load_split(dataset, subset)
        per_id = pd.Series([g.person_id for g in splits.gallery]).value_counts()
        summary.append(
            {
                "dataset": DATASETS[dataset]["label"],
                "subset": subset,
                "queries": len(splits.query),
                "gallery": len(splits.gallery),
                "identities": int(per_id.size),
            }
        )
display(pd.DataFrame(summary))"""
    ),
    md("## 4. Run"),
    code(RUN_EVAL),
    md("## 5. Summary table"),
    code(SUMMARY),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", OUT)
