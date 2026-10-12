# SHAWAF Stage 1 — repository architecture

## Package layout (`shawaf_vlm/`)

| Module | Responsibility |
|--------|----------------|
| `data/` | Dataset loaders (TVPReid, RSTPReid, GroOT, TV-MARS stubs), Hub download helpers |
| `models/` | Encoder registry, per-model wrappers, LanguageBind vendored config |
| `sampling.py` | Frame windows, FPS sampling, uniform repeats for image “tracklets” |
| `pooling.py` | Tracklet pooling: `mean`, `max`, `query_max`, `global_local` (Method C) |
| `metrics.py` | Rank-k, mAP, nDCG for text→gallery retrieval |
| `eval_loop.py` | Windowed encode + pool + score (benchmark core) |
| `cli.py` | `shawaf-vlm-eval` entry point |
| `retrieval/` | Demo/index path: encode gallery, numpy cache, Qdrant upsert, search |

## Scripts (`scripts/`)

See [scripts/README.md](../scripts/README.md). Grouped as **data**, **eval**, **demo**, **figures**.

## Demo (`demo/`)

Thin Gradio UI over `shawaf_vlm.retrieval` — no dataset downloads in the UI.

## Tests (`tests/`)

| File | Focus |
|------|--------|
| `test_datasets.py`, `test_tvpreid.py` | Loader contracts |
| `test_tv_mars.py` | Sampling, pooling, eval loop (no live transformers) |
| `test_model_compat.py` | Transformers / LanguageBind shims (`needs_transformers`) |
| `test_retrieval_cache.py`, `test_tier_a_download.py`, `test_demo_app.py` | Demo stack |

## Data flow (benchmark)

```text
Dataset splits → window clips → encoder.encode_videos → pool per tracklet
Text queries   → encoder.encode_texts → cosine rank gallery
```

## Data flow (interactive demo)

```text
scripts/download_tier_a.py → scripts/index_qdrant.py → demo/app.py search
```

Optional: Qdrant (`docker compose up`) instead of numpy-only index.
