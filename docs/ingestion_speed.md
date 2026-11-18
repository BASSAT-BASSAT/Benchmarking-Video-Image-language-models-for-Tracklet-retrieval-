# Ingestion speed and accuracy

This page records what was measured, what changed in the code, and how to
reproduce it. All runs: RTX 5070 Laptop GPU (8 GB), fp16, Windows, no
`torch.compile` (Triton is not installed).

## Benchmarks used

| Key | What | Size | Notes |
|---|---|---|---|
| `alexandria:all` | Our campus footage (46 clips, 5 days), manual boxes, one caption per person | 131 tracklets, 131 queries, about 60 crops each at 10 fps | Private test set. `scripts/prepare_alexandria.py` builds it. One query is 0.76 Rank-1 points; standard error is about ±4 points |
| `groot:combined` / `appearance` / `action` | GroOT-MOT17 tracks, our Hugging Face mirror | 454 tracks | Public. Our pipeline reproduces the earlier Kaggle number exactly (PE-L combined 29.36) |

**Timing caveat.** Most sweeps ran on battery or with several jobs sharing the
GPU. The GPU was power-capped (throttle reason `0x4`). Use the accuracy columns
from `results/ingestion/*.csv`; use the clean forward-pass table below for speed.

## Where the time goes

PE-L/14 baseline on Alexandria, 64 frames per tracklet, 232 s wall:

| Stage | Share |
|---|---:|
| Vision forward | 83 % |
| Preprocess | 6 % |
| Model load | 6 % |
| Image load | 4 % |

The forward pass dominates. Pipeline fixes save at most about 15 %; the real
levers are fewer frames and smaller models.

Forward cost per frame (fp16, batch 64, measured alone on the GPU):

| Model | Input | Vision params | ms per frame |
|---|---:|---:|---:|
| PE-L/14 | 336 | 317 M | 17.0 |
| PE-L/14 | 224 | 317 M | 9.5 |
| PE-B/16 | 224 | 94 M | 1.65 |
| PE-S/16 | 384 | 24 M | 2.3 |
| PE-T/16 | 384 | 6 M | 0.9 |

## Clean ingestion timings (AC power, GPU otherwise idle)

All 131 Alexandria people; `index_qdrant.py --profile`; ingest time excludes model load (11–18 s).

| Setting | Frames | Ingest | People/s | Rank-1 |
|---|---:|---:|---:|---:|
| PE-L, stretched, 64 frames (old) | 7,860 | 130.9 s | 1.0 | 45.0 |
| PE-L, padded, 64 frames | 7,860 | 128.3 s | 1.0 | 52.7 |
| **PE-L, padded, 16 frames (new default)** | 2,096 | **38.7 s** | 3.4 | 51.2 |
| PE-B, padded, 16 frames | 2,096 | 6.8 s | 19.1 | 45.0 (Rank-5 80.9, mAP 58.4) |
| **PE-B, stretched, 16 frames (fast option)** | 2,096 | ~6.8 s | ~19 | **46.6 (Rank-5 80.2, mAP 60.4)** |

Padding does not help PE-B (224 px input leaves too few pixels on the person);
use its native stretched crops. PE-B is about 6× faster than PE-L and loses
about 4.5 Rank-1 and 3 Rank-5 points.

The threaded loader cut image load + preprocess from about 23 s to about 2 s.

## Accuracy on Alexandria

Rank-1 / mAP. "padded" = `--fit letterbox`; `/n8` = 8 frames per tracklet (default 64).

| Config | R1 `query_max` | R1 `global_local` | mAP `query_max` |
|---|---:|---:|---:|
| `pe_core_l14` (stretched, 64 frames; old default) | 45.0 | 46.6 | 62.4 |
| `pe_core_l14/n8` | 46.6 | 46.6 | 62.5 |
| **`pe_core_l14:letterbox`** | **52.7** | 51.9 | **66.1** |
| `pe_core_l14:pad2` | 49.6 | 48.9 | 64.4 |
| `pe_core_l14:letterbox/n8` | 47.3 | 47.3 | 63.6 |
| **`pe_core_l14:letterbox/n16`** (recommended) | **51.2** | 51.2 | **65.2** |
| `pe_core_l14:letterbox/n32` | 51.9 | 51.9 | 65.5 |
| `pe_core_b16` | 47.3 | 47.3 | — |
| `pe_core_s16` | — | 36.6 | — |
| `pe_core_t16` | — | 26.7 | — |
| `mobileclip2_b` / `:letterbox` | — | 34.3 / 42.0 | — |
| `siglip2` (old setup) | 46.6 | 47.3 | 61.9 |
| `siglip2:letterbox/n8` | 51.2 | 51.2 | 63.5 |
| `siglip2:letterbox` | 47.3 | 49.6 | 61.4 |
| `embeddinggemma2` (old setup) | 47.3 | 47.3 | 59.8 |
| `embeddinggemma2:letterbox/n8` | 42.7 | 42.7 | 59.9 |
| `embeddinggemma2:letterbox` | 43.5 | 45.8 | 60.7 |
| `pe_core_l14:letterbox/n8+quality` | **52.7** | 52.7 | 65.2 |
| `pe_core_l14:letterbox/n8+diverse` | 48.1 | 48.1 | 63.1 |
| `pe_core_l14:letterbox/n16+quality` / `+diverse` | 49.6 / 50.4 | 50.4 / 49.6 | 64.3 / 64.3 |
| `dfn_clip_l14` → `:letterbox` | 34.4 → 41.2 | 34.4 → 42.0 | 47.3 → 55.9 |
| `eva02_clip_l14` → `:letterbox` | 20.6 → 30.5 | 20.6 → 31.3 | 36.2 → 46.4 |
| `mobileclip2_s2`, `mobileclip2_s4` | invalid | | Fail a zero-shot sanity check under open_clip 3.3 / timm 1.0.27 |

Findings:

1. **Padding crops to a square helps most** (PE-L +6 Rank-1, +3.7 mAP). The
   gain grows with how much of the crop's shape is kept: stretched < `pad2` <
   letterbox. Person crops are about 1:2.5; stretching them distorts shapes such
   as a hijab, a bag strap or glasses.
2. **With stretched crops, 8 frames match 64. With padded crops, use 16.**
   PE-L padded: 47.3 / 51.2 / 51.9 / 52.7 Rank-1 at 8 / 16 / 32 / 64 frames.
   Sixteen frames keep almost all of the gain at a quarter of the work.
   Padding is a dependable gain for PE-L only; SigLIP 2 and EmbeddingGemma 2
   move within noise.
3. **Query text cleanup does nothing** (identical to raw). Prompt templates move
   Rank-1 up but Rank-5 down, with flat mAP. The app keeps raw query text.
4. **`query_max` pooling is best with padded crops.**

## Accuracy on GroOT (PE-L/14)

| Subset | Stretched, 64 frames | Padded, 64 frames | Padded, 8 frames |
|---|---:|---:|---:|
| combined R1 / mAP | 29.36 / 39.45 | **30.81 / 40.83** | 29.94 / 38.97 |
| appearance R1 / mAP | 26.16 / 37.10 | **26.45 / 37.51** | 25.87 / — |
| action R1 | 2.86 | 3.30 | 3.08 |

Other models on GroOT (best pooling, R1 / mAP):

| Model | Subset | Stretched / native | Padded, 64 frames | Padded, 8 frames |
|---|---|---|---|---|
| SigLIP 2 | combined | 27.9 / 38.9 | **29.7 / 40.8** | 26.5 / 37.0 |
| SigLIP 2 | appearance | 25.6 / 35.1 | **27.0 / 37.9** | 24.1 / 34.4 |
| SigLIP 2 | action | 2.2 | 2.6 | 2.2 |
| DFN-CLIP-L | combined | 11.3 / 19.0 | **18.6 / 28.5** | — |
| EVA02-CLIP-L | combined | 13.4 / 20.6 | **20.9 / 28.9** | — |

With 64 frames, padding helps every model on GroOT; DFN and EVA02 (whose native
preprocessing centre-crops away the head and feet) gain 7–8 Rank-1. With 8
frames the gain disappears. Action captions sit near 2–3 % for every encoder.

## Failure analysis (Alexandria, PE-L padded, `query_max`)

`scripts/failure_analysis.py` writes `per_query.csv`, `summary.json` and contact
sheets to `results/failure_analysis/`.

| Of 62 misses | Count |
|---|---:|
| Top-1 has an identical caption | 1 |
| Top-1 caption is a near-duplicate (text cosine ≥ 0.90) | 4 |
| Confused with another person in the same clip | 3 |
| Top-1 is from a different clip | 54 |

The contact sheets show that many "different clip" misses are **the same actor
in the same clothes, filmed in another clip the same day** (for example the boy
in the black shirt and green trousers appears in `day4_28`, `38`, `41` and `42`).
Identity labels are per clip, so these count as errors. Counting same-day
look-alikes as correct puts Rank-1 at about 57–60. The fix is global identity
labels, not a model change.

## Code changes

| Area | Change |
|---|---|
| Profiler (`shawaf_vlm/profiling.py`) | CUDA sync at stage boundaries, true wall time, new stages `model_load`, `decode`, `pool`, `save` |
| Loader (`frame_encoders.py`) | Thread-pool prefetch replaces the per-batch DataLoader; pinned memory, non-blocking copies, OOM splitting without re-reading images |
| Preprocessing | `rebuild_preprocess` keeps each model's own interpolation and normalisation; `--fit native\|squash\|letterbox\|pad2`; `make_prefit` pads crops for encoders with their own processor (SigLIP 2, EmbeddingGemma 2) |
| PE-Core resize | `resize_timm_trunk` makes `--image-size` work (resamples the position table, rebuilds the RoPE cache). Before this fix, `--image-size 224` crashed |
| `torch.compile` | Skipped without Triton; reverts to eager mode if the first compiled call fails |
| Frame selection | `--select uniform\|quality\|diverse` (`shawaf_vlm/frame_select.py`); implemented, not yet benchmarked |
| Query text | `--query-modes raw,clean,template,ensemble` (`shawaf_vlm/query_text.py`); the app reads `SHAWAF_QUERY_MODE` |
| Qdrant | One collection per model (fixes the dimension clash), bulk numpy upload, int8 scalar quantisation with rescoring, `query_points` (qdrant-client 1.19 removed `search`) |
| Datasets | `alexandria:<config>` loader; GroOT discovery skips partial downloads |
| New models | `eva02_clip_l14`, `dfn_clip_l14` (group `clip_large`) |
| Demo app | Redesigned (search-first, result cards, Research settings panel); rebuild uses padded crops |

## Environments

- `crowd-gpu` (main): torch 2.12 nightly cu128, transformers 4.48. Everything except EmbeddingGemma 2.
- `gemma-gpu`: a clone of `crowd-gpu` with transformers 5.23 and sentence-transformers 6.3, for `embeddinggemma2` only.

## Reproduce

```bash
python scripts/prepare_alexandria.py
python scripts/ingestion_sweep.py pe_core_l14 pe_core_l14:letterbox pe_core_l14:letterbox/n8 --dataset alexandria:all
python scripts/ingestion_sweep.py pe_core_l14 pe_core_l14:letterbox --dataset groot:combined --out results/ingestion/groot_sweep.csv
python scripts/plot_pareto.py
python scripts/failure_analysis.py
```

Config syntax for the sweep: `model[@size][:fit][/nFRAMES][+select]`, for
example `pe_core_b16@224:letterbox/n16+diverse`.

## Lightweight improvements tested (cleaned Alexandria, 129 people)

All at the same or lower ingestion and search cost; settings fixed in advance
(10 % margin, weight 0.5). Hard Rank-1 / meaning-based Rank-1
(`scripts/rescore.py`, `results/rescore/summary.csv`).

| Model (16 frames) | Baseline | Larger crops (10 % margin) | Full + 0.5·action | 0.5·appearance + 0.5·action | 8 evenly spaced | Sharpest 8 |
|---|---|---|---|---|---|---|
| PE-L, padded | **51.2 / 59.7** | 48.1 / 56.6 | 47.3 / 54.3 | 48.8 / 56.6 | 48.8 / 55.8 | 54.3 / 62.8 |
| PE-B | **47.3 / 55.0** | 42.6 / 51.2 | 41.1 / 48.8 | 38.0 / 45.7 | 47.3 / 54.3 | 41.9 / 49.6 |
| SigLIP 2 | **51.2 / 55.8** | 50.4 / 55.0 | 39.5 / 44.2 | 41.1 / 48.1 | 48.1 / 53.5 | 51.9 / 56.6 |
| EmbeddingGemma 2 | 47.3 / 51.9 | **48.8 / 53.5** | 41.9 / 47.3 | 35.7 / 41.1 | 45.7 / 51.2 | 43.4 / 48.8 |

GroOT combined, hard Rank-1: PE-L 29.7 (16) / 29.9 (8) / 26.7 (sharpest 8);
PE-B 24.4 / 26.2 / 25.3.

None of the three ideas helps consistently. Query weighting hurts every model;
larger crops help only EmbeddingGemma 2; sharpest-8's Alexandria gain for PE-L
does not hold on GroOT. Selecting crops also costs more than encoding them for
PE-B (0.093 vs 0.059 s per person). Eight evenly spaced frames match 16 on
GroOT at half the cost.

## Not done yet

- Re-time throughput on AC power with nothing else on the GPU.
- ONNX / TensorRT (FP16, FP8, INT8) runtime: export code exists in `shawaf_vlm/optim/`, no runtime encoder yet.
- Benchmark `--select quality|diverse`.
- Global identity labels for Alexandria.
- Fine-tuning on public person-description data: see `docs/finetuning_plan.md`.
