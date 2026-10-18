# Retrieval protocol & pooling

How SHAWAF turns a **tracklet** (video clip or crop sequence) and a **text query**
into a ranked gallery. Same ideas for benchmarks (`eval_loop.py`), indexing
(`scripts/index_qdrant.py`), and the [Gradio demo](../demo/app.py).

---

## End-to-end flow

```text
                    ┌── INDEX / GALLERY (once per model + dataset) ──┐
                    │                                                 │
Tracklet on disk    │  sample frames → windows of 8 paths            │
(mp4 or crop JPGs)  │       → encode_videos (one vector per window)   │
                    │       → pool windows → tracklet representation  │
                    │       → store vectors + metadata (numpy/Qdrant) │
                    └─────────────────────────────────────────────────┘

                    ┌── QUERY (each search) ──────────────────────────┐
Text caption        │  encode_texts → query vector q                 │
                    │  score vs gallery (cosine + pooling mode)       │
                    └─────────────────────────────────────────────────┘
```

- **Ground-truth captions are not used** at index time or in the demo unless you
  type them yourself. Benchmark **evaluation** encodes the dataset’s query
  captions only to measure Rank-1 / mAP.
- Encoders always see **8 RGB frames per forward** (a “clip”). A long tracklet is
  many overlapping 8-frame windows, not one monolithic video tensor.

Implementation: [`shawaf_vlm/sampling.py`](../shawaf_vlm/sampling.py),
[`shawaf_vlm/eval_loop.py`](../shawaf_vlm/eval_loop.py),
[`shawaf_vlm/pooling.py`](../shawaf_vlm/pooling.py).

---

## Step 1 — Get frames from a tracklet

| Source | What happens |
|--------|----------------|
| **Single `.mp4`** (TVPReid, GroOT) | Decode uniformly or by FPS; cache JPEGs if needed. |
| **Many crop `.jpg`s** (local annotations) | Use paths directly; optional `preview.mp4` is for UI only. |
| **RSTPReid (images)** | One or few images; **repeat last frame** to reach 8 frames. |

Functions: `resolve_frame_paths`, `resolve_dense_frame_paths` in `sampling.py`.

---

## Step 2 — Temporal protocols (benchmark)

Two families appear in the Tier-A notebooks:

### `uniform8` (single-clip)

- Pick **8 frames** (uniform or `middle` sample) from the tracklet.
- **One** `encode_videos` call → **one** embedding per tracklet.
- Used for RSTPReid (image gallery) and a simple video baseline.
- Eval: `evaluate_text_to_tracklet(..., num_frames=8)`.

### `reid_8fps_n64` (sliding windows)

Default window settings in notebooks:

| Parameter | Typical value | Meaning |
|-----------|---------------|---------|
| `num_frames` / window | 8 | Frames per clip passed to the encoder |
| `stride` | 4 | Step between window starts (50% overlap) |
| `sample_fps` | 8 | Dense sampling rate along the tracklet |
| `max_frames` | 64 | Cap on dense frames before windowing |

Pipeline:

1. Build a dense list of frame paths (~8 FPS, up to 64 frames).
2. Slide a length-8 window with stride 4 → many windows per tracklet.
3. `encode_videos` on **all** windows (batched).
4. **Pool** window embeddings into scores (next section).

Eval: `evaluate_text_to_tracklet_windows(...)`.

Headline README numbers pick the **best** protocol **and** pool per model/dataset
unless a table says otherwise.

---

## Step 3 — Pooling (how windows become one score)

After encoding, each tracklet has clip matrix **C** ∈ ℝ^(K×D) (K windows, D dims),
L2-normalized per row. Query **q** ∈ ℝ^D is L2-normalized text.

| Pool name | Also called | Gallery vector / score | When it helps |
|-----------|-------------|------------------------|---------------|
| **`mean`** | Type A / global | **g** = normalize(mean of rows of C); score = q·g | Stable “whole tracklet” look; paragraph-style queries. |
| **`max`** | — | **g** = normalize(max pool over dim of C); score = q·g | Rare in headlines; element-wise max of clip vectors. |
| **`mean_s8`** | — | Subsample every 8th window, then **`mean`** | Cheaper global vector when K is large. |
| **`query_max`** | Type B / local | score = max_i (q·c_i) over windows i | Best **short moment** matches query; good when one phrase fits one segment. |
| **`global_local`** | Type C / hybrid | score = α·(q·g_mean) + (1−α)·max_i(q·c_i) | Balance overall appearance (α) vs best window (1−α). Demo default α=0.5. |

Formulas live in [`pooling.py`](../shawaf_vlm/pooling.py):

- `pool_clip_features(..., pool="mean"|"max")` — collapse clips to one vector.
- `query_max_similarity` — max cosine over windows.
- `hybrid_global_local_similarity` — Type C with `alpha`.

**Important:** For windowed eval, all pools share the **same** encoded windows;
switching pool does **not** re-run the vision encoder.

### Demo UI labels

| Gradio label | Internal mode |
|--------------|----------------|
| Average (global mean) | `mean` |
| Max (best clip) | `query_max` |
| Type C (global + local) | `global_local` (+ α slider) |

Index build stores **global** (`mean`) and **per-window** vectors so Type C and
`query_max` work at search time without re-encoding.

---

## Step 4 — Local annotations (your data)

1. **Prepare** — crop each person per frame → JPGs; optional `preview.mp4`.
2. **Index** — same windowing on **crop JPGs** (not the preview mp4).
3. **Search** — text only; hits link to `preview.mp4` when present.

See [`prepare_local_annotations.py`](../scripts/prepare_local_annotations.py) and
[`local_annotations.py`](../shawaf_vlm/data/local_annotations.py).

---

## Quick comparison (what to choose)

| You care about… | Protocol | Pool |
|-----------------|----------|------|
| Fast baseline, images | `uniform8` | `none` / single vector |
| Video tracklets (paper-style) | `reid_8fps_n64` | `query_max` or `global_local` |
| Demo exploration | (fixed at index time) | Try **Type C** with α |
| Reproducing README TVPReid row | best of above per model | best pool in grid |

Ablation notebook:
[`pooling_methods_comparison.ipynb`](../notebooks/pooling_methods_comparison.ipynb).

---

## CLI examples

```bash
# Single 8-frame clip per tracklet
python scripts/eval_encoder.py --model pe_core_l14 --num-frames 8

# Sliding windows + multiple pools
python scripts/eval_encoder.py --model pe_core_l14 --windows \
  --stride 4 --sample-fps 8 --max-frames 64 \
  --pools mean,query_max,global_local --hybrid-alpha 0.5
```

---

## Related docs

- [Architecture](ARCHITECTURE.md) — module map  
- [Full benchmark scores](full_benchmark_scores.md) — all models and metrics  
- [Datasets catalog](datasets_catalog.md) — Tier-A and planned sets  
