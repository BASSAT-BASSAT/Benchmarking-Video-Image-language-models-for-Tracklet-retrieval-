# SHAWAF Stage 1 — text-to-tracklet retrieval

Frozen video-language encoders for **text → tracklet** search on [TV-MARS](https://github.com/Tedysu0916/TV-MARS).

This package answers:

> Which tracklets semantically match what the user said?

It does **not** mix Re-ID appearance vectors into the VLM space, does **not** concatenate embeddings, and does **not** use Qdrant. Those are Stage 2 / Stage 3.

The identity encoder stays in the sibling `Person-ReID-BenchMark` repo.

```text
Text query ──► VLM text encoder ──► text vector
                                      ↕ cosine
Tracklet  ──► VLM video encoder ──► video vector
                                      ↓
                              Rank-1 / 5 / 10 / 20 / 50, mAP, MdR, MnR, nDCG@10, mINP
```

## Results — frozen models, no training

All models are tested as-is on the test splits. Each text description queries the gallery, and a hit counts when the returned tracklet shows the same person. Numbers below are Rank-1 for each model's best temporal setting — how the 8-frame clips are sampled and averaged matters a little, but it never changes the ranking. Eleven models ran on a Kaggle/Colab T4 across three environment groups: `base` (IRRA, SigLIP 2, Perception Encoder, InternVideo2 CLIP-S, InternVideo2-1B-s2, X-CLIP, LanguageBind), `extended_modern` (OpenAI CLIP L/14, Jina CLIP v2, Qwen3-VL-Embedding-2B), and `extended_gme` (GME-Qwen2-VL-2B, TVPReid only — it needs its own `transformers` version, so it has not been run on RSTPReid or GroOT yet; see the [checkpoint section](#extended-zero-shot-checkpoint--all-156-configurations) below).

![Rank-1 on every test set](assets/rank1_heatmap.png)

### TVPReid: describing a person finds their tracklet

PRID is the easiest set, iLIDS the hardest (only 75 galleries, cluttered airport halls), Duke the largest (603 tracklets). IRRA wins everywhere by a wide margin because it was designed for text-to-person retrieval. Among general vision-language models, Perception Encoder is strongest on PRID while SigLIP 2 holds up best on Duke. Pure video models — X-CLIP, LanguageBind, InternVideo2 — lag far behind on this task. The `extended` models land mid-pack: Qwen3-VL-Embedding beats LanguageBind on PRID and Duke, GME-Qwen2-VL beats it on Duke, while OpenAI CLIP L/14 and Jina CLIP v2 sit at or below ten points everywhere.

| Model | PRID | iLIDS | Duke |
|---|---:|---:|---:|
| IRRA | **66.55** / 76.09 | **30.00** / 41.29 | **38.06** / 50.79 |
| SigLIP 2 So400m | 35.56 / 50.84 | 18.67 / 28.88 | 21.97 / 33.03 |
| Perception Encoder L/14 | 38.38 / 52.04 | 19.33 / 32.55 | 17.66 / 28.12 |
| InternVideo2 CLIP-S | 32.39 / 43.77 | 13.33 / 24.14 | 18.49 / 28.36 |
| Qwen3-VL-Embedding-2B | 24.65 / 36.99 | 11.33 / 21.37 | 13.18 / 22.73 |
| LanguageBind | 16.20 / 24.82 | 12.67 / 22.09 | 7.30 / 14.46 |
| GME-Qwen2-VL-2B | 11.62 / 22.96 | 10.67 / 20.14 | 9.45 / 17.70 |
| OpenAI CLIP L/14 | 10.92 / 20.05 | 9.33 / 17.48 | 5.14 / 10.10 |
| Jina CLIP v2 | 7.75 / 14.41 | 11.33 / 19.47 | 4.15 / 9.80 |
| X-CLIP | 4.93 / 10.05 | 5.33 / 10.52 | 1.41 / 3.95 |
| InternVideo2-1B-s2 | 4.23 / 10.78 | 4.00 / 10.26 | 1.82 / 4.15 |

Each cell is **Rank-1 / mAP**. Median rank, Rank-5/10/20/50, nDCG@10, and every protocol × pool combination are in the executed notebook.

IRRA is also the cheapest of the accurate models at ~34 ms per clip and 0.31 GB on a T4. Perception Encoder (~249 ms, 1.32 GB) and SigLIP 2 (~437 ms, 2.23 GB) cost 7–12× more compute and still trail by ~28 points on PRID. X-CLIP is equally fast but not competitive on accuracy. In practice: a single 8-frame clip is enough on PRID, while Duke benefits from scoring several sliding windows and keeping the best-matching one.

The `extended` models are no faster despite being newer: OpenAI CLIP L/14 is the cheapest at ~106 ms, Qwen3-VL-Embedding-2B sits at ~343 ms (2.1 GB), Jina CLIP v2 at ~1,329 ms, and GME-Qwen2-VL-2B is the slowest and heaviest of all at ~1,750 ms per clip and 4.2–8.3 GB. GME is the second-strongest extended model — it beats OpenAI CLIP and Jina on PRID and Duke — but it needs ~5× Qwen3's compute to get there. None of the extended models beats the task-tuned baselines anywhere.

![Cost vs accuracy](assets/speed_accuracy.png)

### RSTPReid: same task on still images

2,000 descriptions against 1,000 still images from 200 identities (5 images per person, so mAP is naturally lower than Rank-1). The order is familiar — IRRA first — but still images reshuffle the middle: SigLIP 2 (41.00) passes Perception Encoder (37.40), its image pretraining paying off when there is no motion to aggregate. Qwen3-VL-Embedding (30.05) is the one `extended_modern` model that climbs, passing InternVideo2 CLIP-S.

| Model | Rank-1 / mAP |
|---|---:|
| IRRA | **53.30** / 39.71 |
| SigLIP 2 So400m | 41.00 / 29.34 |
| Perception Encoder L/14 | 37.40 / 29.09 |
| InternVideo2 CLIP-S | 27.00 / 21.34 |
| Qwen3-VL-Embedding-2B | 30.05 / 21.72 |
| LanguageBind | 15.55 / 11.36 |
| Jina CLIP v2 | 12.15 / 9.64 |
| OpenAI CLIP L/14 | 11.05 / 8.21 |
| InternVideo2-1B-s2 | 5.15 / 4.53 |
| X-CLIP | 1.40 / 2.26 |

### GroOT-MOT17: clothes are searchable, actions are not

454 street tracks queried three ways: what the person wears (`appearance`, 344 queries), what they do (`action`, 454 queries), and a cleaned one-sentence mix of both (`combined`, 344 queries). Appearance separates people; action does not — only 112 of the 454 action captions are unique, and "a person walking on the street" alone matches 78 tracks. So every model scores 1–3% Rank-1 on action. That is a dataset ceiling, not a model bug.

Adding appearance back fixes it: the combined sentence beats appearance alone by 2–5 points for every serious model, reaching 29.36 for both IRRA and Perception Encoder.

![GroOT-MOT17 appearance vs action vs combined](assets/groot_caption_types.png)

| Model | appearance | action | combined |
|---|---:|---:|---:|
| Perception Encoder L/14 | 25.29 | 3.30 | **29.36** |
| IRRA | 24.71 | 2.20 | **29.36** |
| SigLIP 2 So400m | **25.58** | 2.20 | 27.91 |
| InternVideo2 CLIP-S | 19.19 | 2.64 | 24.71 |
| Qwen3-VL-Embedding-2B | 20.06 | **3.52** | 22.38 |
| LanguageBind | 14.83 | 2.20 | 16.28 |
| OpenAI CLIP L/14 | 11.92 | 1.76 | 10.17 |
| Jina CLIP v2 | 11.92 | 1.32 | 8.14 |
| X-CLIP | 2.62 | 0.44 | 2.91 |
| InternVideo2-1B-s2 | 0.87 | 0.66 | 1.45 |

Takeaway: current frozen models match clothing well, ignore generic motion, and do best when the query mentions both. Qwen3-VL-Embedding is a partial exception on motion: its 3.52 on action is the best of the ten models that ran GroOT, hinting that video-native pretraining helps even where captions are ambiguous. The `combined` queries were built by rule-merging each track's two captions, then hand-checking all 344 for grammar and two gender conflicts (no new attributes added). See [`shawaf_vlm/data/groot_mot17_combined.json`](shawaf_vlm/data/groot_mot17_combined.json).

### ActivityNet Captions: long videos, general events

An out-of-domain check: the same frozen models on general-purpose, long-form video instead of person tracklets. The gallery is 1,000 `val1` videos (a seeded random sample, `MAX_VIDEOS=1000`, `ANET_SEED=0`; ~180 s on average) from [`friedrichor/ActivityNet_Captions`](https://huggingface.co/datasets/friedrichor/ActivityNet_Captions). Two query types share that gallery:

- `paragraph` — one query per video: all of its event sentences joined into one paragraph.
- `sentence` — one query per annotated event (about 3.6 per video).

Every query has exactly one correct video, so mINP equals mAP. Only the seven `base` models were run; the `extended` models have not been run on ActivityNet yet.

| Model | paragraph | sentence | mean Rank-1 |
|---|---:|---:|---:|
| Perception Encoder L/14 | 68.10 / 78.87 | **41.91** / 54.83 | **55.01** |
| InternVideo2 CLIP-S | **69.80** / 80.24 | 39.73 / 52.87 | 54.76 |
| LanguageBind | 64.10 / 75.18 | 36.96 / 49.82 | 50.53 |
| SigLIP 2 So400m | 57.10 / 70.18 | 35.11 / 48.51 | 46.11 |
| InternVideo2-1B-s2 | 37.70 / 48.77 | 10.62 / 15.48 | 24.16 |
| X-CLIP | 28.60 / 41.53 | 14.04 / 23.59 | 21.32 |
| IRRA | 16.40 / 27.20 | 8.29 / 15.40 | 12.35 |

Each cell is **Rank-1 / mAP**, taken from each model's best protocol and pool for that query type. The best setting per cell:

| Model | paragraph | sentence |
|---|---|---|
| Perception Encoder L/14 | `vt_2fps_n32` / `query_max` | `reid_8fps_n64` / `query_max` |
| InternVideo2 CLIP-S | `vt_2fps_n32` / `mean_s8` | `reid_8fps_n64` / `query_max` |
| LanguageBind | `vt_2fps_n32` / `mean_s8` | `reid_8fps_n64` / `query_max` |
| SigLIP 2 So400m | `reid_8fps_n64` / `query_max` | `reid_8fps_n64` / `query_max` |
| InternVideo2-1B-s2 | `vt_2fps_n32` / `query_max` | `reid_8fps_n64` / `query_max` |
| X-CLIP | `vt_2fps_n32` / `mean` | `reid_8fps_n64` / `mean` |
| IRRA | `reid_8fps_n64` / `query_max` | `reid_8fps_n64` / `query_max` |

**The ranking flips relative to TVPReid.** IRRA, the clear winner on person tracklets, is last here (16.40 paragraph Rank-1): it was trained on person descriptions and does not transfer to general events. InternVideo2 CLIP-S and LanguageBind, which lag far behind on TVPReid, land in the top three. Perception Encoder is the best model overall by mean Rank-1, with InternVideo2 CLIP-S a close second (best on paragraph, second on sentence). InternVideo2-1B-s2, built for 4-frame clips (`uniform4` / `middle4`, 28 runs instead of 26), is the clear underperformer among the video models, and X-CLIP is far behind the leaders even though it is the fastest.

**Paragraph queries beat sentence queries for every model**, by 8 to 30 points, because a whole-video caption carries many more cues than a single event.

| Model | paragraph | sentence | paragraph − sentence |
|---|---:|---:|---:|
| InternVideo2 CLIP-S | 69.80 | 39.73 | 30.07 |
| LanguageBind | 64.10 | 36.96 | 27.14 |
| InternVideo2-1B-s2 | 37.70 | 10.62 | 27.08 |
| Perception Encoder L/14 | 68.10 | 41.91 | 26.19 |
| SigLIP 2 So400m | 57.10 | 35.11 | 21.99 |
| X-CLIP | 28.60 | 14.04 | 14.56 |
| IRRA | 16.40 | 8.29 | 8.11 |

**Temporal setting matters far more here than on TVPReid.** Videos are long, so dense sampling plus `query_max` helps most on sentence queries, where one window can match one event: every model's best `sentence` result uses `reid_8fps_n64`. On paragraph queries, `mean` / `mean_s8` over `vt_2fps_n32` is as good as or better than `query_max` for InternVideo2 CLIP-S, LanguageBind, and X-CLIP. `max` pooling is the weakest pool for every model. For example, Perception Encoder drops to 58.50 paragraph Rank-1 with `reid_8fps_n64` / `max`, against 68.10 for its best setting.

Cost on a Kaggle T4×2 session (two models run concurrently, one per GPU, on long videos):

| Model | ms / clip (single) | ms / clip (windows) | Peak GPU GB | Total min |
|---|---:|---:|---:|---:|
| X-CLIP | 30.76 | 29.94 | 0.45 | 49 |
| IRRA | 75.02 | 76.75 | 0.61 | 91 |
| LanguageBind | 209.93 | 222.12 | 1.22 | 399 |
| InternVideo2-1B-s2 | 301.07 | 301.18 | 5.52 | 545 |
| Perception Encoder L/14 | 374.51 | 375.85 | 1.32 | 640 |
| InternVideo2 CLIP-S | 379.22 | 381.57 | 1.54 | 651 |
| SigLIP 2 So400m | 486.44 | 486.05 | 2.23 | 823 |

These timings come from a different GPU setup and much longer videos than the TVPReid timings above, so do not compare the two sets of numbers directly.

> **Caveat.** The gallery is a 1,000-video random sample of the 4,917-video `val1` split, so these numbers are **not comparable to published ActivityNet Captions retrieval results**. Set `MAX_VIDEOS = None` in the notebook for a full-split run. This benchmark also measures frozen, zero-shot encoders, not models fine-tuned for ActivityNet.

The two ActivityNet notebooks are [`notebooks/kaggle-activitynet-siglip-irra-xclip-pe.ipynb`](notebooks/kaggle-activitynet-siglip-irra-xclip-pe.ipynb) (SigLIP 2, Perception Encoder, IRRA, X-CLIP; 104 result rows) and [`notebooks/kaggle-activitynet-languagebind-internvideo2.ipynb`](notebooks/kaggle-activitynet-languagebind-internvideo2.ipynb) (InternVideo2-1B-s2, LanguageBind, InternVideo2 CLIP-S; 80 result rows). The merged per-configuration results (184 rows, with Rank-5/10/20/50, MdR, MnR and nDCG@10) are in [`results/checkpoints/activitynet_zero_shot_7models_v1.csv`](results/checkpoints/activitynet_zero_shot_7models_v1.csv).

Full per-protocol numbers, Rank-5/10/20/50, nDCG, and timing breakdowns for TVPReid, RSTPReid and GroOT-MOT17 are in [`notebooks/kaggle-zero-shot.ipynb`](notebooks/kaggle-zero-shot.ipynb). The ActivityNet breakdowns are in the two notebooks above.

## Install

Python 3.10+. On this laptop the CUDA env is `crowd-gpu`.

```bash
pip install -e ".[all]"
```

Extras: `xclip`, `languagebind`, `internvideo2`, `extended_modern`,
`extended_gme`, or `all`.

The extended adapters require two mutually exclusive environments:

- `extended_modern`: OpenAI CLIP, Jina CLIP v2, and Qwen3-VL-Embedding with
  Transformers 4.57.3.
- `extended_gme`: GME-Qwen2-VL-2B with Transformers 4.51.3 and its official
  Sentence Transformers route.

Both use PyTorch 2.8 / torchvision 0.23. Switching profiles in Colab requires a
runtime restart; the notebook detects this and stops with instructions. It does
not patch Hugging Face cached source files.

## Dataset

TV-MARS reuses **MARS crops** and adds gated **caption JSON**.

```text
<data_root>/
  bbox_train/
  bbox_test/
<ann_root>/
  train_info/            # gated
  test_query_info/       # gated
  test_gallery_info/     # gated
```

1. MARS pixels: already used by the Re-ID bench (`Person-ReID-BenchMark/datasets/MARS`), or download from the [MARS project page](https://zheng-lab.cecs.anu.edu.au/Project/project_mars.html).
2. Captions: complete the [TV-MARS agreement](https://github.com/Tedysu0916/TV-MARS/raw/main/TV-MARS%20Agreement.pdf) and email **jiajunsu@stu.hqu.edu.cn**.

Until the official splits arrive, `--ann-root` can point at the public `partical_dataset` folder from the TV-MARS GitHub (flat JSON files, same schema).

JSON records look like:

```json
{
  "img_path": ["/media/.../bbox_train/0009/0009C1T0001F001.jpg"],
  "person_id": "0009",
  "track_id": "T0001",
  "captions": ["The person in the video is wearing a white shirt..."]
}
```

Author machine prefixes are stripped; only the `bbox_train/` or `bbox_test/` suffix is joined to `--data-root`.

## Encoders (frozen, zero-shot)

| `--model` | Checkpoint | Notes |
|---|---|---|
| `xclip` | `microsoft/xclip-base-patch32` | Smallest; laptop 8 GB is enough |
| `languagebind` | `LanguageBind/LanguageBind_Video` | Official LanguageBindVideo class (Hub has no AutoModel `auto_map`) |
| `internvideo2` | `OpenGVLab/InternVideo2_CLIP_S` | Default InternVideo2 (~373M) |
| `internvideo2_clip_1b` | `OpenGVLab/InternVideo2-CLIP-1B-224p-f8` | Optional; HF repo is a gated LoRA add-on, not a full AutoModel |
| `openai_clip_vit_l14` | OpenAI `ViT-L/14` | Official frame encoder; frame mean inside each clip |
| `jina_clip_v2` | `jinaai/jina-clip-v2` | Frame encoder; frame mean inside each clip |
| `gme_qwen2_vl_2b` | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | Official image/text ST path; frame mean inside each clip |
| `qwen3_vl_embed_2b` | `Qwen/Qwen3-VL-Embedding-2B` | Native ordered-frame video input; 8,192-token context, fp16/T4 batch 1 |

Default protocol: **8 frames**, 224², cosine on L2 vectors. Same-camera junk is **off** (the query is text). Pass `--junk-same-camera` only if you need Market1501-style filtering.

Whole-tracklet encoding (`--windows`): sliding 8-frame clips, then pool without a second encode:

- `uniform8` — 8 frames across the video (Stage 1 baseline)
- `vt_1fps_n12` — CLIP4Clip sampling (1 fps, 12-frame cap)
- `vt_2fps_n32` — denser sparse coverage
- `reid_8fps_n64` — TVPR-style consecutive clips (8 fps, 64-frame cap)

Pools: `mean` (CLIP4Clip), `mean_s8` (stride 8 from a stride-4 encode), `max`, `query_max` (X-Pool-lite).

`query_max` remains a late-interaction diagnostic and is **not** official
X-Pool. The three new image/text models use the same frame-normalize → mean
frames → clip-normalize contract as IRRA. Qwen3-VL-Embedding receives each
ordered frame list as native video input, then the existing tracklet pools
operate on its clip embeddings.

Related: [CLIP4Clip](https://arxiv.org/abs/2104.08860), [X-Pool](https://arxiv.org/abs/2203.15086), [TVPR](https://arxiv.org/abs/2307.07184).

## Extended zero-shot checkpoint — all 156 configurations

The headline tables above keep one number per model and dataset. This section is the frozen record behind the extended models: **156 completed runs** — 4 models (OpenAI CLIP ViT-L/14, Jina CLIP v2, GME-Qwen2-VL-2B, Qwen3-VL-Embedding-2B) × 3 TVPReid subsets (PRID, iLIDS, Duke) × 13 configurations, with no Re-ID fusion and no fine-tuning. The best rows below match the TVPReid columns above; the multi-dataset Kaggle runs reproduce them within ±0.02 mAP.

Each model/dataset pair covers `uniform8` plus the `vt_1fps_n12`,
`vt_2fps_n32`, and `reid_8fps_n64` protocols with the applicable `mean`,
`mean_s8`, `max`, and `query_max` pools. Here, `query_max` is the repository's
pooling strategy used for this benchmark; it is not an implementation of
official X-Pool.

Rows are selected by highest Rank-1, then mAP, then nDCG@10, with CSV order as
the final deterministic tie-breaker. Every value in a table row comes from the
same result row.

| Model | Dataset | Rank-1 | mAP | nDCG@10 | Best Protocol | Pool | Video ms/item | Peak GPU GB |
|---|---|---:|---:|---:|---|---|---:|---:|
| OpenAI CLIP ViT-L/14 | PRID | 10.92 | 20.06 | 22.72 | `uniform8` | `none` | 132.53 | 0.92 |
| OpenAI CLIP ViT-L/14 | iLIDS | 9.33 | 17.48 | 18.19 | `vt_1fps_n12` | `max` | 96.95 | 0.92 |
| OpenAI CLIP ViT-L/14 | Duke | 5.14 | 10.12 | 10.74 | `vt_1fps_n12` | `query_max` | 107.42 | 0.92 |
| Jina CLIP v2 | PRID | 7.75 | 14.41 | 15.76 | `vt_1fps_n12` | `mean` | 1281.34 | 4.19 |
| Jina CLIP v2 | iLIDS | 11.33 | 19.47 | 21.24 | `reid_8fps_n64` | `query_max` | 1284.85 | 2.58 |
| Jina CLIP v2 | Duke | 4.15 | 9.80 | 11.03 | `reid_8fps_n64` | `query_max` | 1352.50 | 2.58 |
| GME-Qwen2-VL-2B | PRID | 11.62 | 22.96 | 26.36 | `reid_8fps_n64` | `max` | 1681.47 | 8.35 |
| GME-Qwen2-VL-2B | iLIDS | 10.67 | 20.14 | 22.48 | `vt_2fps_n32` | `query_max` | 1657.36 | 4.21 |
| GME-Qwen2-VL-2B | Duke | 9.45 | 17.70 | 20.48 | `reid_8fps_n64` | `query_max` | 1909.83 | 4.26 |
| Qwen3-VL-Embedding-2B | PRID | 24.65 | 37.00 | 41.51 | `reid_8fps_n64` | `query_max` | 337.56 | 4.07 |
| Qwen3-VL-Embedding-2B | iLIDS | 11.33 | 21.37 | 24.02 | `reid_8fps_n64` | `mean_s8` | 329.77 | 4.07 |
| Qwen3-VL-Embedding-2B | Duke | 13.18 | 22.72 | 26.11 | `reid_8fps_n64` | `query_max` | 335.74 | 4.07 |

The full checkpoint with all configurations and unrounded metrics is
[`results/checkpoints/extended_zero_shot_4models_3datasets_v1.csv`](results/checkpoints/extended_zero_shot_4models_3datasets_v1.csv).
It was produced with the Colab notebook below, in two dependency profiles:

- **MODERN:** OpenAI CLIP ViT-L/14, Jina CLIP v2, and
  Qwen3-VL-Embedding-2B with Transformers 4.57.3.
- **GME:** GME-Qwen2-VL-2B with Transformers 4.51.3. GME requires this
  separate Transformers-compatible runtime.

## Extended Colab benchmark

Open `notebooks/colab_extended_zero_shot.ipynb` in Google Colab. It checks out
`main`, installs the pinned T4 environment,
downloads only selected TVPReid test subsets, validates the chosen encoder,
runs PRID `uniform8` first, resumes JSON results, and saves every protocol
immediately. It contains no pre-filled benchmark scores.

## Run locally

```bash
python scripts/eval_encoder.py \
  --model xclip \
  --data-root "../Person-ReID-BenchMark/datasets/MARS" \
  --ann-root /path/to/tv-mars-captions \
  --device cuda
```

Whole-tracklet windows (encode stride 4 once, then pool `mean` / `mean_s8` / `max` / `query_max`):

```bash
python scripts/eval_encoder.py \
  --model languagebind \
  --windows --stride 4 --sample-fps 2 --max-frames 32 \
  --pools mean,mean_s8,max,query_max
```

Results go to `results/<model>_tv_mars.json`.

Or import the same functions from a notebook:

```python
from shawaf_vlm.models import build_encoder
from shawaf_vlm.data.tv_mars import load_tv_mars
from shawaf_vlm.eval_loop import evaluate_text_to_tracklet

encoder = build_encoder("xclip", device="cuda")
splits = load_tv_mars(data_root, ann_root)
metrics = evaluate_text_to_tracklet(encoder, splits, num_frames=8)
```

## Kaggle

[`notebooks/kaggle-zero-shot.ipynb`](notebooks/kaggle-zero-shot.ipynb) is the one multi-dataset zero-shot benchmark.

1. Create a notebook with a **T4 GPU** and **Internet** on. Add the Hugging Face token as the Kaggle secret `hugging_face` (only InternVideo2-1B-s2 needs it).
2. Set `ENV_GROUP` in the configuration cell and **Run All**. Run each group in its own session, because the groups pin different `transformers` versions:
   - `base`: SigLIP 2, Perception Encoder, IRRA, InternVideo2-1B-s2, X-CLIP, LanguageBind, InternVideo2 CLIP-S
   - `extended_modern`: OpenAI CLIP ViT-L/14, Jina CLIP v2, Qwen3-VL-Embedding-2B
   - `extended_gme`: GME-Qwen2-VL-2B
3. The install cell hard-resets `/kaggle/working/shawaf-vlm` to `origin/main` and must print `shawaf_vlm 0.1.18`. If it asks for a restart, use **Restart session**, then **Run All** again.
4. `RUN_DATASETS` picks the datasets:

| Dataset | Type | Source in the notebook |
|---|---|---|
| TVPReid (PRID / iLIDS / Duke) | video tracklets | [bassatbassat/TVPReid](https://huggingface.co/datasets/bassatbassat/TVPReid), test videos only |
| RSTPReid | one image per tracklet | Google Drive via `gdown` (MSMT17 license forbids re-hosting) |
| GroOT-MOT17 (`all` / `appearance` / `action` / `combined`) | video tracklets | [bassatbassat/GroOT-MOT17](https://huggingface.co/datasets/bassatbassat/GroOT-MOT17) |
| ActivityNet Captions (`paragraph` / `sentence`) | long video, whole-video or per-event captions | [friedrichor/ActivityNet_Captions](https://huggingface.co/datasets/friedrichor/ActivityNet_Captions) `val1`, seeded 1,000-video sample; own notebooks, see below |

Each result is saved as JSON right away under `zero_shot_results/<ENV_GROUP>/<dataset>/`. Finished rows are skipped when you rerun. The table cells merge every group, plus any earlier `zero_shot_results` you attach as a Kaggle input. They show the leaderboard, a cross-dataset Rank-1 matrix, pool comparisons, the GroOT appearance vs action vs combined comparison, and speed/GPU. They also write CSVs and `report.md` to `zero_shot_results/tables/`.

The notebook runs `pip install -e ".[all]"` and **imports** `shawaf_vlm`. It does not reimplement the eval loop.

### ActivityNet Captions notebooks

ActivityNet Captions is run from two separate notebooks, both on a Kaggle **T4 ×2** session with Internet on and the `hugging_face` secret attached. With two GPUs visible, two workers run models concurrently, one pinned to each T4.

| Notebook | Models |
|---|---|
| [`notebooks/kaggle-activitynet-siglip-irra-xclip-pe.ipynb`](notebooks/kaggle-activitynet-siglip-irra-xclip-pe.ipynb) | SigLIP 2, Perception Encoder L/14, IRRA, X-CLIP |
| [`notebooks/kaggle-activitynet-languagebind-internvideo2.ipynb`](notebooks/kaggle-activitynet-languagebind-internvideo2.ipynb) | InternVideo2-1B-s2, LanguageBind, InternVideo2 CLIP-S |

Both use `ENV_GROUP = "base"`. The `extended_modern` and `extended_gme` groups are already defined in the first notebook's configuration, so the `extended` models can be run on ActivityNet later with the same code, one group per session.

- The videos live inside a ~42 GB split archive on the Hub (`ActivityNet_Videos.tar.part-000 … 007`). The notebook streams it once and keeps only the selected videos on disk (in `/kaggle/temp`); finished archive parts are deleted immediately.
- `paragraph` and `sentence` share one gallery, so gallery embeddings are reused between them (rows marked `gallery_cached`).
- `MAX_VIDEOS` (default 1000, seeded by `ANET_SEED = 0`) keeps a session inside the Kaggle time limit. Set it to `None` for the full 4,917-video `val1` split; set `ANET_SPLIT = "val2"` for the other split.
- Each result row is written as JSON the moment it finishes, so a restarted session skips what is done. To combine sessions, save each session's `/kaggle/working/zero_shot_results` as a Kaggle dataset and attach it.
- The notebooks write the same table CSVs and `report.md` to `zero_shot_results/tables/`, including `activitynet_paragraph_vs_sentence.csv`.
- Both notebooks pin `shawaf_vlm 0.1.18`.

GroOT-MOT17 was built once from MOT17 train (FRCNN copy) and the [GroOT](https://github.com/uark-cviu/Type-to-Track) captions (Nguyen et al., NeurIPS 2023):

```bash
python scripts/build_groot_mot17.py --out build/GroOT-MOT17 --upload
```

Both sources are CC BY-NC-SA 3.0, so the Hugging Face copy is under the same license.

```bash
git clone --depth 1 https://github.com/BASSAT-BASSAT/Benchmarking-Video-Image-language-models-for-Tracklet-retrieval-.git
```

Environment overrides: `SHAWAF_TVPREID_ROOT`, `SHAWAF_RSTPREID_ROOT`, `SHAWAF_GROOT_ROOT`, `SHAWAF_MARS_ROOT`, `SHAWAF_TV_MARS_ANN`.

## What is intentionally not here

- **Stage 2** — VLM top-K, then Re-ID rerank / late fusion. Needs the FastReID tracklet vectors from `Person-ReID-BenchMark`.
- **Stage 3** — learned projection of Re-ID features into the VLM text space (contrastive alignment). That is a research experiment, not an engineering default.
- Qdrant, Qwen3-VL, CLIP-ReID training.

Establish these three frozen TV-MARS numbers first. Then ask whether appearance Re-ID improves text-to-tracklet retrieval.

## Tests (no GPU)

```bash
pip install -e ".[test]"
pytest -q
```
