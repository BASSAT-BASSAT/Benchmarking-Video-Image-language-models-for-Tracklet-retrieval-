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

## Zero-shot TVPReid results

Tesla T4, test split, both captions used as queries. Each number is that model's best Rank-1 on the subset. Peak GPU and milliseconds per clip stay about the same for every protocol; the Duke video time is the wall clock of the run that produced that model's best Duke Rank-1.

### Rank-1 / mAP / median rank

| Model | PRID | iLIDS | Duke |
|---|---:|---:|---:|
| IRRA | **66.55** / 76.09 / 1 | **30.00** / 41.27 / 5 | **38.06** / 50.79 / 2 |
| Perception Encoder L/14 | 38.38 / 52.04 / 2 | 19.33 / 32.55 / 6 | 17.66 / 28.12 / 11 |
| SigLIP 2 So400m | 35.56 / 50.84 / 3 | 18.67 / 28.88 / 10 | 21.97 / 33.03 / 7 |
| InternVideo2 CLIP-S | 32.39 / 43.77 / 4 | 13.33 / 24.14 / 11.5 | 18.49 / 28.36 / 12 |
| LanguageBind | 16.20 / 24.82 / 16 | 12.67 / 22.09 / 14 | 7.30 / 14.46 / 40.5 |
| X-CLIP | 4.93 / 10.05 / 43.5 | 5.33 / 10.52 / 38.5 | 1.41 / 3.95 / 164 |
| InternVideo2-1B-s2 | 4.23 / 10.78 / 34.5 | 4.00 / 10.26 / 29 | 1.82 / 3.82 / 159 |

Each cell is **Rank-1 / mAP / MdR**.

### Setting, speed, and GPU

| Model | Best setting | ms / clip | Peak GPU | Duke video encode |
|---|---|---:|---:|---:|
| IRRA | PRID 8 fps `mean_s8`; iLIDS 2 fps `mean`; Duke 8 fps `query_max` | 34 | 0.31 GB | 4.8 min |
| Perception Encoder L/14 | PRID 8 fps `mean_s8`; iLIDS and Duke 8 fps `query_max` | 238 | 1.32 GB | 33.8 min |
| SigLIP 2 So400m | 8 fps `query_max` on all three subsets | 421 | 2.23 GB | 59.8 min |
| InternVideo2 CLIP-S | PRID 2 fps `max`; iLIDS and Duke 8 fps `query_max` | 354 | 1.54 GB | 50.2 min |
| LanguageBind | PRID 2 fps `mean`; iLIDS and Duke 8 fps `mean` | 192 | 1.22 GB | 27.2 min |
| X-CLIP | PRID and Duke 8 fps `mean`; iLIDS 2 fps `mean` | 30 | 0.45 GB | 4.2 min |
| InternVideo2-1B-s2 | PRID uniform 4; iLIDS 2 fps `query_max`; Duke 1 fps `mean_s8` | 262 | 2.81 GB | 6.9 min |

Text encoding is under a few seconds in every run. IRRA is the accuracy leader and the cheapest of the strong models. Full rows, including Rank-5/10/20/50, nDCG@10, and every pool, are in [`notebooks/kaggle_zero_shot.ipynb`](notebooks/kaggle_zero_shot.ipynb).

### What the number is averaging

The headline table is **not** one shared mean. It is each model's best Rank-1, and the pool differs.

IRRA, Perception Encoder, and SigLIP 2 embed every frame on its own and **average the frames inside each 8-frame clip**. After that, a second pool merges the clips:

| Pool | What it does |
|---|---|
| one clip (`uniform8`) | 8 frames spread over the tracklet, averaged into one vector |
| `mean` | average every clip vector |
| `mean_s8` | average every other clip (about stride 8) |
| `max` | keep the strongest value in each dimension |
| `query_max` | score each clip against the caption and keep the best clip |

At 1 fps a tracklet often has only one clip, so `mean` and `mean_s8` match. Rank-1 below.

### IRRA Rank-1 by protocol

| Sampling | Pool | PRID | iLIDS | Duke |
|---|---|---:|---:|---:|
| 8 frames, one clip | frame average | 65.49 | 28.00 | 35.82 |
| 1 fps, 12-frame cap | mean | 63.03 | 27.33 | 35.66 |
| 1 fps, 12-frame cap | max | 63.38 | 27.33 | 35.57 |
| 1 fps, 12-frame cap | query_max | 63.03 | 27.33 | 35.90 |
| 2 fps, 32-frame cap | mean | 62.32 | **30.00** | 36.65 |
| 2 fps, 32-frame cap | max | 62.68 | 29.33 | 34.08 |
| 2 fps, 32-frame cap | query_max | 62.68 | 28.67 | 37.73 |
| 8 fps, 64-frame cap | mean | 65.85 | 27.33 | 36.07 |
| 8 fps, 64-frame cap | mean_s8 | **66.55** | 26.00 | 36.65 |
| 8 fps, 64-frame cap | max | 65.14 | 26.00 | 32.34 |
| 8 fps, 64-frame cap | query_max | 65.14 | 28.00 | **38.06** |

On PRID every IRRA protocol is within about 4 points, and median rank stays 1. Duke is where `query_max` pulls ahead of a plain average.

### Top three, Rank-1 by sampling

`mean` is the average of clip vectors. `query_max` keeps the clip that best matches the caption.

| Model | Subset | One clip | 1 fps mean | 2 fps mean | 8 fps mean | 8 fps query_max |
|---|---|---:|---:|---:|---:|---:|
| IRRA | PRID | 65.49 | 63.03 | 62.32 | 65.85 | 65.14 |
| IRRA | iLIDS | 28.00 | 27.33 | 30.00 | 27.33 | 28.00 |
| IRRA | Duke | 35.82 | 35.66 | 36.65 | 36.07 | 38.06 |
| Perception Encoder | PRID | 38.03 | 35.56 | 34.15 | 38.03 | 35.56 |
| Perception Encoder | iLIDS | 18.00 | 16.67 | 19.33 | 18.00 | 19.33 |
| Perception Encoder | Duke | 14.84 | 14.01 | 13.76 | 15.01 | 17.66 |
| SigLIP 2 | PRID | 33.45 | 32.75 | 32.75 | 32.04 | 35.56 |
| SigLIP 2 | iLIDS | 14.67 | 17.33 | 16.00 | 16.67 | 18.67 |
| SigLIP 2 | Duke | 17.58 | 18.16 | 19.98 | 19.40 | 21.97 |

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

## Extended Colab benchmark

Open `notebooks/colab_extended_zero_shot.ipynb` in Google Colab. It checks out
`feat/extended-zero-shot-benchmark`, installs the pinned T4 environment,
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

1. Create a notebook with **GPU** and **Internet** on.
2. Open [`notebooks/kaggle_VLM_eval.ipynb`](notebooks/kaggle_VLM_eval.ipynb) only — do not add extra notebooks.
3. After any CUDA assert, **Restart session**, then **Run All**. The install cell hard-resets `/kaggle/working/shawaf-vlm` to `origin/main` and must print `shawaf_vlm 0.1.15`.
4. It downloads only the **test** videos from [bassatbassat/TVPReid](https://huggingface.co/datasets/bassatbassat/TVPReid).

The notebook `pip install -e ".[all]"` and **imports** `shawaf_vlm` — it does not reimplement the eval loop.

```bash
git clone --depth 1 https://github.com/BASSAT-BASSAT/Benchmarking-Video-Image-language-models-for-Tracklet-retrieval-.git
```

Environment overrides: `SHAWAF_TVPREID_ROOT`, `SHAWAF_MARS_ROOT`, `SHAWAF_TV_MARS_ANN`.

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
