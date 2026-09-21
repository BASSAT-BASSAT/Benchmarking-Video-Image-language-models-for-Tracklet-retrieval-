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
                              Rank-1 / 5 / 10 / 20, mAP
```

## Install

Python 3.10+. On this laptop the CUDA env is `crowd-gpu`.

```bash
pip install -e ".[all]"
```

Extras: `xclip`, `languagebind`, `internvideo2`, or `all`.

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
| `languagebind` | `LanguageBind/LanguageBind_Video` | Stronger video-language baseline |
| `internvideo2` | `OpenGVLab/InternVideo2_CLIP_S` | Default InternVideo2 (~373M) |
| `internvideo2_clip_1b` | `OpenGVLab/InternVideo2-CLIP-1B-224p-f8` | Optional; HF repo is a gated LoRA add-on, not a full AutoModel |

Default protocol: **8 frames**, 224², cosine on L2 vectors. Same-camera junk is **off** (the query is text). Pass `--junk-same-camera` only if you need Market1501-style filtering.

## Run locally

```bash
python scripts/eval_encoder.py \
  --model xclip \
  --data-root "../Person-ReID-BenchMark/datasets/MARS" \
  --ann-root /path/to/tv-mars-captions \
  --device cuda
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
3. **Run All**. The install cell hard-resets `/kaggle/working/shawaf-vlm` to `origin/main` and must print `shawaf_vlm 0.1.2`.
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
