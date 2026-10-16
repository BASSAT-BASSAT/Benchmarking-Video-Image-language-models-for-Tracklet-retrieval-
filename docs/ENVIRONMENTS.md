# Python environments

SHAWAF targets **Transformers 4.x** (see `pyproject.toml` extras). Two common failure modes:

## Symptom: `huggingface-hub>=0.24.0,<1.0 is required` (Hub 1.x installed)

Cause: `pip install -e ".[demo]"` with Gradio 6+ upgraded Hub to 1.x while Transformers 4.x is installed.

**Fix (recommended ML env, e.g. `crowd-gpu`):**

```bash
pip install "huggingface-hub>=0.23,<1.0" "gradio>=4.44,<5" "fastapi>=0.115.2,<0.116" "starlette>=0.40.0,<0.46.0"
```

Or run `scripts/env/repair_crowd_gpu_deps.ps1`.

## Symptom: `cannot import name 'is_offline_mode' from 'huggingface_hub'` (Hub 0.x + Transformers 5.x)

Cause: base conda with **Transformers 5** and an old **Hub 0.36**.

**Fix (base / TF5 env):**

```bash
pip install "huggingface-hub>=1.5,<2.0"
```

Or install the project ML stack instead of base:

```bash
conda activate crowd-gpu
pip install -e ".[all,test]"
```

## Symptom: `No module named 'pyaudioop'` when importing Gradio (Python 3.13+)

```bash
pip install audioop-lts
```

Included in the `demo` extra on Python 3.13+.

## Recommended workflow

| Task | Environment | Install |
|------|-------------|---------|
| Benchmark / encoders | `crowd-gpu` | `pip install -e ".[all]"` |
| Tests | same | `pip install -e ".[all,test]"` then `pytest` |
| Gradio demo | same + demo | `pip install -e ".[all,demo]"` + repair script if needed |
| EmbeddingGemma 2 notebook | **separate** env (TF ≥ 4.57) | `pip install -e ".[embeddinggemma2]"` — see `notebooks/embeddinggemma2_tier_a_zero_shot.ipynb` |

Set `HF_TOKEN` for Hub dataset downloads via `scripts/download_tier_a.py`.

**EmbeddingGemma 2** conflicts with the pinned `transformers==4.51.3` in `[all]` / Kaggle `base`. Do not mix it with the main benchmark extra in one kernel; use the dedicated notebook or `embeddinggemma2` extra only.

**`google/embeddinggemma-2` (Hub):** `sentence-transformers>=6.1.0`, **`transformers>=5.19.0`** (model type `embedding_gemma2` — not in older 5.x), SearchQuery + image gallery, `audio_config=None`. On Kaggle: `pip uninstall -y torchcodec sentence-transformers` then `pip install 'sentence-transformers>=6.1.0,<7' 'transformers>=5.19.0,<6'`. Restart kernel after setup.

## Gradio demo: tracklet previews look empty (grey video box)

TVPReid / iLIDS tracklets are often **MPEG-4 Part 2** (`mp4v`) inside `.mp4`. Chrome and Edge usually **cannot** decode that in an HTML5 `<video>` tag (encoding still works via OpenCV).

The demo always shows a **JPEG frame preview** from the clip. For real playback it transcodes once to H.264 using **ffmpeg** (system PATH, `FFMPEG_BINARY`, or the **`imageio-ffmpeg`** wheel bundled with `pip install -e ".[demo]"` — no Chocolatey/admin required).

```powershell
pip install imageio-ffmpeg
# optional system ffmpeg:
# choco install ffmpeg   # needs elevated shell on Windows
$env:FFMPEG_BINARY = "C:\path\to\ffmpeg.exe"
```

Transcoded files are cached under `%USERPROFILE%\.cache\shawaf_vlm\web_previews\`. Paths with spaces are URL-encoded for Gradio’s `/file=` route.
