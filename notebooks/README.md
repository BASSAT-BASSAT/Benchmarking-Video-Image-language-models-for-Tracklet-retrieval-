# Notebooks — what to run and what is left

Handoff for anyone continuing the benchmark after the Tier-A Kaggle/Colab runs.
Protocols and pooling are explained in
[`docs/retrieval_protocol_and_pooling.md`](../docs/retrieval_protocol_and_pooling.md).

**Where results live**

| Location | Contents |
|----------|----------|
| Kaggle `/kaggle/working/zero_shot_results/<group>/` | Per-job JSON (`status: complete`); attach as a Kaggle dataset to merge groups |
| Repo `results/checkpoints/*.csv` | Small **summaries** committed for README / docs (not full JSON) |
| Local `results/zero_shot_results/` | Same layout if you run notebooks locally (`WORK` path in config) |

Re-runs are **resume-safe**: finished JSON files are skipped automatically.

---

## Status at a glance (March 2026)

| Notebook | Platform | Status |
|----------|----------|--------|
| [`kaggle-zero-shot.ipynb`](kaggle-zero-shot.ipynb) | Kaggle T4 | **Done** for `base`, `extended_modern`, `extended_gme` (scores in README + [`docs/full_benchmark_scores.md`](../docs/full_benchmark_scores.md)) |
| [`colab_extended_zero_shot.ipynb`](colab_extended_zero_shot.ipynb) | Colab T4 | **Done** for TVPReid-only extended VLMs (checkpoint: `results/checkpoints/extended_zero_shot_4models_3datasets_v1.csv`) |
| [`embeddinggemma2_tier_a_zero_shot.ipynb`](embeddinggemma2_tier_a_zero_shot.ipynb) | Kaggle T4 | **Partial** — **5** jobs left: TVPReid **Duke** `reid_8fps_n64` only (see below) |
| [`activity-net-siglip-irra-xclip-perception.ipynb`](activity-net-siglip-irra-xclip-perception.ipynb) | Kaggle (archived) | **Done** — executed notebook with outputs; ActivityNet table in full scores |
| [`activity-net-languagebind-internvideo.ipynb`](activity-net-languagebind-internvideo.ipynb) | Kaggle (archived) | **Done** — same |
| [`pooling_methods_comparison.ipynb`](pooling_methods_comparison.ipynb) | Local / Colab | **Optional** — small TVPReid ablation (change `MODEL` / `DATASET_KEY`) |
| [`kaggle_finetuning.ipynb`](kaggle_finetuning.ipynb) | Kaggle | **Optional** — InternVideo2-1B-s2 fine-tune on TVPReid train; not part of zero-shot headline |

**Not in this folder:** local demo indexing (`scripts/index_qdrant.py`), CLI eval (`scripts/eval_encoder.py`), local annotations (`scripts/prepare_local_annotations.py`).

---

## Priority: finish EmbeddingGemma 2 Tier-A

Notebook: **`embeddinggemma2_tier_a_zero_shot.ipynb`**  
Regenerate from source after editing eval logic: `python scripts/gen_embeddinggemma2_notebook.py`

### Environment (separate from `base` Kaggle group)

1. GPU **T4**, Internet on.
2. `pip install -e ".[embeddinggemma2]"` (needs **transformers ≥ 5.19**, **sentence-transformers ≥ 6.1**).
3. On Kaggle: uninstall **torchcodec** and **torchaudio** if the kernel pulled them in; **restart kernel** before importing `transformers`.
4. Do **not** reuse a session that already imported an older `transformers` from the `base` group.

### How to run (notebook is preconfigured)

§1 is already set to **Duke only** + `reid_8fps_n64` + `RESUME_COMPLETE = True`. **Run All** on a fresh Kaggle notebook.

1. Add a Kaggle **input dataset** that contains your prior `zero_shot_results/embeddinggemma2/` tree (optional but recommended so §1 seeds JSON and §4 does not redo GroOT/PRID).
2. Or set env `SHAWAF_PRIOR_RESULTS` to that folder path before §1.
3. §4 should print **1** pending job: `tvpreid/duke | reid_8fps_n64` (five pool JSON files from one encode pass).

To re-run the **full** Tier-A grid, uncomment the full `RUN_DATASETS` / filter block in §1 (see comments there).

### What is complete (38 / 43 Tier-A jobs)

- **TVPReid prid + ilids:** `uniform8` + full `reid_8fps_n64` (all 5 pools).
- **TVPReid duke:** `uniform8` only.
- **RSTPReid test:** `uniform8` (image gallery; no window protocol).
- **GroOT-MOT17** (`all`, `appearance`, `action`, `combined`): `uniform8` + **full** `reid_8fps_n64` grid (all 5 pools per subset). Your §5 summary table is the source of truth here.

The repo checkpoint [`results/checkpoints/embeddinggemma2_tier_a_v1.csv`](../results/checkpoints/embeddinggemma2_tier_a_v1.csv) was an **early export** that only listed a few GroOT window rows; it did **not** mean GroOT was unfinished on Kaggle. Trust complete JSON under `zero_shot_results/embeddinggemma2/` or the notebook summary table.

### Jobs still to run (5)

| Dataset | Subset | Protocol | Pools |
|---------|--------|----------|--------|
| TVPReid | **duke** | `reid_8fps_n64` | mean, mean_s8, max, query_max, global_local |

§4 skips any job whose JSON already has `status: complete`. If Duke windows finished after an interrupt, you are **done** — refresh the checkpoint CSV from JSON and update the Duke footnote in full scores.

**Duke windows** are the slowest remaining job (~8.5k clips, ~4.5 h on T4 for comparable models). Use `RUN_SUBSET_FILTER` above for a focused session.

### After EmbeddingGemma finishes

1. Export `/kaggle/working/zero_shot_results/embeddinggemma2/` (Kaggle dataset or copy locally).
2. Refresh `results/checkpoints/embeddinggemma2_tier_a_v1.csv` from complete JSON (or extend the CSV manually).
3. Update Duke row in [`docs/full_benchmark_scores.md`](../docs/full_benchmark_scores.md) (remove ‡ footnote) and README limitations line.
4. Optional: add EmbeddingGemma **video_ms_per_item** to the T4 timing table in full scores (from JSON fields).
5. Re-run [`scripts/make_figures.py`](../scripts/make_figures.py) if headline figures should include updated Gemma numbers.

---

## `kaggle-zero-shot.ipynb` (main Tier-A grid)

**Purpose:** Frozen zero-shot on TVPReid, RSTPReid, GroOT-MOT17 for the model groups below.

| `ENV_GROUP` | Models | Kaggle session |
|-------------|--------|----------------|
| `base` | siglip2, pe_core_l14, irra, internvideo2_s2_1b, xclip, languagebind, internvideo2 | One session; HF secret `hugging_face` for gated InternVideo2-1B-s2 |
| `extended_modern` | openai_clip_vit_l14, jina_clip_v2, qwen3_vl_embed_2b | Separate session (transformers pin) |
| `extended_gme` | gme_qwen2_vl_2b | Separate session (different transformers pin) |

**EmbeddingGemma 2** is **not** in this notebook — use `embeddinggemma2_tier_a_zero_shot.ipynb`.

**If you re-run `base`:** older result JSON may omit `global_local`. Re-run **§4** only; §5.3b documents Type C pooling bars.

**If you combine groups:** publish each session’s `zero_shot_results/<group>/` and mount all folders before the summary cells.

---

## `colab_extended_zero_shot.ipynb`

**Purpose:** Same metrics as Kaggle, but **one model per runtime** and **TVPReid subsets only** (prid / ilids / duke).

- Start with `MODEL = "qwen3_vl_embed_2b"` and `SUBSETS = ["prid"]`; expand subsets after PRID `uniform8` passes the gate in §7.
- Switching between **modern** and **gme** profiles requires a **runtime restart** after the install cell.
- Results default to `/content/tvpreid_extended_results/` (or Drive if `MOUNT_DRIVE = True`).

**Left to run:** nothing required for the published extended checkpoint unless you add a new model or subset.

---

## ActivityNet notebooks (archived)

Large notebooks with **saved outputs** (papermill metadata). Use as reference or to reproduce the 1,000-video `val1` sample in [`docs/full_benchmark_scores.md`](../docs/full_benchmark_scores.md).

| File | Models covered |
|------|----------------|
| `activity-net-siglip-irra-xclip-perception.ipynb` | SigLIP 2, IRRA, X-CLIP, Perception Encoder |
| `activity-net-languagebind-internvideo.ipynb` | LanguageBind, InternVideo2 variants |

**Left to run:** none for the documented table unless you change the sample or add models.

---

## `pooling_methods_comparison.ipynb`

**Purpose:** Encode once, compare `mean` / `query_max` / `global_local` on a single dataset subset (default `pe_core_l14` + `tvpreid:prid`).

**Left to run:** only if you want new model/dataset ablations; not required for Tier-A publication.

---

## `kaggle_finetuning.ipynb`

**Purpose:** Partial fine-tune InternVideo2-1B-s2 on TVPReid **train**, then eval on test (different from zero-shot headline).

**Left to run:** entirely optional; needs HF gated checkpoint + `hugging_face` secret.

---

## Check pending jobs from JSON (any notebook)

From repo root, with results under `results/zero_shot_results/embeddinggemma2/`:

```python
import json
from pathlib import Path

RESULTS_DIR = Path("results/zero_shot_results/embeddinggemma2")
for path in sorted(RESULTS_DIR.rglob("*.json")):
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("status") != "complete":
        print("incomplete:", path.relative_to(RESULTS_DIR))
```

The EmbeddingGemma notebook §5 summary cell lists all **complete** rows in a table.
