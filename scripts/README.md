# Scripts

| Script | Group | Purpose |
|--------|-------|---------|
| `eval_encoder.py` | Eval | CLI wrapper around `shawaf-vlm-eval` / TV-MARS paths |
| `download_tier_a.py` | Data | Download missing TVPReid, GroOT-MOT17, RSTPReid |
| `build_groot_mot17.py` | Data | Build GroOT-MOT17 metadata from MOT17 |
| `index_qdrant.py` | Demo | Encode gallery → numpy index (+ optional Qdrant). `--profile` writes stage timings. `--image-size`, `--fit native\|squash\|letterbox\|pad2`, `--max-frames`, `--select uniform\|quality\|diverse` |
| `prepare_alexandria.py` | Data | Unpack `Annotations/dayN-*.zip` → `Annotations/alexandria/`, crop tracklets → `datasets/alexandria` (`alexandria:all`, `dayN`, `dayN_k`) |
| `ingestion_sweep.py` | Speed / accuracy | One eval per config, one CSV row each (Rank-1/mAP per pool + timings). Config syntax `model[@size][:fit][/nFRAMES][+select]`; resumable |
| `plot_pareto.py` | Figures | Rank-1 vs ms per tracklet from the sweep CSV → `results/ingestion/pareto.png/.pdf` |
| `failure_analysis.py` | Analysis | Per-query ranks, miss buckets (duplicate caption, same clip, other clip) and contact sheets |
| `export_vision_onnx.py` | Speed | Experimental. Export a vision tower to ONNX. `--trt` prints `trtexec` and does not run it |
| `distill_pe.py` | Speed | Experimental. `--dry-run` checks person crops for PE distillation. Does not train (known checkpoint bug, see `shawaf_vlm/distill/__init__.py`) |
| `smoke_demo_stack.py` | Demo | Quick import / UI structure checks (no server) |
| `make_figures.py` | Figures | Regenerate `assets/*.png` (base models + `embeddinggemma2_tier_a_v1.csv`) |
| `prepare_local_annotations.py` | Data | Crop `Annotations/` → `datasets/shawaf_annotations` |
| `gen_embeddinggemma2_notebook.py` | Notebooks | Regenerate `embeddinggemma2_tier_a_zero_shot.ipynb` |
| `env/repair_crowd_gpu_deps.ps1` | Env | Pin Hub + Gradio after a broken demo install |

Run from repository root:

```bash
python scripts/download_tier_a.py
python scripts/index_qdrant.py --model siglip2 --dataset tvpreid:prid
pytest
python scripts/smoke_demo_stack.py
```
