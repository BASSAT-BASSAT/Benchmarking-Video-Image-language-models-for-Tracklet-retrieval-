# Scripts

| Script | Group | Purpose |
|--------|-------|---------|
| `eval_encoder.py` | Eval | CLI wrapper around `shawaf-vlm-eval` / TV-MARS paths |
| `download_tier_a.py` | Data | Download missing TVPReid, GroOT-MOT17, RSTPReid |
| `build_groot_mot17.py` | Data | Build GroOT-MOT17 metadata from MOT17 |
| `index_qdrant.py` | Demo | Encode gallery → numpy index (+ optional Qdrant) |
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
