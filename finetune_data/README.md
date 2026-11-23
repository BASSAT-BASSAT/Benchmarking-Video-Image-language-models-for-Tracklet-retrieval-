# Fine-tune data packages

**Recommended start:** RSTPReid + TVPReid **train/val** (downloaded on Kaggle).

```bash
# On Kaggle (Internet on) or locally:
python scripts/prepare_finetune_kaggle.py --out finetune_data

# Or one source at a time:
python scripts/build_finetune_dataset.py --source rstpreid
python scripts/build_finetune_dataset.py --source tvpreid
```

Notebook: [`notebooks/finetune_pe_kaggle.ipynb`](../notebooks/finetune_pe_kaggle.ipynb) clones the repo, downloads both, trains PE-B (Run A).

| Source | Split used | Notes |
|---|---|---|
| RSTPReid | official train + val | Still images; test held out |
| TVPReid | official train + val (PRID/iLIDS/Duke) | Videos → 8 fps dense (max 64) + sliding window 8/4 stills; test held out |
| CUHK / ICFG | optional later | Need licence / Hub access |

**Never train on:** Alexandria, GroOT, any `test` split.

Each package:

```
finetune_data/<source>/images/
finetune_data/<source>/train.jsonl
finetune_data/<source>/val.jsonl
```
