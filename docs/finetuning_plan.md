# Fine-tuning plan

**Goal:** a small encoder fine-tuned on public person-description data that
matches or beats frozen PE-L/14 on Alexandria and GroOT. Neither of those is
ever used for training. If it works, ingestion becomes about 10× cheaper with
no accuracy loss, which is the paper's headline result.

**Why:** frozen models plateau around 47–53 Rank-1 on Alexandria and about 30
on GroOT combined. Preprocessing fixes add at most a few points. Published
work on text-based person search shows that fine-tuning roughly doubles
zero-shot accuracy.

## 1. Data

| Dataset | Source | Size | Use |
|---|---|---|---|
| RSTPReid | Already local: `~/.cache/shawaf_vlm/RSTPReid` | 18,505 train images, 3,701 IDs, 2 captions each | Train + val. **Test stays held out** (it is a Tier-A benchmark) |
| CUHK-PEDES (train) | HF `MaulikMadhavi/CUHK-PEDES-processed` (34,054 rows = official train split; person ID = first 4 digits of the filename) | about 550 MB | Train |
| ICFG-PEDES | HF `cjc/ICFG-PEDES` (1.14 GB zip). **The zip holds a pickle file:** unpack in an isolated folder and convert without unpickling untrusted objects | | Train |
| SYNTH-PEDES (subset) | HF `tuandunghcmut/SYNTH-PEDES` | Take a subset only | Optional stage 0 |
| TBPS-CLIP cleaned annotations | Google Drive link in the TBPS-CLIP README | Small | Cross-check captions |

Licensing: the HF copies are unofficial re-uploads with no stated licence. Use
them for experiments, and send the official CUHK-PEDES and ICFG-PEDES request
forms in parallel so the paper can cite authorised access. Keep the Kaggle
datasets **private**.

Never train on Alexandria, GroOT, RSTPReid test, or TVPReid test. Training on
TVPReid train makes TVPReid results in-domain, so label them that way.

## 2. Packaging for Kaggle

`scripts/build_finetune_dataset.py` (to write) converts each source into:

```
finetune_data/<source>/images/...      crops, pre-resized to 384 px tall
finetune_data/<source>/train.jsonl     {"image", "person_id": "<source>_<id>", "captions": [...], "split": "train|val"}
```

Each source becomes its own private Kaggle dataset, uploaded with
`kaggle datasets create -p finetune_data/<source>`. The notebook clones the
GitHub repo, the same way `notebooks/kaggle_finetuning.ipynb` already does.

## 3. Training (Kaggle, 2× T4, 16 GB each)

- Kaggle limits: fp16 (T4 has no bf16), 12-hour sessions, about 30 GPU-hours
  per week, 20 GB `/kaggle/working`. The script checkpoints every epoch and
  resumes.
- Loss: CLIP InfoNCE with identity-aware positives (any caption of the same
  person is a match), as in IRRA and TBPS-CLIP. Augmentations: horizontal flip
  and random erasing. Crops are padded to a square, matching evaluation.
- **Run A:** PE-B/16 at 224, full fine-tune at a low learning rate.
  RSTPReid + CUHK-PEDES. About 10 epochs.
- **Run B:** PE-L/14 at 336, LoRA or the last 4–6 blocks unfrozen, with
  gradient checkpointing.
- Validation every epoch on RSTPReid val and CUHK val; keep the best checkpoint.
- New code: `shawaf_vlm/finetune/` (data, loss, train loop with resume),
  `notebooks/finetune_pe_kaggle.ipynb`, and CPU tests on tiny fake data.

## 4. Evaluation (local)

1. Download the best checkpoint.
2. Register it as `pe_core_b16_ft` / `pe_core_l14_ft` with an open_clip state-dict loader. This replaces the buggy `DistilledPeEncoder` path.
3. Run `scripts/ingestion_sweep.py` on `alexandria:all` and `groot:*` (zero-shot for both), plus TVPReid. Report RSTPReid test separately as in-domain.
4. Gate: fine-tuned PE-B ≥ frozen PE-L on Alexandria and GroOT.

## 5. Published references to compare against

| Model | Weights | Status |
|---|---|---|
| IRRA | In the registry | Run on Alexandria and GroOT |
| RaSa | Released (CUHK, ICFG and RSTP checkpoints on Google Drive). ALBEF ViT-B/16 + BERT, written for transformers 4.8 | Needs vendored code under `third_party/` |
| PLIP | Released (ResNet50 + BERT, Google Drive) | Needs vendored code |
| TBPS-CLIP | **No trained weights released** (only code and annotations) | Its recipe is our run A baseline |

## Order

1. Send the CUHK-PEDES and ICFG-PEDES request emails (user).
2. Converter + RSTPReid package; first Kaggle run as a pipeline check.
3. Add CUHK-PEDES and ICFG-PEDES, then full runs A and B.
4. Local evaluation and paper tables.
