# Paper claim (locked)

**Working title:** Efficiency and failure modes of text-to-tracklet person retrieval with appearance and action

## Contribution (not “we released a dataset”)

1. **Accuracy–ingestion Pareto on one consumer GPU** — measured Rank-1 / mAP against seconds-per-person for frozen VLMs under a fixed protocol (padded crops, 16 frames, `query_max`), with PE-L as the accuracy default and PE-B as the fast option (~6×).
2. **Action-aware evaluation protocol** — hard identity metrics plus meaning-based multi-match scoring; action-gap (full / appearance / action queries); action-vs-scene probe (`scripts/action_scene_probe.py`) so Alexandria’s verb signal is not mistaken for room bias alone.
3. **Fine-tuning transfer diagnosis** — public person-description fine-tune of PE-B/PE-L (never Alexandria/GroOT), gated by out-of-domain Alexandria + GroOT, with failure buckets and action-gap before/after. Headline experiment: *fine-tuned PE-B ≥ frozen PE-L*. If the gate fails, the paper reports where gains go (clothes lookalikes vs wrong-moment misses).

Alexandria is a **private test set**. GroOT-MOT17 and TVPReid are the public reproducibility checks.

## What is already measured (frozen)

| Setup | Hard R1 | Meaning R1 | s / person |
|---|---:|---:|---:|
| PE-L letterbox 16f (default) | 51.2 | 59.7 | 0.29 |
| PE-B 16f (fast) | 47.3 | 55.0 | 0.05 |
| PE-L old stretched 64f | 45.7 | 54.3 | 1.00 |

Same-cost ablations (larger crops, query weighting, sharpest frames) did **not** transfer. IRRA (person-search trained) loses to frozen PE-L on Alexandria — warn against assuming PEDES fine-tune automatically solves action.

## Bootstrap CIs (required in tables)

Alexandria n≈129 → one query ≈ 0.78 Rank-1 points; SE ≈ ±4 points. Every headline Rank-1 must show a bootstrap 95% CI:

```bash
python scripts/bootstrap_ci.py --ranks results/action_scene/pe_core_l14_letterbox_n16_per_query.csv --column full_rank1 --label pe_l14_frozen
```

Frozen PE-L letterbox 16f hard Rank-1: **51.2 [42.6, 59.7]** (n=129, 2000 bootstraps). See `results/bootstrap/ci.json`.

Agreement on GroOT (direction + rough magnitude) is required for any claim that a change “helps.”

## Action vs scene (Alexandria, PE-L letterbox 16f)

From `results/action_scene/pe_core_l14_letterbox_n16.json`:

| Metric | Value |
|---|---:|
| Full-gallery Rank-1 | 51.2% |
| Same-scene Rank-1 | 90.7% |
| Top-1 action precision | 72.9% |
| Diff-scene same−other action margin | +0.034 |

Same-scene Rank-1 ≫ full Rank-1 → strong scene/clip prior. Positive margin → a weak action signal remains after removing the query’s room. Paper must report both.

## Fine-tune runs (to fill)

| Run | Model | Data | Status |
|---|---|---|---|
| A | PE-B/16 full FT | RSTPReid train (+ CUHK when licensed) | Code ready: `scripts/train_pe_finetune.py --run A` |
| B | PE-L/14 last-4 / LoRA | RSTPReid + CUHK | Code ready: `scripts/train_pe_finetune.py --run B` |
| Eval | `pe_core_b16_ft` / `pe_core_l14_ft` | Alexandria + GroOT + action_gap + failure | `scripts/eval_finetuned.py` |

Set `SHAWAF_PE_B16_FT` / `SHAWAF_PE_L14_FT` to `best.pt` after Kaggle training.

## Semantic labels

- Soft + hijab-exclusion policy via `scripts/review_semantic_maybe.py --apply-soft` (see `shawaf_vlm/semantic_review.py`).
- After review pass: human yes≈79, no≈13; remaining open maybes in `datasets/alexandria/maybe_queue.csv` + `semantic_review.html`.
- PE-L letterbox 16f meaning Rank-1 after soft labels: **60.5** (was 59.7 with auto-yes only).
- Re-score: `python scripts/rescore.py --model pe_core_l14 --fit letterbox --max-frames 16 --from-cache`.

## Action vs scene

```bash
python scripts/action_scene_probe.py --model pe_core_l14 --fit letterbox --max-frames 16
```

Report `full_gallery_rank1`, `same_scene_rank1`, `top1_action_precision`, and `diff_scene_same_vs_other_action_margin`. Margin > 0 supports action signal beyond scene.

## Do not claim

- That Alexandria is the contribution.
- That frozen preprocessing still has large untapped Rank-1 gains.
- That person-search fine-tuning transfers without showing Alexandria/GroOT + action-gap.
