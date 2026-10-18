# Full zero-shot benchmark scores

Complete **Rank-1 / mAP** tables for all models we ran on Tier-A and ActivityNet.
Headline README tables show only the top tier; nothing is deleted here.

**Protocol:** frozen encoders, cosine similarity, best temporal protocol and pool per
cell (same rule as the main README). **EmbeddingGemma 2** uses a separate env;
Only TVPReid **Duke** `reid_8fps_n64` may still be missing from JSON if that run was interrupted; GroOT and the rest of Tier-A are complete on Kaggle.

**Machine-readable checkpoints:**

- [`results/checkpoints/embeddinggemma2_tier_a_v1.csv`](../results/checkpoints/embeddinggemma2_tier_a_v1.csv)
- [`results/checkpoints/extended_zero_shot_4models_3datasets_v1.csv`](../results/checkpoints/extended_zero_shot_4models_3datasets_v1.csv) (OpenAI CLIP, Jina v2, GME-Qwen2-VL, Qwen3 — TVPReid subsets only)

**Notebooks:** [`kaggle-zero-shot.ipynb`](../notebooks/kaggle-zero-shot.ipynb) (base group),
[`colab_extended_zero_shot.ipynb`](../notebooks/colab_extended_zero_shot.ipynb),
[`embeddinggemma2_tier_a_zero_shot.ipynb`](../notebooks/embeddinggemma2_tier_a_zero_shot.ipynb).

**Protocols & pooling (uniform8 vs windows, mean vs query_max vs Type C):**
[`retrieval_protocol_and_pooling.md`](retrieval_protocol_and_pooling.md).

---

## TVPReid (video tracklets)

Gallery: PRID 185 · iLIDS 75 · Duke 603.

| Model | PRID | iLIDS | Duke |
|---|---:|---:|---:|
| **IRRA** | **66.55 / 76.09** | **30.00 / 41.29** | **38.06 / 50.79** |
| EmbeddingGemma 2 † | **42.25 / 55.52** | 20.67 / 32.48 | 29.77 / 39.48 ‡ |
| Perception Encoder L/14 | 38.38 / 52.04 | 19.33 / 32.55 | 17.66 / 28.12 |
| SigLIP 2 So400m | 35.56 / 50.84 | 18.67 / 28.88 | 21.97 / 33.03 |
| InternVideo2 CLIP-S | 32.39 / 43.77 | 13.33 / 24.14 | 18.49 / 28.36 |
| Qwen3-VL-Embedding-2B | 24.65 / 37.00 | 11.33 / 21.37 | 13.18 / 22.73 |
| LanguageBind | 16.20 / 24.82 | 12.67 / 22.09 | 7.30 / 14.46 |
| GME-Qwen2-VL-2B § | 11.62 / 22.96 | 10.67 / 20.14 | 9.45 / 17.70 |
| OpenAI CLIP ViT-L/14 | 10.92 / 20.05 | 9.33 / 17.48 | 5.14 / 10.10 |
| Jina CLIP v2 | 7.75 / 14.41 | 11.33 / 19.47 | 4.15 / 9.80 |
| X-CLIP | 4.93 / 10.05 | 5.33 / 10.52 | 1.41 / 3.95 |
| InternVideo2-1B-s2 | 4.23 / 10.78 | 4.00 / 10.26 | 1.82 / 4.15 |

† Separate `embeddinggemma2` install. ‡ Duke window protocol incomplete in CSV.
§ GME evaluated on TVPReid only (no RSTP / GroOT in extended checkpoint).

---

## RSTPReid (still images)

2,000 queries · 1,000 gallery images · 200 identities.

| Model | Rank-1 / mAP |
|---|---:|
| **IRRA** | **53.30 / 39.71** |
| SigLIP 2 So400m | 41.00 / 29.34 |
| Perception Encoder L/14 | 37.40 / 29.09 |
| EmbeddingGemma 2 † | 35.30 / 24.97 |
| Qwen3-VL-Embedding-2B | 30.05 / 21.72 |
| InternVideo2 CLIP-S | 27.00 / 21.34 |
| LanguageBind | 15.55 / 11.36 |
| Jina CLIP v2 | 12.15 / 9.64 |
| OpenAI CLIP ViT-L/14 | 11.05 / 8.21 |
| InternVideo2-1B-s2 | 5.15 / 4.53 |
| X-CLIP | 1.40 / 2.26 |

---

## GroOT-MOT17

454 tracks · appearance, action, and combined caption types.

| Model | Appearance | Action | Combined |
|---|---:|---:|---:|
| Perception Encoder L/14 | 25.29 | 3.30 | **29.36** |
| IRRA | 24.71 | 2.20 | **29.36** |
| SigLIP 2 So400m | **25.58** | 2.20 | 27.91 |
| EmbeddingGemma 2 † | 21.80 | 1.98 | 26.45 |
| InternVideo2 CLIP-S | 19.19 | 2.64 | 24.71 |
| Qwen3-VL-Embedding-2B | 20.06 | **3.52** | 22.38 |
| LanguageBind | 14.83 | 2.20 | 16.28 |
| OpenAI CLIP ViT-L/14 | 11.92 | 1.76 | 10.17 |
| Jina CLIP v2 | 11.92 | 1.32 | 8.14 |
| X-CLIP | 2.62 | 0.44 | 2.91 |
| InternVideo2-1B-s2 | 0.87 | 0.66 | 1.45 |

GroOT **all** split (EmbeddingGemma 2, best Rank-1 / mAP in grid): uniform8 **10.53 / 16.49**;
`reid_8fps_n64` global_local **10.53 / 16.90** (query_max **10.53 / 16.69**).

---

## ActivityNet Captions (1,000-video `val1` sample)

Not published ActivityNet leaderboard numbers.

| Model | Paragraph | Sentence | Mean Rank-1 |
|---|---:|---:|---:|
| InternVideo2 CLIP-S | **69.80 / 80.24** | 39.73 / 52.87 | 54.76 |
| Perception Encoder L/14 | 68.10 / 78.87 | **41.91 / 54.83** | **55.01** |
| LanguageBind | 64.10 / 75.18 | 36.96 / 49.82 | 50.53 |
| SigLIP 2 So400m | 57.10 / 70.18 | 35.11 / 48.51 | 46.11 |
| InternVideo2-1B-s2 | 37.70 / 48.77 | 10.62 / 15.48 | 24.16 |
| X-CLIP | 28.60 / 41.53 | 14.04 / 23.59 | 21.32 |
| IRRA | 16.40 / 27.20 | 8.29 / 15.40 | 12.35 |

---

## T4 encode time (ms per 8-frame clip)

From base / extended Kaggle runs (hardware-specific).

| Model | ms / clip |
|---|---:|
| X-CLIP | 31.59 |
| IRRA | 35.60 |
| LanguageBind | 200.43 |
| Perception Encoder L/14 | 248.91 |
| InternVideo2-1B-s2 | 267.11 |
| InternVideo2 CLIP-S | 361.04 |
| Qwen3-VL-Embedding-2B | 342.51 |
| SigLIP 2 So400m | 436.66 |
| OpenAI CLIP ViT-L/14 | 106.27 |
| Jina CLIP v2 | 1328.85 |
| GME-Qwen2-VL-2B | 1749.55 |

EmbeddingGemma 2 timing not logged in the bundled CSV yet.

---

## EmbeddingGemma 2 — pooling grid (excerpt)

Full rows in [`embeddinggemma2_tier_a_v1.csv`](../results/checkpoints/embeddinggemma2_tier_a_v1.csv).

| Dataset | Subset | Protocol | Best pool | Rank-1 | mAP |
|---|---|---|---|---:|---:|
| tvpreid | prid | reid_8fps_n64 | max / query_max | 42.25 | 55.80 |
| tvpreid | ilids | uniform8 | — | 20.67 | 32.48 |
| tvpreid | duke | uniform8 | — | 29.77 | 39.48 |
| rstpreid | test | uniform8 | — | 35.30 | 24.97 |
| groot_mot17 | combined | reid_8fps_n64 | query_max | 26.45 | 37.31 |
