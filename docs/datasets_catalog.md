# SHAWAF dataset catalog

Text → person/tracklet retrieval benchmarks aligned with SHAWAF Stage 1 (natural-language query, gallery of tracklets, identity positives).

| Tier | Dataset | Modality | Captions | Integrated in SHAWAF? |
|------|---------|----------|----------|------------------------|
| A | **TVPReid** (PRID / iLIDS / Duke) | Video tracklets | 2 per video (appearance + motion/scene) | Yes — `shawaf_vlm/data/tvpreid.py`, [HF mirror](https://huggingface.co/datasets/bassatbassat/TVPReid) |
| A | **RSTPReid** | Still images (5 cams) | 2 per image | Yes — `rstpreid.py` (Google Drive, not HF) |
| A | **GroOT-MOT17** | MOT17 tracklet mp4s | appearance / action / **combined** | Yes — `groot.py`, [HF mirror](https://huggingface.co/datasets/bassatbassat/GroOT-MOT17) |
| B | **TV-MARS** | Video | Large-scale text–video ReID | Loader stub — `tv_mars.py`; needs MARS crops + gated caption JSON |
| B | Official [TVPReid-Dataset](https://github.com/NjtechCVLab/TVPReid-Dataset) | Video | Same as Tier A | Redundant with HF mirror |
| C | **CUHK-PEDES** | Image | Sentences | Planned — standard image text-ReID |
| C | **ICFG-PEDES** | Image | Cross-scene | Planned |
| C | **UFine6926** | Image | Fine-grained | Planned |
| C | **PA-100K** | Image | Attribute phrases | Weak fit unless filtered |
| D | GroOT / Type-to-Track (original) | MOT + language | — | Distilled into GroOT-MOT17 mirror |
| D | MOT20 | Video | No public caption set | Not retrieval-ready without new labels |
| E | **RefAVA / RAVAR** | Video + bbox | Action classes | **Out of scope** (classification, not tracklet retrieval) |
| E | AVA raw | Video | Atomic actions | Out of scope |

## Download notes

- **TVPReid / GroOT:** public Hugging Face datasets; set `HF_TOKEN` for rate limits.
- **RSTPReid:** MSMT17 lineage — use `gdown` via `download_rstpreid()`; cannot be re-hosted on HF.
- Override roots: `SHAWAF_TVPREID_ROOT`, `SHAWAF_GROOT_ROOT`, `SHAWAF_RSTPREID_ROOT`.

## Phased integration (after demo v0)

1. TV-MARS test split when annotation files are available.
2. CUHK-PEDES test via generic image+JSON loader.
3. ICFG-PEDES / UFine6926 if thesis needs broader image-ReID coverage.
