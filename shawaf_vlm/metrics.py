from __future__ import annotations

from typing import Any

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return features / norms


def cosine_similarity(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
) -> np.ndarray:
    query = l2_normalize(query_features)
    gallery = l2_normalize(gallery_features)
    return (query @ gallery.T).astype(np.float32)


def evaluate_text_retrieval(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray | None = None,
    gallery_camids: np.ndarray | None = None,
    junk_same_camera: bool = False,
    max_rank: int = 50,
    query_chunk: int = 256,
) -> dict[str, float]:
    """
    Text-to-tracklet CMC / mAP.

    By default same-camera junk is off: the query is a caption, not a
    camera view. Set junk_same_camera=True to use Market1501-style
    same-pid + same-camera removal.
    """

    query = l2_normalize(query_features)
    gallery = l2_normalize(gallery_features)
    num_query = int(query.shape[0])
    num_gallery = int(gallery.shape[0])
    if query_camids is None or not junk_same_camera:
        query_camids = np.full(num_query, -1, dtype=np.int64)
        gallery_camids = np.arange(num_gallery, dtype=np.int64)
    elif gallery_camids is None:
        raise ValueError("gallery_camids is required when junk_same_camera=True")

    all_cmc: list[np.ndarray] = []
    all_ap: list[float] = []
    chunk = max(int(query_chunk), 1)

    for start in range(0, num_query, chunk):
        end = min(start + chunk, num_query)
        distmat = (1.0 - query[start:end] @ gallery.T).astype(np.float32)
        chunk_cmc, chunk_ap = _eval_cmc_map(
            distmat=distmat,
            query_pids=query_pids[start:end],
            gallery_pids=gallery_pids,
            query_camids=query_camids[start:end],
            gallery_camids=gallery_camids,
            max_rank=max_rank,
            junk_same_camera=junk_same_camera,
        )
        all_cmc.extend(chunk_cmc)
        all_ap.extend(chunk_ap)

    if not all_ap:
        raise RuntimeError("All query identities are missing from the gallery.")

    cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / float(len(all_ap))
    return {
        "Rank-1": float(cmc[0] * 100.0),
        "Rank-5": float(cmc[min(4, len(cmc) - 1)] * 100.0),
        "Rank-10": float(cmc[min(9, len(cmc) - 1)] * 100.0),
        "Rank-20": float(cmc[min(19, len(cmc) - 1)] * 100.0),
        "mAP": float(np.mean(all_ap) * 100.0),
        "num_valid_queries": float(len(all_ap)),
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        "Text-to-tracklet evaluation",
        f"  Rank-1  : {metrics['Rank-1']:.2f}",
        f"  Rank-5  : {metrics['Rank-5']:.2f}",
        f"  Rank-10 : {metrics['Rank-10']:.2f}",
        f"  Rank-20 : {metrics.get('Rank-20', 0):.2f}",
        f"  mAP     : {metrics['mAP']:.2f}",
        f"  valid Q : {int(metrics['num_valid_queries'])}",
    ]
    return "\n".join(lines)


def _eval_cmc_map(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int,
    junk_same_camera: bool,
) -> tuple[list[np.ndarray], list[float]]:
    num_query, num_gallery = distmat.shape
    if num_gallery < max_rank:
        max_rank = num_gallery

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    all_cmc: list[np.ndarray] = []
    all_ap: list[float] = []

    for q_idx in range(num_query):
        order = indices[q_idx]
        if junk_same_camera:
            remove = (gallery_pids[order] == query_pids[q_idx]) & (
                gallery_camids[order] == query_camids[q_idx]
            )
            keep = np.invert(remove)
        else:
            keep = np.ones(num_gallery, dtype=bool)

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank].astype(np.float32))

        num_rel = float(raw_cmc.sum())
        tmp_cmc = raw_cmc.cumsum().astype(np.float64)
        tmp_cmc = np.asarray(
            [value / (index + 1.0) for index, value in enumerate(tmp_cmc)]
        )
        all_ap.append(float((tmp_cmc * raw_cmc).sum() / num_rel))

    return all_cmc, all_ap
