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
    Text-to-tracklet CMC / mAP plus CLIP4Clip / TVPR ranking stats.

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

    parts: list[dict[str, Any]] = []
    chunk = max(int(query_chunk), 1)
    for start in range(0, num_query, chunk):
        end = min(start + chunk, num_query)
        distmat = (1.0 - query[start:end] @ gallery.T).astype(np.float32)
        parts.append(
            _eval_ranking(
                distmat=distmat,
                query_pids=query_pids[start:end],
                gallery_pids=gallery_pids,
                query_camids=query_camids[start:end],
                gallery_camids=gallery_camids,
                max_rank=max_rank,
                junk_same_camera=junk_same_camera,
            )
        )
    return _summarize_ranking(parts, num_gallery=num_gallery, max_rank=max_rank)


def evaluate_from_distmat(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray | None = None,
    gallery_camids: np.ndarray | None = None,
    junk_same_camera: bool = False,
    max_rank: int = 50,
) -> dict[str, float]:
    """CMC / mAP from a precomputed (Q, G) distance matrix (1 - cosine)."""

    num_query, num_gallery = distmat.shape
    if query_camids is None or not junk_same_camera:
        query_camids = np.full(num_query, -1, dtype=np.int64)
        gallery_camids = np.arange(num_gallery, dtype=np.int64)
    elif gallery_camids is None:
        raise ValueError("gallery_camids is required when junk_same_camera=True")

    parts = [
        _eval_ranking(
            distmat=distmat.astype(np.float32),
            query_pids=query_pids,
            gallery_pids=gallery_pids,
            query_camids=query_camids,
            gallery_camids=gallery_camids,
            max_rank=max_rank,
            junk_same_camera=junk_same_camera,
        )
    ]
    return _summarize_ranking(parts, num_gallery=num_gallery, max_rank=max_rank)


def format_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        "Text-to-tracklet evaluation",
        f"  Rank-1  : {metrics['Rank-1']:.2f}",
        f"  Rank-5  : {metrics['Rank-5']:.2f}",
        f"  Rank-10 : {metrics['Rank-10']:.2f}",
        f"  Rank-20 : {metrics.get('Rank-20', 0):.2f}",
        f"  Rank-50 : {metrics.get('Rank-50', 0):.2f}",
        f"  mAP     : {metrics['mAP']:.2f}",
        f"  MdR     : {metrics.get('MdR', 0):.2f}",
        f"  MnR     : {metrics.get('MnR', 0):.2f}",
        f"  nDCG@10 : {metrics.get('nDCG@10', 0):.2f}",
        f"  mINP    : {metrics.get('mINP', 0):.2f}",
        f"  valid Q : {int(metrics['num_valid_queries'])}",
    ]
    if "num_gallery" in metrics:
        lines.append(f"  gallery : {int(metrics['num_gallery'])}")
    if "num_clips" in metrics:
        lines.append(f"  clips   : {int(metrics['num_clips'])}")
    if "video_s" in metrics:
        lines.append(
            f"  time    : decode {metrics.get('decode_s', 0):.1f}s  "
            f"video {metrics['video_s']:.1f}s  "
            f"text {metrics.get('text_s', 0):.1f}s  "
            f"score {metrics.get('score_s', 0):.1f}s  "
            f"total {metrics.get('total_s', 0):.1f}s"
        )
        if metrics.get("video_ms_per_item", 0):
            lines.append(
                f"  video   : {metrics['video_ms_per_item']:.1f} ms/item"
            )
    if "peak_gpu_gb" in metrics:
        lines.append(
            f"  GPU     : peak {metrics['peak_gpu_gb']:.2f} GB allocated, "
            f"{metrics.get('reserved_gpu_gb', 0):.2f} GB reserved"
        )
    return "\n".join(lines)


def _eval_ranking(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int,
    junk_same_camera: bool,
) -> dict[str, Any]:
    num_query, num_gallery = distmat.shape
    rank_len = min(max_rank, num_gallery)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, np.newaxis]).astype(np.int32)

    all_cmc: list[np.ndarray] = []
    all_ap: list[float] = []
    first_ranks: list[float] = []
    last_ranks: list[float] = []
    ndcg10: list[float] = []

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
        prefix = cmc[:rank_len].astype(np.float32)
        if prefix.shape[0] < rank_len:
            fill = float(prefix[-1]) if prefix.size else 0.0
            prefix = np.concatenate(
                [prefix, np.full(rank_len - prefix.shape[0], fill, dtype=np.float32)]
            )
        all_cmc.append(prefix)

        num_rel = float(raw_cmc.sum())
        tmp_cmc = raw_cmc.cumsum().astype(np.float64)
        tmp_cmc = np.asarray(
            [value / (index + 1.0) for index, value in enumerate(tmp_cmc)]
        )
        all_ap.append(float((tmp_cmc * raw_cmc).sum() / num_rel))

        hit_positions = np.where(raw_cmc > 0)[0]
        first_ranks.append(float(hit_positions[0] + 1))
        last_ranks.append(float(hit_positions[-1] + 1))
        ndcg10.append(_ndcg_at_k(raw_cmc, k=10))

    return {
        "cmc": all_cmc,
        "ap": all_ap,
        "first_rank": first_ranks,
        "last_rank": last_ranks,
        "ndcg10": ndcg10,
    }


def _summarize_ranking(
    parts: list[dict[str, Any]],
    num_gallery: int,
    max_rank: int,
) -> dict[str, float]:
    all_cmc: list[np.ndarray] = []
    all_ap: list[float] = []
    first_ranks: list[float] = []
    last_ranks: list[float] = []
    ndcg10: list[float] = []
    for part in parts:
        all_cmc.extend(part["cmc"])
        all_ap.extend(part["ap"])
        first_ranks.extend(part["first_rank"])
        last_ranks.extend(part["last_rank"])
        ndcg10.extend(part["ndcg10"])

    if not all_ap:
        raise RuntimeError("All query identities are missing from the gallery.")

    cmc = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / float(len(all_ap))
    first = np.asarray(first_ranks, dtype=np.float64)
    last = np.asarray(last_ranks, dtype=np.float64)
    rank_len = int(cmc.shape[0])

    def _at(k: int) -> float:
        return float(cmc[min(k - 1, rank_len - 1)] * 100.0)

    return {
        "Rank-1": _at(1),
        "Rank-5": _at(5),
        "Rank-10": _at(10),
        "Rank-20": _at(20),
        "Rank-50": _at(min(50, max_rank)),
        "mAP": float(np.mean(all_ap) * 100.0),
        "MdR": float(np.median(first)),
        "MnR": float(np.mean(first)),
        "nDCG@10": float(np.mean(ndcg10) * 100.0),
        "mINP": float(np.mean(1.0 / last) * 100.0),
        "num_valid_queries": float(len(all_ap)),
        "num_gallery": float(num_gallery),
    }


def _ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    cutoff = min(int(k), int(relevance.shape[0]))
    if cutoff < 1:
        return 0.0
    gains = relevance[:cutoff].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    dcg = float((gains * discounts).sum())
    ideal_count = min(int(relevance.sum()), cutoff)
    if ideal_count < 1:
        return 0.0
    idcg = float((1.0 / np.log2(np.arange(2, ideal_count + 2))).sum())
    if idcg <= 0:
        return 0.0
    return dcg / idcg


# Kept for older tests / callers that imported the private helper.
def _eval_cmc_map(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int,
    junk_same_camera: bool,
) -> tuple[list[np.ndarray], list[float]]:
    ranked = _eval_ranking(
        distmat=distmat,
        query_pids=query_pids,
        gallery_pids=gallery_pids,
        query_camids=query_camids,
        gallery_camids=gallery_camids,
        max_rank=max_rank,
        junk_same_camera=junk_same_camera,
    )
    return ranked["cmc"], ranked["ap"]
