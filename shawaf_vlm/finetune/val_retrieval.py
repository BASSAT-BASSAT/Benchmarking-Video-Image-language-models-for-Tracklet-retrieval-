"""Text-to-gallery Rank-1 / mAP on the fine-tune val split (checkpoint selection).

Gallery items follow the eval protocol: a TVPReid video is one tracklet (its
frames mean-pooled), an RSTPReid image is one item. Every caption is a query whose
positives are the gallery items with the same ``person_id``.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from shawaf_vlm.finetune.data import FinetuneSample, _resolve_image


def gallery_key(sample: FinetuneSample) -> str:
    return sample.person_id if sample.source == "tvpreid" else f"{sample.person_id}|{sample.image}"


class _ImageOnly(Dataset):
    def __init__(self, samples, image_root, preprocess) -> None:
        self.samples = samples
        self.image_root = Path(image_root)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(_resolve_image(sample, self.image_root)).convert("RGB")
        return self.preprocess(image), index


def subsample_val(
    samples: list[FinetuneSample], max_identities: int | None, seed: int = 0
) -> list[FinetuneSample]:
    """Keep every frame of a random subset of identities (fixed across epochs)."""

    if not max_identities:
        return list(samples)
    pids = sorted({s.person_id for s in samples})
    if len(pids) <= max_identities:
        return list(samples)
    keep = set(random.Random(seed).sample(pids, max_identities))
    return [s for s in samples if s.person_id in keep]


@torch.no_grad()
def val_retrieval_metrics(
    model,
    tokenizer,
    samples: list[FinetuneSample],
    *,
    image_root,
    preprocess,
    device: str,
    batch_size: int = 128,
    num_workers: int = 2,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    use_cuda = device.startswith("cuda")

    loader = DataLoader(
        _ImageOnly(samples, image_root, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    chunks = []
    for pixels, _ in loader:
        with torch.autocast("cuda", enabled=use_cuda):
            out = model.encode_image(pixels.to(device))
        chunks.append(F.normalize(out.float(), dim=-1).cpu())
    feats = torch.cat(chunks)

    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, sample in enumerate(samples):
        groups.setdefault(gallery_key(sample), []).append(i)
    gallery = torch.stack([F.normalize(feats[idx].mean(0), dim=-1) for idx in groups.values()])
    gallery_pids = [samples[idx[0]].person_id for idx in groups.values()]

    captions: list[str] = []
    query_pids: list[str] = []
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        for caption in sample.captions:
            if (sample.person_id, caption) in seen:
                continue
            seen.add((sample.person_id, caption))
            captions.append(caption)
            query_pids.append(sample.person_id)

    text_chunks = []
    for start in range(0, len(captions), batch_size):
        tokens = tokenizer(captions[start : start + batch_size])
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        with torch.autocast("cuda", enabled=use_cuda):
            out = model.encode_text(tokens)
        text_chunks.append(F.normalize(out.float(), dim=-1).cpu())
    text = torch.cat(text_chunks)

    pid_index = {pid: i for i, pid in enumerate(sorted(set(gallery_pids) | set(query_pids)))}
    g = torch.tensor([pid_index[p] for p in gallery_pids])
    q = torch.tensor([pid_index[p] for p in query_pids])
    sims = text @ gallery.T
    order = sims.argsort(dim=1, descending=True)
    matches = (g[order] == q[:, None]).float()

    r1 = matches[:, 0].mean().item()
    r5 = (matches[:, :5].sum(1) > 0).float().mean().item()
    hits = matches.cumsum(1)
    ranks = torch.arange(1, matches.shape[1] + 1, dtype=torch.float32)
    precision = hits / ranks
    n_pos = matches.sum(1).clamp_min(1.0)
    mean_ap = ((precision * matches).sum(1) / n_pos).mean().item()

    if was_training:
        model.train()
    return {
        "val_r1": 100.0 * r1,
        "val_r5": 100.0 * r5,
        "val_map": 100.0 * mean_ap,
        "val_queries": float(len(captions)),
        "val_gallery": float(len(gallery_pids)),
    }
