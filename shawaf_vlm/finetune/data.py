"""Person-description fine-tune samples from packaged jsonl."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler

# Never train on these *sources* (held-out eval / gated).
# TVPReid *train/val* packaging uses source ``tvpreid`` and is allowed.
FORBIDDEN_SOURCES = frozenset(
    {
        "alexandria",
        "groot",
        "tv_mars",
        "rstpreid_test",
        "tvpreid_test",
    }
)


@dataclass(frozen=True)
class FinetuneSample:
    image: str
    person_id: str
    captions: tuple[str, ...]
    split: str
    source: str
    base_dir: str = ""  # package folder that contains images/


def load_jsonl(path: Path | str) -> list[FinetuneSample]:
    """Load ``train.jsonl`` / ``val.jsonl`` written by ``build_finetune_dataset``."""

    root = Path(path)
    base_dir = str(root.parent.resolve())
    rows: list[FinetuneSample] = []
    with root.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source = str(record.get("source") or root.parent.name)
            if source.lower() in FORBIDDEN_SOURCES:
                raise ValueError(f"Refusing to load forbidden training source {source!r}")
            captions = record.get("captions") or []
            if isinstance(captions, str):
                captions = [captions]
            captions = tuple(str(c).strip() for c in captions if str(c).strip())
            if not captions:
                continue
            rows.append(
                FinetuneSample(
                    image=str(record["image"]),
                    person_id=str(record["person_id"]),
                    captions=captions,
                    split=str(record.get("split") or "train"),
                    source=source,
                    base_dir=str(record.get("base_dir") or base_dir),
                )
            )
    return rows


def merge_sources(jsonl_paths: list[Path | str]) -> list[FinetuneSample]:
    rows: list[FinetuneSample] = []
    for path in jsonl_paths:
        rows.extend(load_jsonl(path))
    return rows


class PersonCaptionDataset(Dataset):
    """One image + one randomly chosen caption per ``__getitem__``."""

    def __init__(
        self,
        samples: list[FinetuneSample],
        *,
        image_root: Path | str,
        preprocess,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.image_root = Path(image_root)
        self.preprocess = preprocess
        self.augment = augment
        self._rng = random.Random(seed)
        self._pid_to_int = {
            pid: index for index, pid in enumerate(sorted({s.person_id for s in self.samples}))
        }

    def __len__(self) -> int:
        return len(self.samples)

    def person_id_int(self, person_id: str) -> int:
        return self._pid_to_int[person_id]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        path = _resolve_image(sample, self.image_root)
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = _train_augment(image, self._rng)
        caption = self._rng.choice(sample.captions)
        pixel = self.preprocess(image)
        return pixel, caption, self.person_id_int(sample.person_id)


def _resolve_image(sample: FinetuneSample, image_root: Path) -> Path:
    candidates = [
        Path(sample.image),
        image_root / sample.image,
    ]
    if sample.base_dir:
        candidates.append(Path(sample.base_dir) / sample.image)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing image {sample.image!r} (tried {[str(c) for c in candidates]})"
    )


def _train_augment(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = _random_erase(image, rng)
    return image


def _random_erase(image: Image.Image, rng: random.Random, area_ratio: float = 0.2) -> Image.Image:
    width, height = image.size
    erase_w = max(1, int(width * (area_ratio ** 0.5) * rng.uniform(0.5, 1.0)))
    erase_h = max(1, int(height * (area_ratio ** 0.5) * rng.uniform(0.5, 1.0)))
    left = rng.randint(0, max(0, width - erase_w))
    top = rng.randint(0, max(0, height - erase_h))
    patch = Image.new(
        "RGB",
        (erase_w, erase_h),
        (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)),
    )
    out = image.copy()
    out.paste(patch, (left, top))
    return out


class IdentityBalancedSampler(Sampler[int]):
    """Each epoch draws ``per_identity`` random rows per person_id.

    TVPReid windows yield dozens of near-duplicate frames per video while an
    RSTPReid identity has a handful of images; sampling per identity keeps the
    sources balanced and turns the extra frames into temporal augmentation.
    """

    def __init__(self, samples: list[FinetuneSample], per_identity: int = 2, seed: int = 0) -> None:
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            groups.setdefault(sample.person_id, []).append(index)
        self.groups = list(groups.values())
        self.per_identity = max(1, int(per_identity))
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.groups) * self.per_identity

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        picks: list[int] = []
        for members in self.groups:
            if len(members) >= self.per_identity:
                picks.extend(rng.sample(members, self.per_identity))
            else:
                picks.extend(rng.choice(members) for _ in range(self.per_identity))
        rng.shuffle(picks)
        return iter(picks)


def collate_batch(batch):
    import torch

    pixels = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    person_ids = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return pixels, captions, person_ids
