"""Person-crop folders for feature distillation. Tier-A test splits are refused."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

_BENCHMARKS = {"tvpreid", "rstpreid", "groot", "groot_mot17"}


def refuse_tier_a_test(path: Path | str) -> None:
    parts = {part.lower() for part in Path(path).parts}
    if "test" in parts and parts & _BENCHMARKS:
        raise ValueError(f"Refusing Tier-A test split path: {path}")


class CropFolderDataset:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        refuse_tier_a_test(self.root)
        paths: list[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(self.root.rglob(pattern))
        self.paths = sorted(paths)
        for path in self.paths:
            refuse_tier_a_test(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        return path, rgb
