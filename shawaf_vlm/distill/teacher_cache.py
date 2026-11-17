"""Store teacher image embeddings so training does not re-encode crops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_teacher_cache(directory: Path | str, paths: list[Path | str], embeddings: np.ndarray) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"embeddings must be rank 2, got shape {array.shape}.")
    if array.shape[0] != len(paths):
        raise ValueError(f"Got {array.shape[0]} embeddings for {len(paths)} paths.")
    np.save(directory / "embeddings.npy", array)
    manifest = {
        "paths": [str(Path(path)) for path in paths],
        "count": len(paths),
        "dim": int(array.shape[1]),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def precompute_teacher_cache(encoder, paths: list[Path | str], directory: Path | str):
    """Encode each crop once and store the teacher rows beside a path manifest."""

    unique, features = encoder.encode_frames([Path(path) for path in paths])
    array = features.detach().cpu().float().numpy()
    write_teacher_cache(directory, unique, array)
    return unique, array


def read_teacher_cache(directory: Path | str) -> tuple[list[Path], np.ndarray]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(directory / "embeddings.npy")
    paths = [Path(path) for path in manifest["paths"]]
    if embeddings.shape[0] != len(paths):
        raise ValueError("Teacher cache manifest and embeddings.npy disagree on length.")
    return paths, embeddings
