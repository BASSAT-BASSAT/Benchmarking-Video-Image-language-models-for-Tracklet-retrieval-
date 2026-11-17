"""Stage timings for gallery ingestion.

Stages:

- ``model_load``  build the encoder (weights, compile wrap)
- ``decode``      sample / decode frames per tracklet (video seek or crop listing)
- ``load``        read image files into RGB
- ``preprocess``  resize / normalize to pixel tensors
- ``forward``     vision tower
- ``pool``        window pooling and device-to-host copy
- ``save``        numpy index on disk
- ``upsert``      Qdrant

Pass ``sync`` (for example ``torch.cuda.synchronize``) so that asynchronous GPU
work is charged to the stage that launched it instead of the next one.

Call ``start()`` / ``stop()`` around the whole run to get true wall time. Then
``wall_s`` is that measurement, ``unaccounted_s`` is the time outside every
stage, and throughput divides by real wall time. Without ``start()`` the wall
time falls back to the sum of the stages.
"""

from __future__ import annotations

import time
from collections.abc import Callable

STAGES = ("model_load", "decode", "load", "preprocess", "forward", "pool", "save", "upsert")


class StageTimer:
    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        sync: Callable[[], None] | None = None,
    ) -> None:
        self._clock = clock or time.perf_counter
        self.sync = sync
        self.totals = {name: 0.0 for name in STAGES}
        self.tracklets = 0
        self.frames = 0
        self.info: dict[str, object] = {}
        self._wall_start: float | None = None
        self._wall: float | None = None

    def start(self) -> None:
        self._sync()
        self._wall_start = self._clock()
        self._wall = None

    def stop(self) -> None:
        if self._wall_start is None:
            return
        self._sync()
        self._wall = self._clock() - self._wall_start

    def stage(self, name: str) -> "_Stage":
        if name not in self.totals:
            known = ", ".join(STAGES)
            raise KeyError(f"Unknown stage {name!r}. Choose one of: {known}")
        return _Stage(self, name)

    def _sync(self) -> None:
        if self.sync is not None:
            self.sync()

    def summary(self) -> dict[str, object]:
        staged = float(sum(self.totals.values()))
        if self._wall is not None:
            wall = float(self._wall)
        elif self._wall_start is not None:
            wall = float(self._clock() - self._wall_start)
        else:
            wall = staged
        # Throughput excludes one-off model load so runs with and without a
        # warm encoder are comparable; both numbers are reported.
        ingest = max(wall - self.totals["model_load"], 0.0)
        out: dict[str, object] = {
            "stages_s": {name: self.totals[name] for name in STAGES},
            "wall_s": wall,
            "ingest_s": ingest,
            "unaccounted_s": max(wall - staged, 0.0),
            "tracklets": int(self.tracklets),
            "frames": int(self.frames),
            "tracklets_per_s": (self.tracklets / wall) if wall > 0 else 0.0,
            "frames_per_s": (self.frames / wall) if wall > 0 else 0.0,
            "ingest_tracklets_per_s": (self.tracklets / ingest) if ingest > 0 else 0.0,
            "ingest_frames_per_s": (self.frames / ingest) if ingest > 0 else 0.0,
        }
        if self.info:
            out["info"] = dict(self.info)
        return out


class _Stage:
    def __init__(self, timer: StageTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_Stage":
        self._timer._sync()
        self._start = self._timer._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._timer._sync()
        self._timer.totals[self._name] += self._timer._clock() - self._start


def cuda_sync_for(device: str) -> Callable[[], None] | None:
    """Return ``torch.cuda.synchronize`` for CUDA devices, else ``None``."""

    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.synchronize


def device_info(device: str) -> dict[str, object]:
    info: dict[str, object] = {"device": str(device)}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    if str(device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
        info["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return info
