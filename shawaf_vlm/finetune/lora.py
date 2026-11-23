"""Lightweight LoRA for open_clip / timm PE towers (no bitsandbytes / QLoRA).

QLoRA = 4-bit base + LoRA. PE-B fits in fp16 on a T4; we only need LoRA adapters
on attention Linears so the base stays frozen and checkpoints stay mergeable.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in ``nn.Linear`` with frozen base weights + trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self._merged or self.rank == 0:
            return result
        # x @ A^T @ B^T * scale  (A: r×in, B: out×r)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return result + update

    @torch.no_grad()
    def merge(self) -> None:
        if self._merged or self.rank == 0:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.add_(delta.to(dtype=self.base.weight.dtype))
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged or self.rank == 0:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.sub_(delta.to(dtype=self.base.weight.dtype))
        self._merged = False


_ATTN_LEAF = re.compile(r"(?:^|\.)(qkv|q_proj|k_proj|v_proj|out_proj)$")


def _is_vision_attn_linear(name: str) -> bool:
    n = name.replace("-", "_").lower()
    if not (n.startswith("visual") or ".visual." in f".{n}."):
        return False
    if _ATTN_LEAF.search(n):
        return True
    # timm Attention: ...attn.proj / ...attn.qkv
    return n.endswith(".attn.proj") or n.endswith(".attn.qkv")


def _is_text_attn_linear(name: str) -> bool:
    """Text-tower attention only (not MLP c_fc/c_proj)."""

    n = name.replace("-", "_").lower()
    if n.startswith("visual") or ".visual." in f".{n}.":
        return False
    if ".attn." not in n and not n.endswith(".attn"):
        return False
    return bool(_ATTN_LEAF.search(n)) or n.endswith(".attn.proj")


def discover_lora_targets(model: nn.Module, *, on_text: bool) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _is_vision_attn_linear(name) or (on_text and _is_text_attn_linear(name)):
            names.append(name)
    return names


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float | None = None,
    on_text: bool = False,
) -> list[str]:
    """Freeze the tower and wrap attention Linears with LoRA. Returns wrapped names."""

    alpha_v = float(alpha if alpha is not None else rank * 2)
    targets = discover_lora_targets(model, on_text=on_text)
    if not targets:
        raise RuntimeError(
            "No attention Linear modules matched for LoRA "
            f"(on_text={on_text}). Inspect named_modules() for qkv/out_proj."
        )

    for param in model.parameters():
        param.requires_grad = False

    wrapped: list[str] = []
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        child = getattr(parent, leaf)
        if isinstance(child, LoRALinear):
            wrapped.append(name)
            continue
        if not isinstance(child, nn.Linear):
            continue
        setattr(parent, leaf, LoRALinear(child, rank=rank, alpha=alpha_v))
        wrapped.append(name)

    # Keep logit_scale frozen unless the train loop unfreezes it.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    return wrapped


def iter_lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def merge_lora_(model: nn.Module) -> int:
    """Absorb adapters into base weights (for eval-compatible checkpoints)."""

    n = 0
    for module in iter_lora_modules(model):
        module.merge()
        n += 1
    return n


def unmerge_lora_(model: nn.Module) -> int:
    n = 0
    for module in iter_lora_modules(model):
        module.unmerge()
        n += 1
    return n


def lora_export_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a plain PE ``state_dict`` with LoRA merged (model left unmerged)."""

    merge_lora_(model)
    try:
        # Strip LoRA wrapper keys: only base.* leaves matter after merge.
        raw = model.state_dict()
        out: dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            if ".lora_A" in key or ".lora_B" in key:
                continue
            # LoRALinear stores base under ``...base.weight``; remap to original Linear keys.
            if ".base.weight" in key:
                out[key.replace(".base.weight", ".weight")] = value.detach().cpu().contiguous()
            elif ".base.bias" in key:
                out[key.replace(".base.bias", ".bias")] = value.detach().cpu().contiguous()
            elif "lora_" in key:
                continue
            else:
                out[key] = value.detach().cpu().contiguous()
        return out
    finally:
        unmerge_lora_(model)
