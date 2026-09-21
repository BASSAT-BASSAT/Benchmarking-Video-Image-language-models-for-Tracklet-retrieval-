"""Transformers 5 shims for the vendored LanguageBind video encoder.

LanguageBind's modeling file was written against transformers 4.30 and
imports symbols that CLIP no longer exports (`_expand_mask`, `clip_loss`,
and a CLIPAttention.forward signature with `causal_attention_mask`).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput


try:
    from transformers.models.clip.modeling_clip import CLIPOutput
except ImportError:  # pragma: no cover - depends on transformers version
    from dataclasses import dataclass

    @dataclass
    class CLIPOutput(ModelOutput):  # type: ignore[no-redef]
        loss: Optional[torch.FloatTensor] = None
        logits_per_image: Optional[torch.FloatTensor] = None
        logits_per_text: Optional[torch.FloatTensor] = None
        text_embeds: Optional[torch.FloatTensor] = None
        image_embeds: Optional[torch.FloatTensor] = None
        text_model_output: Optional[object] = None
        vision_model_output: Optional[object] = None

try:
    from transformers.models.clip.modeling_clip import CLIPTextEmbeddings
except ImportError:  # pragma: no cover
    CLIPTextEmbeddings = None  # type: ignore[misc, assignment]

try:
    from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
except ImportError:  # pragma: no cover
    class CLIPVisionModelWithProjection:  # type: ignore[no-redef]
        pass

try:
    from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection
except ImportError:  # pragma: no cover
    class CLIPTextModelWithProjection:  # type: ignore[no-redef]
        pass


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
) -> torch.Tensor:
    """Restore the CLIP helper removed from transformers >= 4.35."""

    bsz, src_len = mask.size()
    tgt_len = src_len if tgt_len is None else tgt_len
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    return (contrastive_loss(similarity) + contrastive_loss(similarity.t())) / 2.0


class CLIPMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class CLIPAttention(nn.Module):
    """CLIP attention with the transformers 4.30 forward signature LanguageBind uses."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim {self.embed_dim} is not divisible by num_heads {self.num_heads}"
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if causal_attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_output_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_output_weights = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_output_weights


def disable_incompatible_torchao() -> None:
    """Skip PEFT's torchao dispatcher when the installed torchao is too old.

    Kaggle currently ships torchao 0.10.0. Recent PEFT raises ImportError unless
    torchao>=0.16 even for ordinary nn.Linear LoRA, which LanguageBind uses.
    """

    import sys

    try:
        import peft.import_utils as import_utils
    except ImportError:
        return
    checker = getattr(import_utils, "is_torchao_available", None)
    if checker is None:
        return
    try:
        checker()
        return
    except Exception:
        pass

    def _unavailable() -> bool:
        return False

    import_utils.is_torchao_available = _unavailable
    for name, module in list(sys.modules.items()):
        if "peft" in name and hasattr(module, "is_torchao_available"):
            setattr(module, "is_torchao_available", _unavailable)
