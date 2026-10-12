"""LanguageBind / transformers shim tests (need compatible huggingface-hub + transformers)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_transformers


def test_remap_peft_state_dict_maps_base_layer() -> None:
    import torch
    from torch import nn

    from shawaf_vlm.models.languagebind_hf.compat import remap_peft_state_dict

    class _Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = nn.Linear(4, 4)

    class _Fake(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _Wrapped()

    model = _Fake()
    raw = {
        "q_proj.weight": torch.ones(4, 4),
        "q_proj.bias": torch.ones(4),
    }
    mapped = remap_peft_state_dict(raw, model)
    assert "q_proj.base_layer.weight" in mapped
    assert "q_proj.base_layer.bias" in mapped
    torch.testing.assert_close(mapped["q_proj.base_layer.weight"], raw["q_proj.weight"])


def test_clip_text_embeddings_clamps_oob_ids() -> None:
    import torch
    from types import SimpleNamespace

    from shawaf_vlm.models.languagebind_hf.compat import CLIPTextEmbeddings

    config = SimpleNamespace(
        hidden_size=8,
        vocab_size=10,
        max_position_embeddings=4,
    )
    embeddings = CLIPTextEmbeddings(config)
    ids = torch.tensor([[0, 9, 99, -1, 3]])
    out = embeddings(input_ids=ids)
    assert out.shape == (1, 4, 8)


def test_languagebind_video_config_type() -> None:
    from shawaf_vlm.models.languagebind_hf.configuration_video import (
        LanguageBindVideoConfig,
    )

    assert LanguageBindVideoConfig.model_type == "LanguageBindVideo"


def test_disable_incompatible_torchao_patches_peft_probe() -> None:
    import sys
    import types

    saved = {
        name: sys.modules.get(name)
        for name in ("peft", "peft.import_utils", "peft.tuners.lora.torchao")
    }
    fake_utils = types.ModuleType("peft.import_utils")

    def _boom() -> bool:
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.10.0, "
            "but only versions above 0.16.0 are supported"
        )

    fake_utils.is_torchao_available = _boom
    fake_peft = types.ModuleType("peft")
    fake_lora = types.ModuleType("peft.tuners.lora.torchao")
    fake_lora.is_torchao_available = _boom
    try:
        sys.modules["peft"] = fake_peft
        sys.modules["peft.import_utils"] = fake_utils
        sys.modules["peft.tuners.lora.torchao"] = fake_lora
        from shawaf_vlm.models.languagebind_hf.compat import disable_incompatible_torchao

        disable_incompatible_torchao()
        assert fake_utils.is_torchao_available() is False
        assert fake_lora.is_torchao_available() is False
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_flash_attn_stub_does_not_break_transformers_probe() -> None:
    from shawaf_vlm.models.internvideo2 import install_flash_attn_stub

    install_flash_attn_stub()
    from transformers.utils.import_utils import is_flash_attn_2_available

    assert is_flash_attn_2_available() in (True, False)


@pytest.mark.needs_transformers_v5
def test_transformers_bert_shims_restore_internvideo_imports() -> None:
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils as tokenization_utils

    from shawaf_vlm.models.internvideo2_s2 import (
        install_transformers_bert_shims,
        install_transformers_tokenizer_shims,
    )

    import transformers.tokenization_utils_sentencepiece as sentencepiece

    install_transformers_bert_shims()
    install_transformers_tokenizer_shims()
    assert callable(sentencepiece._is_control)
    assert callable(tokenization_utils._is_control)
    assert callable(tokenization_utils._is_punctuation)
    assert callable(tokenization_utils._is_whitespace)
    assert callable(modeling_utils.apply_chunking_to_forward)
    assert callable(modeling_utils.prune_linear_layer)
    assert callable(modeling_utils.find_pruneable_heads_and_indices)
    assert callable(modeling_utils.PreTrainedModel.get_head_mask)


@pytest.mark.needs_transformers_v5
def test_pretrained_shim_supplies_tied_weight_keys() -> None:
    import inspect

    import torch
    from torch import nn
    from transformers import BertConfig
    from transformers.modeling_utils import PreTrainedModel

    from shawaf_vlm.models.internvideo2_s2 import install_transformers_pretrained_shims

    install_transformers_pretrained_shims()

    class TinyBert(PreTrainedModel):
        config_class = BertConfig

        def __init__(self, config):
            super().__init__(config)
            self.proj = nn.Linear(4, 4)

        def _init_weights(self, module):
            return None

    config = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=100,
    )
    model = TinyBert(config)
    assert isinstance(model.all_tied_weights_keys, dict)
    del model.all_tied_weights_keys
    parameters = inspect.signature(PreTrainedModel.mark_tied_weights_as_initialized).parameters
    if "loading_info" in parameters:
        model.mark_tied_weights_as_initialized(loading_info=None)
    else:
        model.mark_tied_weights_as_initialized()
    assert isinstance(model.all_tied_weights_keys, dict)
    del model
    del torch


@pytest.mark.needs_transformers_v5
def test_vendored_bert_tokenizer_loads_vocab_before_parent_init(tmp_path: Path) -> None:
    from shawaf_vlm.models.internvideo2_s2 import (
        _patch_vendored_bert_tokenizer,
        _register_internvideo_parent,
        install_transformers_tokenizer_shims,
        multi_modality_dir,
    )

    try:
        root = multi_modality_dir()
    except FileNotFoundError:
        pytest.skip("InternVideo checkout is not present")
    install_transformers_tokenizer_shims()
    _register_internvideo_parent(root)
    import importlib

    vendored = importlib.import_module(
        "internvideo_mm.models.backbones.bert.tokenization_bert"
    )
    import sys

    sys.modules["models.backbones.bert.tokenization_bert"] = vendored
    _patch_vendored_bert_tokenizer()
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\n", encoding="utf-8")
    tokenizer = vendored.BertTokenizer(str(vocab))
    assert tokenizer.get_vocab()["red"] == 5
    assert tokenizer.vocab_size >= 5
