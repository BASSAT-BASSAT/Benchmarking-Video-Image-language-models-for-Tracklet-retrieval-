from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shawaf_vlm.data.tv_mars import (
    CaptionFilesNotFound,
    camera_id_from_crop,
    load_tv_mars,
    remap_crop_path,
)
from shawaf_vlm.metrics import evaluate_text_retrieval
from shawaf_vlm.sampling import sample_frame_paths


def test_remap_strips_author_prefix(tmp_path: Path) -> None:
    data_root = tmp_path / "MARS"
    raw = "/media/jqzhu/e/jjsu/datasets/bbox_train/0009/0009C1T0001F001.jpg"
    mapped = remap_crop_path(raw, data_root)
    assert mapped == data_root / "bbox_train" / "0009" / "0009C1T0001F001.jpg"


def test_camera_id_from_filename() -> None:
    assert camera_id_from_crop("0009C5T0004F001.jpg") == 5


def test_sample_pads_short_tracklets(tmp_path: Path) -> None:
    paths = [tmp_path / f"f{index}.jpg" for index in range(3)]
    sampled = sample_frame_paths(paths, num_frames=8)
    assert len(sampled) == 8
    assert sampled[-1] == paths[-1]


def test_load_flat_json_and_official_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "MARS"
    crop = data_root / "bbox_train" / "0009" / "0009C1T0001F001.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"fake")

    record = {
        "img_path": [f"/media/host/datasets/bbox_train/0009/{crop.name}"],
        "person_id": "0009",
        "track_id": "T0001",
        "captions": ["A person riding a bicycle."],
    }

    flat = tmp_path / "partical_dataset"
    flat.mkdir()
    (flat / "0009.json").write_text(json.dumps([record]), encoding="utf-8")
    splits = load_tv_mars(data_root, flat)
    assert splits.source == "flat_json"
    assert len(splits.query) == 1
    assert splits.query[0].person_id == 9
    assert splits.gallery[0].crop_paths[0] == crop

    official = tmp_path / "tv_mars"
    (official / "test_query_info").mkdir(parents=True)
    (official / "test_gallery_info").mkdir()
    (official / "test_query_info" / "q.json").write_text(
        json.dumps([record]),
        encoding="utf-8",
    )
    (official / "test_gallery_info" / "g.json").write_text(
        json.dumps([record]),
        encoding="utf-8",
    )
    official_splits = load_tv_mars(data_root, official)
    assert official_splits.source == "official"


def test_missing_captions_error(tmp_path: Path) -> None:
    data_root = tmp_path / "MARS"
    data_root.mkdir()
    with pytest.raises(CaptionFilesNotFound, match="agreement"):
        load_tv_mars(data_root, tmp_path / "missing")


def test_text_retrieval_ranks_matching_identity() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    metrics = evaluate_text_retrieval(
        query_features=query,
        gallery_features=gallery,
        query_pids=np.array([1, 2]),
        gallery_pids=np.array([1, 2, 2]),
    )
    assert metrics["Rank-1"] == 100.0
    assert metrics["num_valid_queries"] == 2.0


def test_l2_normalize_unwraps_model_output() -> None:
    from shawaf_vlm.models.runtime import l2_normalize_torch, unwrap_features

    class _Pooling:
        last_hidden_state = "tokens"
        pooler_output = np.array([[0.2, 0.8]], dtype=np.float32)

    unwrapped = unwrap_features(_Pooling())
    assert unwrapped.shape == (1, 2)
    nested = unwrap_features((_Pooling(),))
    np.testing.assert_array_equal(nested, unwrapped)

    import torch

    class _HFOutput:
        last_hidden_state = torch.zeros(1, 8, 2)
        pooler_output = torch.tensor([[3.0, 4.0]])

    normalized = l2_normalize_torch(_HFOutput())
    torch.testing.assert_close(normalized, torch.tensor([[0.6, 0.8]]))


def test_ensure_cuda_healthy_skips_cpu() -> None:
    from shawaf_vlm.models.runtime import ensure_cuda_healthy

    ensure_cuda_healthy("cpu")


def test_languagebind_skips_automodel() -> None:
    import inspect

    from shawaf_vlm.models.languagebind import LanguageBindEncoder

    source = inspect.getsource(LanguageBindEncoder.__init__)
    assert "LanguageBindVideo.from_pretrained" not in source
    assert "AutoModel.from_pretrained" not in source
    assert "CLIPTokenizer.from_pretrained" in source
    assert "_load_languagebind_weights" in source


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


def test_flash_attn_stub_satisfies_internvideo2_imports() -> None:
    import importlib.util

    from shawaf_vlm.models.internvideo2 import install_flash_attn_stub

    install_flash_attn_stub()
    install_flash_attn_stub()
    import flash_attn
    from flash_attn.modules.mlp import FusedMLP
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    assert flash_attn is not None
    assert flash_attn.__spec__ is not None
    assert importlib.util.find_spec("flash_attn") is not None
    assert FusedMLP is not None
    assert DropoutAddRMSNorm is not None
    assert callable(flash_attn_varlen_qkvpacked_func)
    assert callable(unpad_input)
    assert callable(pad_input)


def test_flash_attn_stub_does_not_break_transformers_probe() -> None:
    from shawaf_vlm.models.internvideo2 import install_flash_attn_stub

    install_flash_attn_stub()
    from transformers.utils.import_utils import is_flash_attn_2_available

    assert is_flash_attn_2_available() in (True, False)


def test_force_naive_attention_clears_flags() -> None:
    from types import SimpleNamespace

    from shawaf_vlm.models.internvideo2 import _force_naive_attention

    vision = {
        "use_flash_attn": True,
        "use_fused_mlp": True,
        "use_fused_rmsnorm": True,
    }
    _force_naive_attention(SimpleNamespace(model={"vision_encoder": vision}))
    assert vision["use_flash_attn"] is False
    assert vision["use_fused_mlp"] is False
    assert vision["use_fused_rmsnorm"] is False


def test_internvideo2_loader_installs_flash_stub() -> None:
    import inspect

    from shawaf_vlm.models.internvideo2 import InternVideo2Encoder

    source = inspect.getsource(InternVideo2Encoder.__init__)
    assert source.index("_purge_broken_flash_attn()") < source.index(
        "from transformers import"
    )
    assert source.index("from transformers import") < source.index(
        "install_flash_attn_stub()"
    )
    assert "_force_naive_attention(config)" in source
    assert "trust_remote_code=True" in source
    assert "low_cpu_mem_usage=False" in source
    assert "with _cpu_model_init()" in source


def test_flash_attn_stub_repairs_spec_less_module() -> None:
    import importlib.util
    import sys
    import types

    from shawaf_vlm.models.internvideo2 import install_flash_attn_stub

    sys.modules["flash_attn"] = types.ModuleType("flash_attn")
    install_flash_attn_stub()
    assert importlib.util.find_spec("flash_attn") is not None
    assert sys.modules["flash_attn"].__spec__ is not None


def test_purge_broken_flash_attn_makes_find_spec_safe() -> None:
    import importlib.util
    import sys
    import types

    from shawaf_vlm.models.internvideo2 import _purge_broken_flash_attn

    sys.modules["flash_attn"] = types.ModuleType("flash_attn")
    _purge_broken_flash_attn()
    importlib.util.find_spec("flash_attn")


def test_cpu_model_init_replaces_meta_device() -> None:
    import torch

    from shawaf_vlm.models.internvideo2 import _replace_meta_init_contexts

    out = _replace_meta_init_contexts([torch.device("meta"), torch.device("cpu")])
    assert [ctx.type for ctx in out] == ["cpu", "cpu"]
