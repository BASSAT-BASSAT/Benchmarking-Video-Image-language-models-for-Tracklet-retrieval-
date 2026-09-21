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
