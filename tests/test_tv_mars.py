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


def test_sliding_windows_covers_the_end(tmp_path: Path) -> None:
    from shawaf_vlm.sampling import sliding_windows

    paths = [tmp_path / f"f{index:02d}.jpg" for index in range(20)]
    windows = sliding_windows(paths, window=8, stride=4)
    assert windows[0] == paths[:8]
    assert windows[-1] == paths[-8:]
    assert all(len(window) == 8 for window in windows)
    denser = sliding_windows(paths, window=8, stride=4)
    coarser = sliding_windows(paths, window=8, stride=8)
    assert len(denser) > len(coarser)


def test_clip_mean_and_query_max_pool() -> None:
    from shawaf_vlm.pooling import pool_clip_features, query_max_similarity

    clips = np.array([[3.0, 0.0], [2.0, 0.1]], dtype=np.float32)
    mean = pool_clip_features(clips, "mean")
    assert mean.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(mean), 1.0, atol=1e-5)
    text = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sim = query_max_similarity(text, [clips])
    assert sim.shape == (2, 1)
    assert sim[0, 0] > sim[1, 0]


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
    assert metrics["MdR"] == 1.0
    assert metrics["MnR"] == 1.0
    assert metrics["Rank-50"] == 100.0
    assert metrics["nDCG@10"] == 100.0
    assert metrics["mINP"] == 75.0


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
    assert "_build_internvideo2_model" in source
    assert "AutoModel.from_pretrained" not in source
    assert "install_flash_attn_stub()" in source
    assert "_force_naive_attention(config)" in source
    assert "trust_remote_code=True" in source


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


def test_internvideo2_manual_weight_loader() -> None:
    import inspect

    from shawaf_vlm.models.internvideo2 import (
        _build_internvideo2_model,
        _load_internvideo2_state_dict,
        _resolve_internvideo2_class,
    )

    assert "get_class_from_dynamic_module" in inspect.getsource(
        _resolve_internvideo2_class
    )
    assert "load_state_dict" in inspect.getsource(_build_internvideo2_model)
    assert "model.safetensors" in inspect.getsource(_load_internvideo2_state_dict)


def test_middle_frame_indices_match_internvideo_intervals() -> None:
    from shawaf_vlm.sampling import _target_indices, middle_frame_indices

    middle = middle_frame_indices(100, 4)
    uniform = _target_indices(100, 4)
    assert middle == [12, 37, 62, 87]
    assert middle != uniform
    assert middle_frame_indices(1, 4) == [0, 0, 0, 0]


def test_internvideo_criterions_import_is_not_top_level() -> None:
    import importlib

    from shawaf_vlm.models.internvideo2_s2 import (
        _register_internvideo_parent,
        multi_modality_dir,
    )

    try:
        root = multi_modality_dir()
    except FileNotFoundError:
        pytest.skip("InternVideo checkout is not present")
    _register_internvideo_parent(root)
    module = importlib.import_module("internvideo_mm.models.criterions")
    assert module.__package__ == "internvideo_mm.models"


def test_s2_registry_and_retrieval_json(tmp_path: Path) -> None:
    import torch
    from torch import nn

    from shawaf_vlm import __version__
    from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits
    from shawaf_vlm.finetune_internvideo2 import (
        freeze_partial,
        write_retrieval_json,
        write_s2_config,
    )
    from shawaf_vlm.models.internvideo2_s2 import unwrap_state_dict
    from shawaf_vlm.models.registry import all_specs

    assert __version__ == "0.1.16"
    spec = all_specs()["internvideo2_s2_1b"]
    assert spec.checkpoint == "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    splits = TVMarsSplits(
        query=[
            CaptionQuery("red jacket", 1, 0, "a", (video,)),
            CaptionQuery("black bag", 1, 0, "a", (video,)),
        ],
        gallery=[GalleryTracklet((video,), 1, 0, "a")],
        source="tvpreid:prid:train",
    )
    anno = write_retrieval_json(splits, tmp_path / "train.json")
    rows = json.loads(anno.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["caption"] == "red jacket"
    assert rows[0]["image"].endswith("clip.mp4")

    config_path = write_s2_config(
        tmp_path / "config.py",
        train_json=anno,
        val_json=anno,
        pretrained_path=tmp_path / "weights.pt",
        output_dir=tmp_path / "out",
    )
    text = config_path.read_text(encoding="utf-8")
    assert 'model_cls="InternVideo2_Stage2_visual"' in text
    assert "vtc=1.0" in text
    assert "vtm=0.0" in text
    assert "use_bf16 = False" in text
    assert "use_flash_attn=False" in text

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)

    class TextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([Block() for _ in range(4)])

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = nn.Module()
            self.vision_encoder.blocks = nn.ModuleList([Block() for _ in range(6)])
            self._text = TextEncoder()
            self.vision_proj = nn.Linear(2, 2)
            self.text_proj = nn.Linear(2, 2)
            self.temp = nn.Parameter(torch.ones([]))

        def get_text_encoder(self) -> TextEncoder:
            return self._text

    model = Tiny()
    freeze_partial(model, vision_last_blocks=2, text_last_layers=1)
    assert not model.vision_encoder.blocks[0].lin.weight.requires_grad
    assert model.vision_encoder.blocks[-1].lin.weight.requires_grad
    assert not model.get_text_encoder().encoder.layer[0].lin.weight.requires_grad
    assert model.get_text_encoder().encoder.layer[-1].lin.weight.requires_grad
    assert model.vision_proj.weight.requires_grad
    assert model.text_proj.weight.requires_grad
    assert model.temp.requires_grad

    wrapped = unwrap_state_dict({"module.vision_proj.weight": torch.zeros(1)})
    assert "vision_proj.weight" in wrapped


def test_fps_indices_cover_duration() -> None:
    from shawaf_vlm.sampling import _fps_target_indices

    # 100 frames at 25 fps is 4s; 2 fps → 8 samples, capped below max_frames.
    indices = _fps_target_indices(100, 25.0, 2.0, 32)
    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == 99


def test_select_clip_rows_derives_stride_8() -> None:
    from shawaf_vlm.eval_loop import _select_clip_rows

    clips = np.arange(14, dtype=np.float32).reshape(7, 2)
    picked = _select_clip_rows(clips, factor=2)
    assert picked.shape[0] == 4
    np.testing.assert_array_equal(picked[0], clips[0])
    np.testing.assert_array_equal(picked[-1], clips[-1])


def test_window_eval_pools_perfect_match(tmp_path: Path) -> None:
    from shawaf_vlm.data.tv_mars import CaptionQuery, GalleryTracklet, TVMarsSplits
    from shawaf_vlm.eval_loop import evaluate_text_to_tracklet_windows

    def make_track(pid: int, cam: int, count: int) -> GalleryTracklet:
        paths = []
        for index in range(count):
            path = tmp_path / f"{pid:04d}C{cam}T0001F{index:03d}.jpg"
            path.write_bytes(b"x")
            paths.append(path)
        return GalleryTracklet(
            crop_paths=tuple(paths),
            person_id=pid,
            camera_id=cam,
            track_id="T0001",
        )

    gallery = [make_track(1, 1, 20), make_track(2, 2, 20)]
    queries = [
        CaptionQuery(
            text="person one",
            person_id=1,
            camera_id=1,
            track_id="T0001",
            crop_paths=gallery[0].crop_paths[:1],
        ),
        CaptionQuery(
            text="person two",
            person_id=2,
            camera_id=2,
            track_id="T0001",
            crop_paths=gallery[1].crop_paths[:1],
        ),
    ]
    splits = TVMarsSplits(query=queries, gallery=gallery, source="unit")

    class FakeEncoder:
        name = "fake"

        def encode_videos(self, videos, batch_size=4):
            rows = []
            for clip in videos:
                vector = np.zeros(4, dtype=np.float32)
                vector[int(clip[0].name[:4])] = 1.0
                rows.append(vector)
            return np.stack(rows, axis=0)

        def encode_texts(self, texts, batch_size=32):
            lookup = {"person one": 1, "person two": 2}
            rows = []
            for text in texts:
                vector = np.zeros(4, dtype=np.float32)
                vector[lookup[text]] = 1.0
                rows.append(vector)
            return np.stack(rows, axis=0)

    scored = evaluate_text_to_tracklet_windows(
        encoder=FakeEncoder(),
        splits=splits,
        num_frames=8,
        stride=4,
        max_frames=32,
        pools=("mean", "mean_s8", "max", "query_max"),
    )
    for pool in ("mean", "mean_s8", "max", "query_max"):
        assert scored[pool]["Rank-1"] == 100.0
        assert scored[pool]["mAP"] == 100.0
        assert scored[pool]["MdR"] == 1.0
        assert scored[pool]["num_clips"] > len(gallery)
        assert "video_s" in scored[pool]
        assert "peak_gpu_gb" in scored[pool]
