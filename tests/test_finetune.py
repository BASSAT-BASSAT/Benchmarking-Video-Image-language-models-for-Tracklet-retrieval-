"""CPU tests for fine-tune loss, packaging helpers, and a tiny train smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from shawaf_vlm.finetune.checkpoint import load_finetuned_pe_state, save_finetune_checkpoint
from shawaf_vlm.finetune.data import FinetuneSample, PersonCaptionDataset, load_jsonl
from shawaf_vlm.finetune.losses import identity_infonce


def test_identity_infonce_identical_features_is_finite() -> None:
    feats = torch.randn(4, 8)
    ids = torch.tensor([0, 0, 1, 1])
    loss = identity_infonce(feats, feats, ids, logit_scale=10.0)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_identity_infonce_grads_flow() -> None:
    image = torch.randn(3, 5, requires_grad=True)
    text = torch.randn(3, 5, requires_grad=True)
    ids = torch.arange(3)
    loss = identity_infonce(image, text, ids)
    loss.backward()
    assert image.grad is not None and float(image.grad.abs().sum()) > 0


def test_load_jsonl_rejects_forbidden_source(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "image": "images/a.jpg",
                "person_id": "alexandria_1",
                "captions": ["a person"],
                "split": "train",
                "source": "alexandria",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_jsonl(path)


def test_load_jsonl_allows_tvpreid_train_source(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "image": "images/a.jpg",
                "person_id": "tvpreid_prid_1",
                "captions": ["a person in black"],
                "split": "train",
                "source": "tvpreid",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_jsonl(path)
    assert len(rows) == 1 and rows[0].source == "tvpreid"


def test_person_caption_dataset_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    img = tmp_path / "images"
    img.mkdir()
    Image.new("RGB", (32, 64), color=(10, 20, 30)).save(img / "a.jpg")
    samples = [
        FinetuneSample(
            image="images/a.jpg",
            person_id="rstpreid_1",
            captions=("a boy in black", "boy wearing black"),
            split="train",
            source="rstpreid",
        )
    ]

    def preprocess(image):
        import torchvision.transforms.functional as TF

        return TF.to_tensor(image.resize((16, 16)))

    ds = PersonCaptionDataset(samples, image_root=tmp_path, preprocess=preprocess, augment=False)
    pixel, caption, pid = ds[0]
    assert pixel.shape[0] == 3
    assert isinstance(caption, str) and caption
    assert int(pid) == 0

    model = torch.nn.Linear(3, 3)
    ckpt = tmp_path / "best.pt"
    save_finetune_checkpoint(
        ckpt,
        model,
        base_key="pe_core_b16",
        base_checkpoint="hf-hub:timm/PE-Core-B-16",
        epoch=0,
        metrics={"val_loss": 1.0},
    )
    fresh = torch.nn.Linear(3, 3)
    meta = load_finetuned_pe_state(ckpt, fresh, strict=True)
    assert meta["base_key"] == "pe_core_b16"
    assert meta["epoch"] == 0


def test_train_pe_smoke_max_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    open_clip = pytest.importorskip("open_clip")

    # Tiny fake open_clip stand-in so the test stays offline and fast.
    class _Tok:
        def __call__(self, texts):
            return torch.zeros(len(texts), 4, dtype=torch.long)

    class _Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(3 * 8 * 8, 8)
            self.text = torch.nn.Linear(4, 8)
            self.logit_scale = torch.nn.Parameter(torch.tensor(2.0))

        def encode_image(self, pixels):
            flat = pixels.reshape(pixels.shape[0], -1)
            return self.visual(flat)

        def encode_text(self, tokens):
            return self.text(tokens.float())

    def fake_create(name):
        model = _Fake()

        def preprocess(image):
            import torchvision.transforms.functional as TF

            return TF.to_tensor(image.resize((8, 8)))

        return model, preprocess

    monkeypatch.setattr(open_clip, "create_model_from_pretrained", fake_create)
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda name: _Tok())

    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(4):
        Image.new("RGB", (24, 48), color=(i * 40, 10, 10)).save(images / f"{i}.jpg")
        rows.append(
            {
                "image": f"images/{i}.jpg",
                "person_id": f"rstpreid_{i // 2}",
                "captions": [f"person {i} in black"],
                "split": "train",
                "source": "rstpreid",
            }
        )
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    from shawaf_vlm.finetune.train import FinetuneConfig, train_pe

    out = tmp_path / "out"
    best = train_pe(
        FinetuneConfig(
            base_key="pe_core_b16",
            train_jsonl=[str(train_jsonl)],
            val_jsonl=[],
            image_root=str(tmp_path),
            output_dir=str(out),
            epochs=1,
            batch_size=2,
            device="cpu",
            max_steps=2,
            num_workers=0,
            log_every=1,
            warmup_steps=0,
            use_lora=False,
            unfreeze_last_blocks=None,
        )
    )
    assert Path(best).is_file()


def test_lora_wraps_vision_attn_and_exports_merged_keys() -> None:
    from shawaf_vlm.finetune.lora import LoRALinear, apply_lora, lora_export_state_dict

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Linear(8, 24)
            self.attn.proj = torch.nn.Linear(8, 8)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Sequential(Block(), Block())
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList([Block()])

    model = Fake()
    # named paths: visual.0.attn.qkv etc. — Sequential uses indices
    wrapped = apply_lora(model, rank=4, on_text=True)
    assert wrapped, "expected LoRA targets"
    assert all(isinstance(model.get_submodule(n), LoRALinear) for n in wrapped)
    exported = lora_export_state_dict(model)
    assert any(k.endswith("attn.qkv.weight") or k.endswith("qkv.weight") for k in exported)
    assert not any("lora_A" in k for k in exported)
