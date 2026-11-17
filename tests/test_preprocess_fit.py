"""Preprocess rebuild (size / fit) and timm trunk resize. No downloads."""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from shawaf_vlm.models.frame_encoders import rebuild_preprocess, resize_timm_trunk

PE_LIKE = T.Compose(
    [
        T.Resize((336, 336), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
MCLIP_LIKE = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop((256, 256)),
        T.ToTensor(),
        T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ]
)


def _tall_person() -> Image.Image:
    """Red head band on top, blue feet band at the bottom, grey body."""

    img = Image.new("RGB", (40, 100), (128, 128, 128))
    for x in range(40):
        for y in range(10):
            img.putpixel((x, y), (255, 0, 0))
            img.putpixel((x, 99 - y), (0, 0, 255))
    return img


def test_native_without_size_is_identity() -> None:
    assert rebuild_preprocess(PE_LIKE, None, "native") is PE_LIKE


def test_native_resize_keeps_normalization_and_interpolation() -> None:
    pre = rebuild_preprocess(PE_LIKE, 224, "native")
    resize = pre.transforms[0]
    assert resize.size == (224, 224)
    assert resize.interpolation == T.InterpolationMode.BILINEAR
    norm = pre.transforms[-1]
    assert tuple(norm.mean) == (0.5, 0.5, 0.5)
    assert tuple(pre(_tall_person()).shape) == (3, 224, 224)


def test_native_center_crop_drops_head_and_feet_letterbox_keeps_them() -> None:
    native = rebuild_preprocess(MCLIP_LIKE, None, "native")(_tall_person())
    boxed = rebuild_preprocess(MCLIP_LIKE, None, "letterbox")(_tall_person())
    assert tuple(boxed.shape) == (3, 256, 256)
    # centre crop of a 40x100 crop keeps the middle 40x40: no red head row
    assert native[0].max() < 0.9
    assert boxed[0].max() > 0.9 and boxed[2].max() > 0.9


def test_pad2_pads_only_to_one_to_two() -> None:
    from shawaf_vlm.models.frame_encoders import _Letterbox

    padded = _Letterbox((0, 0, 0), max_ratio=2.0)(Image.new("RGB", (40, 100)))
    assert padded.size == (50, 100)
    assert _Letterbox((0, 0, 0), max_ratio=2.0)(Image.new("RGB", (60, 100))).size == (60, 100)
    out = rebuild_preprocess(PE_LIKE, None, "pad2")(_tall_person())
    assert tuple(out.shape) == (3, 336, 336)


def test_make_prefit_for_processor_encoders() -> None:
    from shawaf_vlm.models.frame_encoders import make_prefit

    assert make_prefit(None) is None and make_prefit("native") is None
    assert make_prefit("letterbox")(Image.new("RGB", (40, 100))).size == (100, 100)
    assert make_prefit("pad2")(Image.new("RGB", (40, 100))).size == (50, 100)
    with pytest.raises(ValueError):
        make_prefit("squash")


def test_squash_is_square_and_unknown_fit_rejected() -> None:
    out = rebuild_preprocess(MCLIP_LIKE, 128, "squash")(_tall_person())
    assert tuple(out.shape) == (3, 128, 128)
    with pytest.raises(ValueError):
        rebuild_preprocess(PE_LIKE, 224, "stretch")


def test_resize_timm_trunk_on_random_eva() -> None:
    timm = pytest.importorskip("timm")
    from timm.models.eva import Eva

    trunk = Eva(
        img_size=64,
        patch_size=16,
        embed_dim=32,
        depth=1,
        num_heads=2,
        num_classes=0,
        use_rot_pos_emb=True,
        use_abs_pos_emb=True,
        ref_feat_shape=(4, 4),
    ).eval()
    visual = torch.nn.Module()
    visual.trunk = trunk
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ref = trunk(x)
        assert resize_timm_trunk(visual, 64)
        assert torch.equal(trunk(x), ref)
        assert resize_timm_trunk(visual, 96)
        out = trunk(torch.randn(2, 3, 96, 96))
    assert out.shape == (2, 32)
    assert trunk.patch_embed.grid_size == (6, 6)
    assert resize_timm_trunk(torch.nn.Linear(2, 2), 96) is False
