"""TVPReid fine-tune helpers for InternVideo2-1B-s2.

The notebook writes retrieval JSON plus a stage-2 config, then launches this
module under torchrun. Training still uses InternVideo's ``tasks/pretrain.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from shawaf_vlm.data.tv_mars import TVMarsSplits

VISION_LAST_BLOCKS = 4
TEXT_LAST_LAYERS = 2
GRAD_ACCUM = 8


def retrieval_rows(splits: TVMarsSplits) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for query in splits.query:
        if not query.crop_paths:
            continue
        rows.append(
            {
                "image": str(Path(query.crop_paths[0])),
                "caption": query.text,
            }
        )
    return rows


def write_retrieval_json(
    splits_list: TVMarsSplits | list[TVMarsSplits],
    path: Path | str,
) -> Path:
    """Write InternVideo retrieval JSON: one ``{image, caption}`` row per caption."""

    if isinstance(splits_list, TVMarsSplits):
        groups = [splits_list]
    else:
        groups = list(splits_list)
    rows: list[dict[str, str]] = []
    for splits in groups:
        rows.extend(retrieval_rows(splits))
    if not rows:
        raise RuntimeError(f"No captions to write into {path}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} retrieval pairs to {out}", flush=True)
    return out


def freeze_partial(
    model,
    vision_last_blocks: int = VISION_LAST_BLOCKS,
    text_last_layers: int = TEXT_LAST_LAYERS,
) -> None:
    """Train the projection, the last video blocks, and the last BERT layers."""

    for param in model.parameters():
        param.requires_grad = False
    blocks = list(model.vision_encoder.blocks)
    for block in blocks[-min(vision_last_blocks, len(blocks)) :]:
        for param in block.parameters():
            param.requires_grad = True
    text = model.get_text_encoder() if hasattr(model, "get_text_encoder") else model.text_encoder
    layers = list(text.encoder.layer)
    for layer in layers[-min(text_last_layers, len(layers)) :]:
        for param in layer.parameters():
            param.requires_grad = True
    for name in ("vision_proj", "text_proj"):
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True
    if hasattr(model, "temp") and getattr(model.temp, "requires_grad", None) is not None:
        model.temp.requires_grad = True
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(
        f"Partial freeze: trainable {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
        f"(last {vision_last_blocks} video blocks, last {text_last_layers} BERT layers, projections)",
        flush=True,
    )


def write_s2_config(
    path: Path | str,
    *,
    train_json: Path | str,
    val_json: Path | str,
    pretrained_path: Path | str,
    output_dir: Path | str,
    epochs: int = 5,
    learning_rate: float = 5e-6,
    batch_size: int = 1,
    num_frames: int = 4,
) -> Path:
    """Write a 1B stage-2 config the official pretrain script can import."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = _CONFIG_TEMPLATE.format(
        train_json=_py(train_json),
        val_json=_py(val_json),
        pretrained_path=_py(pretrained_path),
        output_dir=_py(output_dir),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        num_frames=int(num_frames),
    )
    out.write_text(text, encoding="utf-8")
    print(f"Wrote stage-2 config {out}", flush=True)
    return out


def launch_training(
    config_path: Path | str,
    *,
    vision_last_blocks: int = VISION_LAST_BLOCKS,
    text_last_layers: int = TEXT_LAST_LAYERS,
    grad_accum: int = GRAD_ACCUM,
) -> None:
    """Run ``tasks/pretrain.py`` on one GPU via torchrun."""

    env = os.environ.copy()
    env["SHAWAF_FINETUNE_CHILD"] = "1"
    env["SHAWAF_S2_CONFIG"] = str(Path(config_path).resolve())
    env["SHAWAF_VISION_LAST"] = str(vision_last_blocks)
    env["SHAWAF_TEXT_LAST"] = str(text_last_layers)
    env["SHAWAF_GRAD_ACCUM"] = str(grad_accum)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "--master_port",
        "29511",
        "-m",
        "shawaf_vlm.finetune_internvideo2",
    ]
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, env=env)


def run_training_child() -> None:
    """Entry point inside the torchrun worker."""

    config_path = os.environ["SHAWAF_S2_CONFIG"]
    vision_last = int(os.environ.get("SHAWAF_VISION_LAST", VISION_LAST_BLOCKS))
    text_last = int(os.environ.get("SHAWAF_TEXT_LAST", TEXT_LAST_LAYERS))
    grad_accum = int(os.environ.get("SHAWAF_GRAD_ACCUM", "1"))

    from shawaf_vlm.models.internvideo2_s2 import ensure_internvideo_importable, multi_modality_dir

    root = multi_modality_dir()
    ensure_internvideo_importable()
    _install_decord_stub()
    os.chdir(root)
    _install_partial_freeze(vision_last, text_last)
    _install_cuda_losses()
    if grad_accum > 1:
        _install_grad_accum(grad_accum)

    sys.argv = [
        "pretrain.py",
        config_path,
        "evaluate",
        "False",
    ]
    from tasks.pretrain import main as train_main
    from utils.config_utils import setup_main

    config = setup_main()
    train_main(config)


def _install_decord_stub() -> None:
    """InternVideo imports decord even when the reader is PyAV."""

    try:
        import decord  # noqa: F401
        return
    except ImportError:
        pass
    import types

    module = types.ModuleType("decord")

    class VideoReader:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("decord is not installed; set video_reader_type='av'")

    class _Bridge:
        @staticmethod
        def set_bridge(*args, **kwargs) -> None:
            return None

    module.VideoReader = VideoReader
    module.bridge = _Bridge()
    sys.modules["decord"] = module
    print("decord is missing; using a stub so the av video reader can import", flush=True)


def _install_partial_freeze(vision_last: int, text_last: int) -> None:
    from models.internvideo2_stage2_visual import InternVideo2_Stage2_visual

    original = InternVideo2_Stage2_visual.__init__

    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        freeze_partial(
            self,
            vision_last_blocks=vision_last,
            text_last_layers=text_last,
        )

    InternVideo2_Stage2_visual.__init__ = wrapped


def _install_cuda_losses() -> None:
    import torch
    from models.internvideo2_stage2_visual import InternVideo2_Stage2_visual

    original = InternVideo2_Stage2_visual.forward

    def wrapped(self, *args, **kwargs):
        losses = original(self, *args, **kwargs)
        device = self.temp.device
        moved = {}
        for key, value in losses.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    InternVideo2_Stage2_visual.forward = wrapped


def _install_grad_accum(steps: int) -> None:
    import torch

    original_zero = torch.optim.Optimizer.zero_grad
    original_step = torch.optim.Optimizer.step
    counter = {"n": 0}

    def zero_grad(self, *args, **kwargs):
        if counter["n"] % steps == 0:
            original_zero(self, *args, **kwargs)

    def step(self, *args, **kwargs):
        counter["n"] += 1
        if counter["n"] % steps != 0:
            return None
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.div_(steps)
        return original_step(self, *args, **kwargs)

    torch.optim.Optimizer.zero_grad = zero_grad
    torch.optim.Optimizer.step = step

    scheduler_cls = getattr(torch.optim.lr_scheduler, "LRScheduler", None)
    if scheduler_cls is None:
        scheduler_cls = torch.optim.lr_scheduler._LRScheduler
    original_sched = scheduler_cls.step

    def sched_step(self, *args, **kwargs):
        if counter["n"] % steps == 0:
            return original_sched(self, *args, **kwargs)
        return None

    scheduler_cls.step = sched_step
    print(f"Gradient accumulation every {steps} steps", flush=True)


def _py(value: Path | str) -> str:
    return json.dumps(str(value))


_CONFIG_TEMPLATE = """\
import os
from configs.model import TextEncoders

train_json = {train_json}
val_json = {val_json}
num_frames = {num_frames}
num_frames_test = {num_frames}
batch_size = {batch_size}
batch_size_test = 1
max_txt_l = 40
num_workers = 2

train_file = dict(
    anno_path=train_json,
    data_root="",
    media_type="video",
)
test_file = dict(
    tvpreid_val=dict(
        anno_path=val_json,
        data_root="",
        media_type="video",
    )
)
test_types = ["tvpreid_val"]
best_key = ["tvpreid_val_match", "t2v_r1"]

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames=num_frames,
        sample_type="rand",
        num_frames_test=num_frames_test,
        sample_type_test="middle",
        random_aug=False,
        video_reader_type="av",
    ),
    max_txt_l=dict(image=max_txt_l, video=max_txt_l),
    batch_size=dict(image=batch_size, video=batch_size),
    batch_size_test=dict(image=batch_size_test, video=batch_size_test),
)

text_encoder = dict(TextEncoders["bert_large"])
text_encoder["config"] = os.path.join(os.getcwd(), "configs", "config_bert_large.json")

model = dict(
    model_cls="InternVideo2_Stage2_visual",
    vision_encoder=dict(
        name="pretrain_internvideo2_1b_patch14_224",
        img_size=224,
        num_frames=num_frames,
        tubelet_size=1,
        patch_size=14,
        d_model=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type="l2",
        clip_return_layer=6,
        clip_student_return_interval=1,
        pretrained=None,
        use_checkpoint=True,
        checkpoint_num=40,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        sep_image_video_pos_embed=True,
        keep_temporal=False,
        only_mask=True,
    ),
    text_encoder=text_encoder,
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
    find_unused_parameters=True,
    freeze_vision=False,
    freeze_text=False,
)

criterion = dict(
    loss_weight=dict(vtc=1.0, mlm=0.0, vtm=0.0, uta=0.0, mvm=0.0),
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1.0, 1.0],
    uta_image_only=False,
)

optimizer = dict(
    opt="adamW",
    lr={learning_rate},
    opt_betas=[0.9, 0.98],
    weight_decay=0.05,
    max_grad_norm=3.0,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs={epochs}, min_lr_multi=0.01, warmup_epochs=1)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)
use_half_precision = True
use_bf16 = False
gradient_checkpointing = True
use_flash_sdp = False
use_mem_efficient_sdp = False
compile_model = False

wandb = dict(enable=False, entity="", project="shawaf-internvideo2-s2")
dist_url = "env://"
device = "cuda"
mode = "pt"

output_dir = {output_dir}
resume = False
debug = False
log_freq = 10
seed = 42
save_latest = True
auto_resume = False
jump_evaluate = True
pretrained_path = {pretrained_path}
save_ckpt_iter = None
delete_ds_optim_states = True
deepspeed = dict(enable=False, stage=0)
"""


if __name__ == "__main__":
    if os.environ.get("SHAWAF_FINETUNE_CHILD") == "1":
        run_training_child()
    else:
        raise SystemExit(
            "Launch training with shawaf_vlm.finetune_internvideo2.launch_training"
        )
