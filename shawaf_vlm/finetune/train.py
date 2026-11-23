"""Train / resume Perception Encoder fine-tuning (Kaggle T4 or local GPU)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

from shawaf_vlm.finetune.checkpoint import load_finetuned_pe_state, save_finetune_checkpoint
from shawaf_vlm.finetune.data import PersonCaptionDataset, collate_batch, load_jsonl, merge_sources
from shawaf_vlm.finetune.losses import identity_infonce


@dataclass
class FinetuneConfig:
    base_key: str = "pe_core_b16"  # Run A; use pe_core_l14 for Run B
    train_jsonl: list[str] = field(default_factory=list)
    val_jsonl: list[str] = field(default_factory=list)
    image_root: str = "."
    output_dir: str = "results/finetune/pe_b16"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_workers: int = 2
    fit: str = "letterbox"
    image_size: int | None = None  # None = model default (224 for B, 336 for L)
    device: str = "cuda"
    seed: int = 0
    resume: str | None = None
    # Freeze early vision; train last N blocks + text. None = full FT (overfits fast on PEDES).
    unfreeze_last_blocks: int | None = 6
    use_lora: bool = False
    lora_rank: int = 8
    grad_checkpoint: bool = False
    # Keep CLIP temperature fixed — letting it grow sharpens logits and accelerates overfit.
    freeze_logit_scale: bool = True
    max_steps: int | None = None  # smoke / CI
    log_every: int = 20
    # Kaggle /kaggle/working is tight: keep only last.pt + best.pt (no optimizer).
    save_every_epoch: bool = False
    save_optimizer: bool = False
    # Stop when val_loss has not improved for this many epochs (None = run all epochs).
    early_stop_patience: int | None = 2


BASE_CHECKPOINTS = {
    "pe_core_b16": "hf-hub:timm/PE-Core-B-16",
    "pe_core_l14": "hf-hub:timm/PE-Core-L-14-336",
}


def train_pe(config: FinetuneConfig) -> Path:
    """Run fine-tuning and return the best checkpoint path."""

    torch.manual_seed(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    device = _resolve_device(config.device)
    model, preprocess, tokenizer, base_ckpt = _build_trainable_pe(config, device)
    train_rows = merge_sources([Path(p) for p in config.train_jsonl])
    if not train_rows:
        raise ValueError("No training rows — pass at least one train jsonl")
    train_ds = PersonCaptionDataset(
        train_rows,
        image_root=config.image_root,
        preprocess=preprocess,
        augment=True,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )

    val_loader = None
    if config.val_jsonl:
        val_rows = merge_sources([Path(p) for p in config.val_jsonl])
        if val_rows:
            val_ds = PersonCaptionDataset(
                val_rows,
                image_root=config.image_root,
                preprocess=preprocess,
                augment=False,
                seed=config.seed + 1,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_batch,
            )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * max(1, len(train_loader)))
    if config.max_steps:
        total_steps = min(total_steps, config.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_cuda = device.startswith("cuda")
    try:
        scaler = GradScaler("cuda", enabled=use_cuda)
    except TypeError:  # older torch.cuda.amp.GradScaler
        scaler = GradScaler(enabled=use_cuda)

    start_epoch = 0
    best_val = float("inf")
    best_path = output / "best.pt"
    epochs_without_improve = 0
    if config.resume:
        meta = load_finetuned_pe_state(config.resume, model, strict=False)
        start_epoch = int(meta.get("epoch", 0)) + 1
        print(f"resumed from {config.resume} at epoch {start_epoch}", flush=True)

    global_step = 0
    history: list[dict] = []
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.perf_counter()
        for pixels, captions, person_ids in train_loader:
            pixels = pixels.to(device, non_blocking=True)
            person_ids = person_ids.to(device)
            tokens = tokenizer(captions)
            if hasattr(tokens, "to"):
                tokens = tokens.to(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                amp_ctx = autocast("cuda", enabled=use_cuda)
            except TypeError:
                amp_ctx = autocast(enabled=use_cuda)
            with amp_ctx:
                image_feat = model.encode_image(pixels)
                text_feat = model.encode_text(tokens)
                logit_scale = getattr(model, "logit_scale", torch.tensor(4.6052, device=device))
                loss = identity_infonce(image_feat, text_feat, person_ids, logit_scale=logit_scale)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if global_step >= config.warmup_steps:
                scheduler.step()
            running += float(loss.detach())
            n_batches += 1
            global_step += 1
            if config.log_every and global_step % config.log_every == 0:
                print(
                    f"epoch {epoch} step {global_step} loss {running / n_batches:.4f}",
                    flush=True,
                )
            if config.max_steps and global_step >= config.max_steps:
                break
        train_loss = running / max(n_batches, 1)
        val_loss = _eval_loss(model, tokenizer, val_loader, device) if val_loader else train_loss
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.perf_counter() - t0,
            "step": global_step,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        opt = optimizer if config.save_optimizer else None
        last_path = output / "last.pt"
        save_finetune_checkpoint(
            last_path,
            model,
            base_key=config.base_key,
            base_checkpoint=base_ckpt,
            epoch=epoch,
            metrics=metrics,
            optimizer=opt,
        )
        if config.save_every_epoch:
            save_finetune_checkpoint(
                output / f"epoch_{epoch:03d}.pt",
                model,
                base_key=config.base_key,
                base_checkpoint=base_ckpt,
                epoch=epoch,
                metrics=metrics,
                optimizer=opt,
            )
        if val_loss <= best_val:
            best_val = val_loss
            epochs_without_improve = 0
            save_finetune_checkpoint(
                best_path,
                model,
                base_key=config.base_key,
                base_checkpoint=base_ckpt,
                epoch=epoch,
                metrics=metrics,
            )
            print(f"new best val_loss={val_loss:.4f} → {best_path}", flush=True)
        else:
            epochs_without_improve += 1
            print(
                f"no val improve ({epochs_without_improve}"
                f"{'' if config.early_stop_patience is None else f'/{config.early_stop_patience}'})"
                f" best={best_val:.4f}",
                flush=True,
            )
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if config.max_steps and global_step >= config.max_steps:
            break
        if (
            config.early_stop_patience is not None
            and epochs_without_improve >= config.early_stop_patience
        ):
            print(
                f"early stop at epoch {epoch}: no val improve for "
                f"{config.early_stop_patience} epoch(s); best={best_path}",
                flush=True,
            )
            break

    if not best_path.is_file():
        # No val improvement (or no val): promote last weights.
        last = output / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"No checkpoint written under {output}")
        best_path.write_bytes(last.read_bytes())
    return best_path


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; training on CPU", flush=True)
        return "cpu"
    return requested


def _build_trainable_pe(config: FinetuneConfig, device: str):
    import open_clip

    from shawaf_vlm.models.frame_encoders import rebuild_preprocess

    base_ckpt = BASE_CHECKPOINTS[config.base_key]
    model, preprocess = open_clip.create_model_from_pretrained(base_ckpt)
    tokenizer = open_clip.get_tokenizer(base_ckpt)
    preprocess = rebuild_preprocess(preprocess, config.image_size, config.fit)
    if config.grad_checkpoint and hasattr(model, "visual"):
        visual = model.visual
        if hasattr(visual, "set_grad_checkpointing"):
            visual.set_grad_checkpointing(True)
        elif hasattr(visual, "grad_checkpointing"):
            visual.grad_checkpointing = True

    if config.use_lora:
        _apply_lora(model, rank=config.lora_rank)
    elif config.unfreeze_last_blocks is not None:
        _freeze_except_last_blocks(model, config.unfreeze_last_blocks)

    if config.freeze_logit_scale:
        for name, param in model.named_parameters():
            if "logit_scale" in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params {trainable:,} / {total:,}"
        f" ({100.0 * trainable / max(total, 1):.1f}%)"
        f" unfreeze_last_blocks={config.unfreeze_last_blocks}"
        f" lora={config.use_lora} freeze_logit_scale={config.freeze_logit_scale}",
        flush=True,
    )

    model = model.to(device)
    model.train()
    return model, preprocess, tokenizer, base_ckpt


def _freeze_except_last_blocks(model: torch.nn.Module, last_n: int) -> None:
    for param in model.parameters():
        param.requires_grad = False
    # Train text tower; logit_scale handled separately via freeze_logit_scale.
    for name, param in model.named_parameters():
        if "text" in name.lower() or "logit_scale" in name:
            param.requires_grad = True
    visual = getattr(model, "visual", None)
    blocks = None
    if visual is not None:
        for attr in ("transformer", "trunk", "blocks"):
            candidate = getattr(visual, attr, None)
            if candidate is None:
                continue
            if hasattr(candidate, "resblocks"):
                blocks = list(candidate.resblocks)
                break
            if hasattr(candidate, "blocks"):
                blocks = list(candidate.blocks)
                break
            if isinstance(candidate, torch.nn.ModuleList):
                blocks = list(candidate)
                break
    if not blocks:
        # Fall back to full fine-tune of visual if structure is unknown.
        for param in model.visual.parameters():
            param.requires_grad = True
        return
    for block in blocks[-max(1, last_n) :]:
        for param in block.parameters():
            param.requires_grad = True
    # Projection / head
    for name, param in model.named_parameters():
        if any(key in name for key in ("proj", "head", "ln_post", "attn_pool")):
            param.requires_grad = True


def _apply_lora(model: torch.nn.Module, rank: int = 8) -> None:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Run B with use_lora=True needs peft. pip install peft") from exc

    # open_clip models are not always PEFT-ready; target Linear modules in visual+text.
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj", "c_fc", "c_proj", "out_proj", "fc1", "fc2"],
        bias="none",
    )
    # If get_peft_model fails (unsupported), caller should use unfreeze_last_blocks instead.
    get_peft_model(model, config)


@torch.no_grad()
def _eval_loss(model, tokenizer, loader, device: str) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    n = 0
    for pixels, captions, person_ids in loader:
        pixels = pixels.to(device)
        person_ids = person_ids.to(device)
        tokens = tokenizer(captions)
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        image_feat = model.encode_image(pixels)
        text_feat = model.encode_text(tokens)
        logit_scale = getattr(model, "logit_scale", torch.tensor(4.6052, device=device))
        loss = identity_infonce(image_feat, text_feat, person_ids, logit_scale=logit_scale)
        total += float(loss)
        n += 1
    model.train()
    return total / max(n, 1)
