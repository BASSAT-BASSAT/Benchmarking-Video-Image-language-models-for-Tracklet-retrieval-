"""InternVideo2-1B stage-2 retrieval encoder (BERT + 1B video tower)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from shawaf_vlm.models.internvideo2 import install_flash_attn_stub, _purge_broken_flash_attn
from shawaf_vlm.models.runtime import (
    frames_to_uint8_tchw,
    l2_normalize_torch,
    place_model,
    resolve_device,
    to_numpy,
)

S2_1B_REPO = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
S2_1B_WEIGHT = "InternVideo2-stage2_1b-224p-f4.pt"
NUM_FRAMES = 4
EMBED_DIM = 512
MAX_TXT_LEN = 40
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def internvideo_root() -> Path:
    """Locate the nested InternVideo checkout."""

    env = os.environ.get("INTERNVIDEO_ROOT", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "InternVideo")
    candidates.append(Path("/kaggle/working/InternVideo"))
    for path in candidates:
        multi = path / "InternVideo2" / "multi_modality"
        if multi.is_dir():
            return path
    raise FileNotFoundError(
        "InternVideo checkout not found. Set INTERNVIDEO_ROOT or clone "
        "https://github.com/OpenGVLab/InternVideo.git to InternVideo/ "
        "or /kaggle/working/InternVideo."
    )


def multi_modality_dir() -> Path:
    return internvideo_root() / "InternVideo2" / "multi_modality"


_PACKAGE = "internvideo_mm"


def _clear_broken_models_package(root: Path) -> None:
    """Drop a top-level ``models`` import of this checkout.

    InternVideo's ``models.criterions`` uses ``from ..utils``. That only works
    when ``models`` is a subpackage. A previous import that put
    ``multi_modality`` itself on ``sys.path`` loads ``models`` as top-level and
    then raises ImportError.
    """

    existing = sys.modules.get("models")
    if existing is None or getattr(existing, "__name__", "") == f"{_PACKAGE}.models":
        return
    target = str((root / "models").resolve())
    file_path = str(getattr(existing, "__file__", "") or "")
    search_path = [str(path) for path in getattr(existing, "__path__", []) or []]
    if target != file_path and target not in search_path and not file_path.startswith(target):
        return
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]


def _alias_internvideo_modules() -> None:
    prefix = f"{_PACKAGE}."
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix):
            continue
        short = name[len(prefix) :]
        current = sys.modules.get(short)
        if current is None or current is module:
            sys.modules[short] = module


def _register_internvideo_parent(root: Path) -> None:
    """Register InternVideo's multi_modality tree as ``internvideo_mm``.

    ``models/__init__.py`` is not executed. Importing through this parent lets
    ``models.criterions`` resolve ``from ..utils`` instead of failing as a
    top-level package.
    """

    import types

    _clear_broken_models_package(root)
    if _PACKAGE not in sys.modules:
        parent = types.ModuleType(_PACKAGE)
        parent.__path__ = [str(root)]
        parent.__package__ = _PACKAGE
        sys.modules[_PACKAGE] = parent
    models_name = f"{_PACKAGE}.models"
    if models_name not in sys.modules:
        models_pkg = types.ModuleType(models_name)
        models_pkg.__path__ = [str(root / "models")]
        models_pkg.__package__ = models_name
        sys.modules[models_name] = models_pkg


def _install_internvideo_package(root: Path) -> None:
    """Import InternVideo models under a parent package, then alias the short names."""

    import importlib

    _register_internvideo_parent(root)
    models_name = f"{_PACKAGE}.models"
    # Skip models/__init__.py. It also imports the audiovisual tower, which needs torchaudio.
    for dotted in (
        f"{_PACKAGE}.models.internvideo2_clip",
        f"{_PACKAGE}.models.internvideo2_stage2_visual",
        f"{_PACKAGE}.models.backbones.bert.tokenization_bert",
    ):
        install_transformers_tokenizer_shims()
        importlib.import_module(dotted)

    models_pkg = sys.modules[models_name]
    clip = sys.modules[f"{_PACKAGE}.models.internvideo2_clip"]
    visual = sys.modules[f"{_PACKAGE}.models.internvideo2_stage2_visual"]
    models_pkg.InternVideo2_CLIP = clip.InternVideo2_CLIP
    models_pkg.InternVideo2_Stage2_visual = visual.InternVideo2_Stage2_visual
    _alias_internvideo_modules()


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Restore the helper removed from newer transformers."""

    import torch

    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
    """Older BERT forward calls this. Newer transformers dropped it from PreTrainedModel."""

    if head_mask is None:
        return [None] * num_hidden_layers
    converted = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
    if is_attention_chunked:
        converted = converted.unsqueeze(-1)
    return converted


def install_transformers_bert_shims() -> None:
    """Put InternVideo's vendored BERT imports back on transformers.modeling_utils."""

    import transformers.modeling_utils as modeling_utils

    try:
        from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer
    except ImportError:
        apply_chunking_to_forward = None
        prune_linear_layer = None
    if apply_chunking_to_forward is not None and not hasattr(
        modeling_utils, "apply_chunking_to_forward"
    ):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if prune_linear_layer is not None and not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
    if not hasattr(modeling_utils.PreTrainedModel, "get_head_mask"):
        modeling_utils.PreTrainedModel.get_head_mask = _get_head_mask


def install_transformers_tokenizer_shims() -> None:
    """Put InternVideo's vendored BertTokenizer helpers back on tokenization_utils.

    transformers 5 aliases ``tokenization_utils`` at a module whose ``__getattr__``
    forwards to ``tokenization_utils_sentencepiece``. Importing InternVideo's CLIP
    tower replaces that alias, so the helpers have to live on the target module.
    """

    import sys

    from transformers.tokenization_python import _is_control, _is_punctuation, _is_whitespace
    import transformers.tokenization_utils_sentencepiece as sentencepiece

    helpers = (
        ("_is_control", _is_control),
        ("_is_punctuation", _is_punctuation),
        ("_is_whitespace", _is_whitespace),
    )
    modules = [sentencepiece, sys.modules.get("transformers.tokenization_utils")]
    for module in modules:
        if module is None:
            continue
        for name, fn in helpers:
            setattr(module, name, fn)


def ensure_internvideo_importable() -> Path:
    """Put InternVideo2/multi_modality on sys.path and stub flash-attn."""

    _purge_broken_flash_attn()
    install_flash_attn_stub()
    install_transformers_bert_shims()
    install_transformers_tokenizer_shims()
    root = multi_modality_dir()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _install_internvideo_package(root)
    return root


def build_s2_config(root: Path | None = None):
    """EasyDict config for InternVideo2_Stage2_visual at 4 frames, naive attention."""

    root = multi_modality_dir() if root is None else Path(root)
    ensure_internvideo_importable()
    from utils.easydict import EasyDict

    bert_config = root / "configs" / "config_bert_large.json"
    if not bert_config.is_file():
        raise FileNotFoundError(f"Missing BERT config {bert_config}")
    return EasyDict(
        {
            "num_frames": NUM_FRAMES,
            "num_frames_test": NUM_FRAMES,
            "max_txt_l": MAX_TXT_LEN,
            "size_t": 224,
            "device": "cpu",
            "use_half_precision": True,
            "use_bf16": False,
            "gradient_checkpointing": True,
            "compile_model": False,
            "model": {
                "model_cls": "InternVideo2_Stage2_visual",
                "vision_encoder": {
                    "name": "pretrain_internvideo2_1b_patch14_224",
                    "img_size": 224,
                    "num_frames": NUM_FRAMES,
                    "tubelet_size": 1,
                    "patch_size": 14,
                    "d_model": 1408,
                    "clip_embed_dim": 768,
                    "clip_teacher_embed_dim": 3200,
                    "clip_teacher_final_dim": 768,
                    "clip_norm_type": "l2",
                    "clip_return_layer": 6,
                    "clip_student_return_interval": 1,
                    "pretrained": None,
                    "use_checkpoint": True,
                    "checkpoint_num": 40,
                    "use_flash_attn": False,
                    "use_fused_rmsnorm": False,
                    "use_fused_mlp": False,
                    "clip_teacher": None,
                    "clip_input_resolution": 224,
                    "clip_teacher_return_interval": 1,
                    "video_mask_type": "random",
                    "video_mask_ratio": 0.8,
                    "image_mask_type": "random",
                    "image_mask_ratio": 0.5,
                    "sep_image_video_pos_embed": True,
                    "keep_temporal": False,
                    "only_mask": True,
                },
                "text_encoder": {
                    "name": "bert_large",
                    "pretrained": "bert-large-uncased",
                    "config": str(bert_config),
                    "d_model": 1024,
                    "fusion_layer": 19,
                },
                "multimodal": {"enable": True},
                "embed_dim": EMBED_DIM,
                "temp": 0.07,
                "find_unused_parameters": True,
                "freeze_vision": False,
                "freeze_text": False,
            },
            "criterion": {
                "loss_weight": {"vtc": 1.0, "mlm": 0.0, "vtm": 0.0, "uta": 0.0},
                "vtm_hard_neg": True,
                "mlm_masking_prob": 0.5,
                "distill_final_features": True,
                "clip_loss_ratio": [1.0, 1.0],
                "uta_image_only": False,
            },
        }
    )


def download_s2_checkpoint(repo_id: str = S2_1B_REPO) -> Path:
    """Download the gated stage-2 1B weights. Requires HF access to the repo."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download InternVideo2-1B-s2 weights."
        ) from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=S2_1B_WEIGHT,
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {repo_id}/{S2_1B_WEIGHT}. "
            "The repo is gated: accept the license on Hugging Face and set "
            "HF_TOKEN (or run huggingface-cli login) before this cell."
        ) from exc
    return Path(path)


def unwrap_state_dict(payload: object) -> dict:
    state = payload
    if isinstance(state, dict):
        for key in ("model", "module", "state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is {type(payload).__name__}, expected a state dict.")
    cleaned: dict = {}
    for key, value in state.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    return cleaned


def load_s2_weights(model, checkpoint: Path) -> None:
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state = unwrap_state_dict(payload)
    message = model.load_state_dict(state, strict=False)
    missing = list(getattr(message, "missing_keys", []))
    unexpected = list(getattr(message, "unexpected_keys", []))
    print(
        f"Loaded {checkpoint.name}: {len(state)} tensors, "
        f"missing {len(missing)}, unexpected {len(unexpected)}",
        flush=True,
    )
    if any(key.startswith("vision_proj") for key in missing):
        raise RuntimeError(
            f"{checkpoint} did not contain vision_proj weights. "
            f"First missing keys: {missing[:8]}"
        )


class InternVideo2S2Encoder:
    """Frozen or fine-tuned InternVideo2-1B-s2 cosine encoder."""

    num_frames = NUM_FRAMES

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str | Path | None = None,
        name: str = "internvideo2_s2_1b",
    ) -> None:
        import torch

        self.name = name
        self.device = resolve_device(device)
        self.checkpoint = str(checkpoint or S2_1B_REPO)
        root = ensure_internvideo_importable()
        from models.backbones.bert.tokenization_bert import BertTokenizer
        from models.internvideo2_stage2_visual import InternVideo2_Stage2_visual

        config = build_s2_config(root)
        config.device = self.device
        self.config = config
        print(
            "Building InternVideo2-1B-s2 on CPU "
            f"(flash-attn off, {NUM_FRAMES} frames, fp16 on CUDA)",
            flush=True,
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased",
            local_files_only=False,
        )
        self.model = InternVideo2_Stage2_visual(
            config=config,
            tokenizer=self.tokenizer,
            is_pretrain=True,
        )
        weight_path = self._resolve_checkpoint(checkpoint)
        load_s2_weights(self.model, weight_path)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        place_model(self.model, self.device, dtype=dtype)
        self.model.eval()

    def _resolve_checkpoint(self, checkpoint: str | Path | None) -> Path:
        if checkpoint is None:
            return download_s2_checkpoint()
        path = Path(checkpoint)
        if path.is_file():
            return path
        text = str(checkpoint)
        if text == S2_1B_REPO or text.startswith("OpenGVLab/"):
            return download_s2_checkpoint(text if "/" in text else S2_1B_REPO)
        raise FileNotFoundError(f"InternVideo2-1B-s2 checkpoint not found: {checkpoint}")

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 1,
    ) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(videos), batch_size),
            desc=f"{self.name} videos",
            unit="batch",
        ):
            batch = videos[start : start + batch_size]
            pixel_values = torch.stack(
                [self._frames_to_model(paths) for paths in batch],
                dim=0,
            ).to(self.device)
            chunks.append(to_numpy(self._forward_videos(pixel_values)))
        return np.concatenate(chunks, axis=0)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"{self.name} texts",
            unit="batch",
        ):
            batch = texts[start : start + batch_size]
            chunks.append(to_numpy(self._forward_texts(batch)))
        return np.concatenate(chunks, axis=0)

    def _frames_to_model(self, paths: list[Path]):
        import torch
        import torch.nn.functional as F

        frames = frames_to_uint8_tchw(paths).float() / 255.0
        frames = F.interpolate(
            frames,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=frames.dtype).view(1, 3, 1, 1)
        return (frames - mean) / std

    def _forward_videos(self, pixel_values):
        import torch

        with torch.no_grad():
            _tokens, pooled = self.model.encode_vision(pixel_values, test=True)
            features = self.model.vision_proj(pooled)
        return l2_normalize_torch(_squeeze_embed(features))

    def _forward_texts(self, texts: list[str]):
        import torch

        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_TXT_LEN,
            return_tensors="pt",
        )
        encoded = encoded.to(self.device)
        with torch.no_grad():
            _tokens, pooled = self.model.encode_text(encoded)
            features = self.model.text_proj(pooled)
        return l2_normalize_torch(_squeeze_embed(features))


def _squeeze_embed(features):
    if features.ndim == 3 and features.shape[1] == 1:
        return features[:, 0]
    if features.ndim == 3:
        return features.mean(dim=1)
    return features
