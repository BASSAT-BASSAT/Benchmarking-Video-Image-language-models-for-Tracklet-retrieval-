"""Official Qwen3-VL-Embedding-2B video/text adapter.

The model and preprocessing constants mirror QwenLM/Qwen3-VL-Embedding.  A
tracklet is passed as one ordered video frame sequence; it is never reduced to
independent image embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from shawaf_vlm.models.runtime import place_model, resolve_device, to_numpy

QWEN3_VL_EMBED_2B = "Qwen/Qwen3-VL-Embedding-2B"
QWEN3_MAX_LENGTH = 8192
QWEN3_MAX_FRAMES = 64
QWEN3_IMAGE_FACTOR = 32
QWEN3_MIN_PIXELS = 4 * QWEN3_IMAGE_FACTOR * QWEN3_IMAGE_FACTOR
QWEN3_MAX_PIXELS = 1800 * QWEN3_IMAGE_FACTOR * QWEN3_IMAGE_FACTOR
QWEN3_FRAME_MAX_PIXELS = 768 * QWEN3_IMAGE_FACTOR * QWEN3_IMAGE_FACTOR
QWEN3_TOTAL_PIXELS = 10 * QWEN3_FRAME_MAX_PIXELS
NEUTRAL_INSTRUCTION = "Represent the user's input."


def _embedding_model_class():
    """Create the official embedding backbone without importing it at package load."""

    from transformers.modeling_outputs import ModelOutput
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )

    @dataclass
    class Qwen3VLForEmbeddingOutput(ModelOutput):
        last_hidden_state: Any = None
        attention_mask: Any = None

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        accepts_loss_kwargs = False

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            cache_position=None,
            **kwargs,
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )
            return Qwen3VLForEmbeddingOutput(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )

    return Qwen3VLForEmbedding


class Qwen3VLEmbeddingEncoder:
    """Qwen's native video embedding model under the Stage 1 encoder contract."""

    name = "qwen3_vl_embed_2b"
    checkpoint = QWEN3_VL_EMBED_2B

    def __init__(
        self,
        device: str = "cuda",
        max_length: int = QWEN3_MAX_LENGTH,
        max_frames: int = QWEN3_MAX_FRAMES,
        instruction: str = NEUTRAL_INSTRUCTION,
    ) -> None:
        import torch
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        if max_length < QWEN3_MAX_LENGTH:
            raise ValueError(
                f"Qwen3-VL-Embedding requires max_length >= {QWEN3_MAX_LENGTH}; "
                "short contexts can truncate video tokens."
            )
        self.device = resolve_device(device)
        self.max_length = int(max_length)
        self.max_frames = int(max_frames)
        self.instruction = instruction
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model_class = _embedding_model_class()
        self.model = model_class.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        place_model(self.model, self.device, dtype=dtype)
        self.processor = Qwen3VLProcessor.from_pretrained(
            self.checkpoint,
            padding_side="right",
        )

    def encode_videos(
        self,
        videos: list[list[Path]],
        batch_size: int = 1,
    ) -> np.ndarray:
        inputs = [{"video": [Path(path) for path in frames]} for frames in videos]
        return self._encode(inputs, batch_size=max(int(batch_size), 1), label="videos")

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        inputs = [{"text": text} for text in texts]
        return self._encode(inputs, batch_size=max(int(batch_size), 1), label="texts")

    def _encode(
        self,
        items: list[dict[str, Any]],
        batch_size: int,
        label: str,
    ) -> np.ndarray:
        import torch

        chunks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(items), batch_size),
            desc=f"{self.name} {label}",
            unit="batch",
        ):
            conversations = [self._format_input(item) for item in items[start : start + batch_size]]
            model_inputs = self._preprocess(conversations)
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                embeddings = self._pool_last(
                    outputs.last_hidden_state,
                    model_inputs["attention_mask"],
                )
                embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            chunks.append(to_numpy(embeddings))
            del model_inputs, outputs, embeddings
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return np.concatenate(chunks, axis=0)

    def _format_input(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        video = item.get("video")
        if video:
            ordered = [Path(path).resolve().as_uri() for path in video]
            if len(ordered) > self.max_frames:
                indices = np.linspace(0, len(ordered) - 1, self.max_frames, dtype=int)
                ordered = [ordered[int(index)] for index in indices]
            content.append(
                {
                    "type": "video",
                    "video": ordered,
                    "total_pixels": QWEN3_TOTAL_PIXELS,
                }
            )
        text = item.get("text")
        if text is not None:
            content.append({"type": "text", "text": str(text)})
        if not content:
            content.append({"type": "text", "text": "NULL"})
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.instruction}],
            },
            {"role": "user", "content": content},
        ]

    def _preprocess(self, conversations: list[list[dict[str, Any]]]) -> dict[str, Any]:
        from qwen_vl_utils.vision_process import process_vision_info

        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            # Never truncate visual tokens: doing so caused the documented
            # token/feature mismatch in the earlier max_length=512 adapter.
            truncation=False,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        sequence_length = int(inputs["input_ids"].shape[1])
        if sequence_length > self.max_length:
            raise ValueError(
                f"Qwen3-VL input uses {sequence_length} tokens, exceeding the "
                f"official {self.max_length}-token embedding budget. Refusing "
                "to truncate video tokens."
            )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _pool_last(hidden_state, attention_mask):
        import torch

        last_one = attention_mask.flip(dims=[1]).argmax(dim=1)
        columns = attention_mask.shape[1] - last_one - 1
        rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[rows, columns]
