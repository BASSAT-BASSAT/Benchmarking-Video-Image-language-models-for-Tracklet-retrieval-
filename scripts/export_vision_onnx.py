#!/usr/bin/env python3
"""Export a registry vision tower to ONNX. ``--trt`` only prints the trtexec command."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from shawaf_vlm.models.registry import build_encoder
from shawaf_vlm.optim.export import export_module_onnx, trtexec_command


class _VisionWrapper(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.model = encoder.model

    def forward(self, images):
        return self.model.encode_image(images)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a vision tower to ONNX.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Print a trtexec FP16 command. Does not build an engine.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    encoder = build_encoder(args.model, device=args.device)
    if not hasattr(encoder, "model"):
        raise TypeError(f"{args.model} has no .model vision tower to export.")
    export_module_onnx(_VisionWrapper(encoder), args.out, args.image_size)
    print(f"Wrote {args.out}")
    if args.trt:
        print(trtexec_command(args.out, args.image_size))


if __name__ == "__main__":
    main()
