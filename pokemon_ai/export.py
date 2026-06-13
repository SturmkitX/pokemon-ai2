from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import TrainConfig
from .model import StylizerUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = TrainConfig.from_dict(checkpoint.get("config", {}))
    image_size = args.image_size or config.image_size
    device = torch.device(args.device)

    model = StylizerUNet(config.base_channels).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()

    example = torch.randn(1, 3, image_size, image_size, device=device)
    traced = torch.jit.trace(model, example)
    traced = torch.jit.freeze(traced)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))


if __name__ == "__main__":
    main()
