from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from .checkpoint import load_checkpoint
from .config import TrainConfig
from .model import ResNetStylizer, StylizerUNet
from .utils import denormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = TrainConfig.from_dict(checkpoint.get("config", {}))
    image_size = args.image_size or config.image_size
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if config.generator_arch == "unet":
        model = StylizerUNet(config.base_channels).to(device)
    elif config.generator_arch == "resnet":
        model = ResNetStylizer(config.base_channels, config.res_blocks).to(device)
    else:
        raise ValueError(f"Unsupported generator architecture: {config.generator_arch}")
    model.load_state_dict(checkpoint["generator"])
    model.eval()

    image = Image.open(args.input).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor).squeeze(0).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(denormalize(output)).save(output_path)


if __name__ == "__main__":
    main()
