from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from .staged_model import StageUNet
from .staged_preprocess import source_condition
from .utils import denormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rough-checkpoint", required=True)
    parser.add_argument("--refine-checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--blur-factor", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def load_model(path: str, in_channels: int, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    train_args = state["args"]
    model = StageUNet(in_channels, 3, int(train_args["base_channels"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, train_args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rough_model, rough_args = load_model(args.rough_checkpoint, 7, device)
    refine_model, refine_args = load_model(args.refine_checkpoint, 10, device)
    image_size = args.image_size or int(refine_args["image_size"])
    blur_factor = args.blur_factor or int(refine_args["blur_factor"])

    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    source = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        cond = source_condition(source, blur_factor)
        rough = rough_model(cond)
        final = refine_model(torch.cat([cond, rough], dim=1)).squeeze(0).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(denormalize(final)).save(output_path)


if __name__ == "__main__":
    main()
