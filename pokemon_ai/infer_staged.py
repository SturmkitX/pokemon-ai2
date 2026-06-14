from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from .staged_model import StageUNet, stage_channels
from .staged_preprocess import source_condition
from .utils import denormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-checkpoint", required=True)
    parser.add_argument("--edge-checkpoint", required=True)
    parser.add_argument("--refine-checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--blur-factor", type=int, default=None)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def load_stage(path: str, stage: str, device: torch.device) -> tuple[StageUNet, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    train_args = state["args"]
    in_channels, out_channels = stage_channels(stage)
    model = StageUNet(in_channels, out_channels, int(train_args["base_channels"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, train_args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    layout_model, layout_args = load_stage(args.layout_checkpoint, "layout", device)
    edge_model, edge_args = load_stage(args.edge_checkpoint, "edge", device)
    refine_model, refine_args = load_stage(args.refine_checkpoint, "refine", device)

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
    image = Image.open(args.input).convert("RGB")
    source = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        cond = source_condition(source, blur_factor)
        layout = layout_model(cond)
        edge = edge_model(torch.cat([cond, layout], dim=1))
        refined = refine_model(torch.cat([cond, layout, edge], dim=1)).squeeze(0).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(denormalize(refined)).save(output_path)


if __name__ == "__main__":
    main()
