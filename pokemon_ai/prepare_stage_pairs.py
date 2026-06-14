from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from .dataset import PairedImageDataset
from .staged_preprocess import blur_color, make_stage_tensors, sobel_edges
from .utils import denormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="cache/prepare-stage-pairs")
    parser.add_argument("--pair-name-regex", default="")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--blur-factor", type=int, default=16)
    return parser.parse_args()


def save_tensor_image(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    to_pil_image(denormalize(tensor.cpu())).save(path)


def save_contact_sheet(path: Path, images: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_images = []
    for tensor in images:
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        pil_images.append(to_pil_image(denormalize(tensor.cpu())))
    width = sum(image.width for image in pil_images)
    height = max(image.height for image in pil_images)
    sheet = Image.new("RGB", (width, height), (20, 20, 20))
    x = 0
    for image in pil_images:
        sheet.paste(image, (x, 0))
        x += image.width
    sheet.save(path)


def main() -> None:
    args = parse_args()
    dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        args.cache_dir,
        args.image_size,
        args.pair_name_regex,
    )
    output_dir = Path(args.output_dir)

    for item in tqdm(dataset, desc="prepare stage pairs"):
        name = str(item["name"])
        source = item["input"]
        target = item["target"]
        assert isinstance(source, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        layout_input, layout_target = make_stage_tensors(source, target, "layout", args.blur_factor)
        edge_input, edge_target = make_stage_tensors(source, target, "edge", args.blur_factor)
        refine_input, refine_target = make_stage_tensors(source, target, "refine", args.blur_factor)

        source_blur = blur_color(source, args.blur_factor)
        source_edge = sobel_edges(source)

        save_tensor_image(output_dir / "layout" / "input_rgb" / f"{name}.png", source)
        save_tensor_image(output_dir / "layout" / "input_color" / f"{name}.png", source_blur)
        save_tensor_image(output_dir / "layout" / "input_edge" / f"{name}.png", source_edge)
        save_tensor_image(output_dir / "layout" / "target" / f"{name}.png", layout_target)

        save_tensor_image(output_dir / "edge" / "input_layout" / f"{name}.png", layout_target)
        save_tensor_image(output_dir / "edge" / "target" / f"{name}.png", edge_target)

        save_tensor_image(output_dir / "refine" / "input_layout" / f"{name}.png", layout_target)
        save_tensor_image(output_dir / "refine" / "input_edge" / f"{name}.png", edge_target)
        save_tensor_image(output_dir / "refine" / "target" / f"{name}.png", refine_target)

        save_contact_sheet(
            output_dir / "chains" / f"{name}.png",
            [
                source,
                source_blur,
                source_edge,
                layout_target,
                edge_target,
                refine_target,
            ],
        )

        # Keep machine-readable tensors too. They preserve the exact multi-channel inputs
        # used by train_staged.py.
        tensor_path = output_dir / "tensors" / f"{name}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layout_input": layout_input,
                "layout_target": layout_target,
                "edge_input": edge_input,
                "edge_target": edge_target,
                "refine_input": refine_input,
                "refine_target": refine_target,
            },
            tensor_path,
        )


if __name__ == "__main__":
    main()
