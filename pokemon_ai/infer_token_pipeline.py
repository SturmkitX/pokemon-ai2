from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from .token_model import build_token_predictor_from_state
from .token_preprocess import token_source_condition
from .utils import denormalize, seed_everything
from .vq_tokenizer import build_vq_tokenizer_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--rough-predictor-checkpoint", required=True)
    parser.add_argument("--refine-predictor-checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--rough-steps", type=int, default=4)
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--blur-factor", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_image(path: str | Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(image).unsqueeze(0)


@torch.no_grad()
def generate_tokens(model, condition: torch.Tensor, steps: int, rough_tokens: torch.Tensor | None = None) -> torch.Tensor:
    b = condition.shape[0]
    grid = model.config.grid_size
    tokens = torch.full((b, grid, grid), model.mask_token_id, device=condition.device, dtype=torch.long)
    unknown = torch.ones_like(tokens, dtype=torch.bool)
    total = grid * grid
    for step in range(max(steps, 1)):
        logits = model(tokens, condition, rough_tokens)
        probs = logits.softmax(dim=1)
        confidence, pred = probs.max(dim=1)
        tokens = torch.where(unknown, pred, tokens)
        next_unknown_count = int(round(total * torch.cos(torch.tensor((step + 1) / max(steps, 1) * torch.pi * 0.5)).item()))
        if step == steps - 1:
            next_unknown_count = 0
        if next_unknown_count > 0:
            new_unknown = torch.zeros_like(unknown)
            conf_flat = confidence.view(b, -1)
            unknown_flat = unknown.view(b, -1)
            for item in range(b):
                candidates = unknown_flat[item].nonzero(as_tuple=False).flatten()
                if candidates.numel() <= next_unknown_count:
                    keep = candidates
                else:
                    order = conf_flat[item, candidates].argsort()
                    keep = candidates[order[:next_unknown_count]]
                new_unknown.view(b, -1)[item, keep] = True
            tokens[new_unknown] = model.mask_token_id
            unknown = new_unknown
        else:
            unknown.zero_()
    return tokens.clamp_max(model.config.codebook_size - 1)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_vq_tokenizer_from_state(torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False), device).eval()
    rough_model = build_token_predictor_from_state(torch.load(args.rough_predictor_checkpoint, map_location=device, weights_only=False), device).eval()
    refine_model = build_token_predictor_from_state(torch.load(args.refine_predictor_checkpoint, map_location=device, weights_only=False), device).eval()

    source = load_image(args.input, args.image_size).to(device)
    condition = token_source_condition(source, args.blur_factor)
    with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
        rough_tokens = generate_tokens(rough_model, condition, args.rough_steps)
        final_tokens = generate_tokens(refine_model, condition, args.refine_steps, rough_tokens)
        output = tokenizer.decode_indices(final_tokens)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(output.cpu()), output_path)


if __name__ == "__main__":
    main()
