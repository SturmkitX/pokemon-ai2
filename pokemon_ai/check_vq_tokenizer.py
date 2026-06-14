from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from .token_dataset import OutputImageDataset
from .utils import denormalize
from .vq_tokenizer import build_vq_tokenizer_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--output", default="runs/vq-tokenizer-check/sanity.png")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pair-name-regex", default="")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def codebook_stats(indices: torch.Tensor, codebook_size: int) -> tuple[float, float]:
    counts = torch.bincount(indices.reshape(-1).cpu(), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1.0)
    used = int((counts > 0).sum().item())
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    return used / codebook_size, entropy.exp().item()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    tokenizer = build_vq_tokenizer_from_state(state, device).eval()
    dataset = OutputImageDataset(args.rough_dir, args.final_dir, args.image_size, args.pair_name_regex)
    limit = min(args.num_images, len(dataset))
    subset = Subset(dataset, list(range(limit)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    originals: list[torch.Tensor] = []
    recons: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    recon_losses: list[float] = []

    for batch in tqdm(loader, desc="vq sanity"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            out = tokenizer(images)
        recon = out["recon"].float()
        recon_losses.append(F.l1_loss(recon, images.float()).item())
        originals.append(images.cpu())
        recons.append(recon.cpu())
        all_indices.append(out["indices"].cpu())

    original = torch.cat(originals, dim=0)[:limit]
    recon = torch.cat(recons, dim=0)[:limit]
    indices = torch.cat(all_indices, dim=0)
    used, perplexity = codebook_stats(indices, tokenizer.config.codebook_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(torch.cat([original, recon], dim=0)), output, nrow=limit)

    print(f"checkpoint: {args.tokenizer_checkpoint}")
    print(f"epoch: {state.get('epoch', 'unknown')}")
    print(f"images_checked: {limit}")
    print(f"mean_l1: {sum(recon_losses) / max(len(recon_losses), 1):.6f}")
    print(f"codebook_used_fraction: {used:.4f}")
    print(f"codebook_perplexity: {perplexity:.2f}")
    print(f"sheet: {output}")


if __name__ == "__main__":
    main()
