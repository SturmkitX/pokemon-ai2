from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .losses import PerceptualLoss
from .token_dataset import OutputImageDataset
from .utils import denormalize, seed_everything
from .vq_tokenizer import VQTokenizer, VQTokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--pair-name-regex", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--downsample-factor", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-perceptual", type=float, default=0.2)
    parser.add_argument("--lambda-vq", type=float, default=1.0)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--sample-every-epochs", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


@torch.no_grad()
def save_samples(path: Path, model: VQTokenizer, batch: dict, device: torch.device, amp: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    images = batch["image"].to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        recon = model(images)["recon"]
    n = min(images.shape[0], 8)
    grid = torch.cat([images[:n].cpu(), recon[:n].cpu()], dim=0)
    save_image(denormalize(grid), path, nrow=n)
    model.train()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    metrics_path = run_dir / "metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OutputImageDataset(args.rough_dir, args.final_dir, args.image_size, args.pair_name_regex)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    sample_loader = DataLoader(dataset, batch_size=min(args.batch_size, 8), shuffle=False, num_workers=0)

    config = VQTokenizerConfig(
        codebook_size=args.codebook_size,
        embedding_dim=args.embedding_dim,
        downsample_factor=args.downsample_factor,
        base_channels=args.base_channels,
        commitment_cost=args.commitment_cost,
    )
    model = VQTokenizer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    perceptual = PerceptualLoss().to(device) if args.lambda_perceptual > 0 else None
    start_epoch = 0

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))

    if not metrics_path.exists() or start_epoch == 0:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["epoch", "loss", "recon", "perceptual", "vq"])

    for epoch in range(start_epoch, args.epochs):
        sums = {"loss": 0.0, "recon": 0.0, "perceptual": 0.0, "vq": 0.0}
        batches = 0
        progress = tqdm(loader, desc=f"vq epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(images)
                recon_loss = F.l1_loss(out["recon"], images)
                perceptual_loss = perceptual(out["recon"], images) if perceptual is not None else torch.zeros((), device=device)
                vq_loss = out["vq_loss"]
                loss = args.lambda_recon * recon_loss + args.lambda_perceptual * perceptual_loss + args.lambda_vq * vq_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batches += 1
            sums["loss"] += loss.item()
            sums["recon"] += recon_loss.item()
            sums["perceptual"] += perceptual_loss.item()
            sums["vq"] += vq_loss.item()
            progress.set_postfix({"loss": f"{loss.item():.3f}", "recon": f"{recon_loss.item():.3f}", "vq": f"{vq_loss.item():.3f}"})

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([epoch + 1] + [f"{sums[key] / max(batches, 1):.6f}" for key in ("loss", "recon", "perceptual", "vq")])

        if (epoch + 1) % args.sample_every_epochs == 0:
            save_samples(sample_dir / f"epoch-{epoch + 1:04d}.png", model, next(iter(sample_loader)), device, args.amp)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "config": vars(config),
            "args": vars(args),
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_dir / "latest.pt")
        if (epoch + 1) % args.save_every_epochs == 0:
            torch.save(state, checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")


if __name__ == "__main__":
    main()
