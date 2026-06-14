from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .losses import PerceptualLoss
from .token_dataset import OutputImageDataset
from .utils import denormalize, seed_everything
from .vq_tokenizer import VQTokenizer, VQTokenizerConfig


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers: list[nn.Module] = [
            nn.Conv2d(3, channels[0], 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = channels[0]
        for out_channels in channels[1:]:
            layers += [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_gan_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def codebook_stats(indices: torch.Tensor, codebook_size: int) -> tuple[float, float]:
    counts = torch.bincount(indices.detach().reshape(-1), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1.0)
    used = (counts > 0).float().mean().item()
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    perplexity = entropy.exp().item()
    return used, perplexity


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
    parser.add_argument("--lambda-gan", type=float, default=0.0)
    parser.add_argument("--gan-start-epoch", type=int, default=10)
    parser.add_argument("--disc-base-channels", type=int, default=64)
    parser.add_argument("--skip-nonfinite", action=argparse.BooleanOptionalAction, default=True)
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
    discriminator = PatchDiscriminator(args.disc_base_channels).to(device) if args.lambda_gan > 0 else None
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9)) if discriminator is not None else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    perceptual = PerceptualLoss().to(device) if args.lambda_perceptual > 0 else None
    start_epoch = 0

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if discriminator is not None and state.get("discriminator") is not None:
            discriminator.load_state_dict(state["discriminator"])
        if optimizer_d is not None and state.get("optimizer_d") is not None:
            optimizer_d.load_state_dict(state["optimizer_d"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        if state.get("scaler_d") is not None:
            scaler_d.load_state_dict(state["scaler_d"])
        start_epoch = int(state.get("epoch", 0))

    if not metrics_path.exists() or start_epoch == 0:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["epoch", "loss", "recon", "perceptual", "vq", "gan_g", "gan_d", "codebook_used", "perplexity"])

    for epoch in range(start_epoch, args.epochs):
        sums = {"loss": 0.0, "recon": 0.0, "perceptual": 0.0, "vq": 0.0, "gan_g": 0.0, "gan_d": 0.0, "used": 0.0, "perplexity": 0.0}
        batches = 0
        progress = tqdm(loader, desc=f"vq epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(images)

            recon = out["recon"].float()
            images_f = images.float()
            gan_g_loss = torch.zeros((), device=device)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                recon_loss = F.l1_loss(recon, images_f)
                perceptual_loss = perceptual(recon, images_f) if perceptual is not None else torch.zeros((), device=device)
                vq_loss = out["vq_loss"].float()
                if discriminator is not None and epoch + 1 >= args.gan_start_epoch:
                    gan_g_loss = generator_gan_loss(discriminator(recon))
                loss = args.lambda_recon * recon_loss + args.lambda_perceptual * perceptual_loss + args.lambda_vq * vq_loss + args.lambda_gan * gan_g_loss

            if not torch.isfinite(loss):
                message = (
                    f"non-finite VQ loss skipped: loss={loss.item()} "
                    f"recon={recon_loss.item()} perceptual={perceptual_loss.item()} vq={vq_loss.item()}"
                )
                if args.skip_nonfinite:
                    progress.write(message)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise FloatingPointError(message)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            gan_d_loss = torch.zeros((), device=device)
            if discriminator is not None and optimizer_d is not None and epoch + 1 >= args.gan_start_epoch:
                with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    real_logits = discriminator(images)
                    fake_logits = discriminator(recon.detach())
                    gan_d_loss = discriminator_loss(real_logits.float(), fake_logits.float())
                if torch.isfinite(gan_d_loss):
                    optimizer_d.zero_grad(set_to_none=True)
                    scaler_d.scale(gan_d_loss).backward()
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()

            used, perplexity = codebook_stats(out["indices"], args.codebook_size)
            batches += 1
            sums["loss"] += loss.item()
            sums["recon"] += recon_loss.item()
            sums["perceptual"] += perceptual_loss.item()
            sums["vq"] += vq_loss.item()
            sums["gan_g"] += gan_g_loss.item()
            sums["gan_d"] += gan_d_loss.item()
            sums["used"] += used
            sums["perplexity"] += perplexity
            progress.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "recon": f"{recon_loss.item():.3f}",
                    "vq": f"{vq_loss.item():.3f}",
                    "used": f"{used:.2f}",
                }
            )

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch + 1]
                + [
                    f"{sums[key] / max(batches, 1):.6f}"
                    for key in ("loss", "recon", "perceptual", "vq", "gan_g", "gan_d", "used", "perplexity")
                ]
            )

        if batches == 0:
            raise FloatingPointError(
                "Every VQ batch produced a non-finite loss. Restart from a clean run with --no-amp and a lower --lr."
            )

        if (epoch + 1) % args.sample_every_epochs == 0:
            save_samples(sample_dir / f"epoch-{epoch + 1:04d}.png", model, next(iter(sample_loader)), device, args.amp)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "discriminator": discriminator.state_dict() if discriminator is not None else None,
            "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
            "scaler": scaler.state_dict(),
            "scaler_d": scaler_d.state_dict(),
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
