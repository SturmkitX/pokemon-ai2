from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .dataset import PairedImageDataset
from .latent_model import ConditionalLatentUNet
from .latent_utils import LatentPairDataset, cache_latents
from .train_latent import decode_latents, load_vae
from .utils import denormalize, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs-pokemon-sd15-v3/input")
    parser.add_argument("--target-dir", default="data/pairs-pokemon-sd15-v3/target")
    parser.add_argument("--pair-name-regex", default="_v000$")
    parser.add_argument("--image-cache-dir", default="cache/pairs-pokemon-sd15-v3-512-flow-images")
    parser.add_argument("--latent-cache-dir", default="cache/pairs-pokemon-sd15-v3-512-flow-latents")
    parser.add_argument("--run-dir", default="runs/flow-student-pokemon-sd15-v3-512")
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--vae-subfolder", default="")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-cache-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--noise-strength", type=float, default=0.65)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--condition-drop-prob", type=float, default=0.0)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--sample-every-epochs", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


@torch.no_grad()
def sample_flow(
    *,
    model: ConditionalLatentUNet,
    source: torch.Tensor,
    steps: int,
    noise_strength: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(source.shape, generator=generator, device=device)
    latent = source + noise_strength * noise
    dt = 1.0 / steps
    for index in range(steps):
        t_float = torch.full((source.shape[0],), index / steps, device=device)
        t_embed = (t_float * 999).long()
        velocity = model(latent, source, t_embed)
        latent = latent + dt * velocity
    return latent


@torch.no_grad()
def save_flow_samples(
    *,
    path: Path,
    model: ConditionalLatentUNet,
    vae,
    batch: dict[str, torch.Tensor],
    steps: int,
    noise_strength: float,
    seed: int,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    source = batch["source"].to(device, dtype=torch.float32)
    target = batch["target"].to(device, dtype=torch.float32)
    generated = sample_flow(
        model=model,
        source=source,
        steps=steps,
        noise_strength=noise_strength,
        seed=seed,
        device=device,
    )
    source_img = decode_latents(vae, source).cpu()
    generated_img = decode_latents(vae, generated).cpu()
    target_img = decode_latents(vae, target).cpu()
    grid = torch.cat([source_img, generated_img, target_img], dim=0)
    save_image(denormalize(grid), path, nrow=source.shape[0])
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
    vae = load_vae(args, device)
    image_dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        args.image_cache_dir,
        args.image_size,
        args.pair_name_regex,
    )
    names = cache_latents(
        vae=vae,
        dataset=image_dataset,
        latent_dir=args.latent_cache_dir,
        batch_size=args.latent_cache_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    latent_dataset = LatentPairDataset(names, args.latent_cache_dir)
    loader = DataLoader(
        latent_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = ConditionalLatentUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        step = int(state.get("step", 0))

    if not metrics_path.exists() or start_epoch == 0:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["epoch", "step", "loss"])

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        batches = 0
        progress = tqdm(loader, desc=f"flow epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            source = batch["source"].to(device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)
            noise = torch.randn_like(source)
            start = source + args.noise_strength * noise
            time = torch.rand(source.shape[0], device=device)
            x_t = (1.0 - time[:, None, None, None]) * start + time[:, None, None, None] * target
            velocity = target - start
            cond = source
            if args.condition_drop_prob > 0:
                keep = (torch.rand(source.shape[0], device=device) >= args.condition_drop_prob).float()
                cond = cond * keep[:, None, None, None]
            time_embed = (time * 999).long()

            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                pred = model(x_t, cond, time_embed)
                loss = F.mse_loss(pred, velocity)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            batches += 1
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([epoch + 1, step, f"{total_loss / max(batches, 1):.6f}"])

        if (epoch + 1) % args.sample_every_epochs == 0:
            sample_batch = next(iter(loader))
            save_flow_samples(
                path=sample_dir / f"epoch-{epoch + 1:04d}.png",
                model=model,
                vae=vae,
                batch=sample_batch,
                steps=args.sample_steps,
                noise_strength=args.noise_strength,
                seed=args.seed + epoch,
                device=device,
            )

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch + 1,
            "step": step,
            "args": vars(args),
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_dir / "latest.pt")
        if (epoch + 1) % args.save_every_epochs == 0:
            torch.save(state, checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")


if __name__ == "__main__":
    main()
