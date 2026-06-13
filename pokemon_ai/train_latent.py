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
from .latent_utils import (
    LatentPairDataset,
    add_noise,
    cache_latents,
    make_beta_schedule,
    make_training_timestep_pool,
    parse_step_choices,
    sample_latents,
)
from .utils import denormalize, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/pairs-pokemon-sd15-v3/input")
    parser.add_argument("--target-dir", default="data/pairs-pokemon-sd15-v3/target")
    parser.add_argument("--pair-name-regex", default="_v000$")
    parser.add_argument("--image-cache-dir", default="cache/pairs-pokemon-sd15-v3-256-latent-images")
    parser.add_argument("--latent-cache-dir", default="cache/pairs-pokemon-sd15-v3-256-latents")
    parser.add_argument("--run-dir", default="runs/latent-student-pokemon-sd15-v3-256")
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--vae-subfolder", default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-cache-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--train-step-choices", default="4,8,12")
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--sample-every-epochs", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def load_vae(args: argparse.Namespace, device: torch.device):
    from diffusers import AutoencoderKL

    kwargs = {"torch_dtype": torch.float16 if device.type == "cuda" else torch.float32}
    if args.vae_subfolder:
        kwargs["subfolder"] = args.vae_subfolder
    vae = AutoencoderKL.from_pretrained(args.vae_model, **kwargs).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    vae_dtype = next(vae.parameters()).dtype
    images = vae.decode((latents / scaling).to(dtype=vae_dtype)).sample
    return images.clamp(-1, 1)


def save_latent_samples(
    *,
    path: Path,
    model: ConditionalLatentUNet,
    vae,
    batch: dict[str, torch.Tensor],
    alpha_bars: torch.Tensor,
    sample_steps: int,
    seed: int,
    device: torch.device,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    source_latent = batch["source"].to(device)
    target_latent = batch["target"].to(device)
    generated_latent = sample_latents(
        model=model,
        source_latent=source_latent,
        shape=target_latent.shape,
        alpha_bars=alpha_bars,
        num_steps=sample_steps,
        seed=seed,
        device=device,
    )
    source = decode_latents(vae, source_latent).cpu()
    generated = decode_latents(vae, generated_latent).cpu()
    target = decode_latents(vae, target_latent).cpu()
    grid = torch.cat([source, generated, target], dim=0)
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
    _, _, alpha_bars = make_beta_schedule(args.num_train_timesteps, device)
    train_step_choices = parse_step_choices(args.train_step_choices)
    timestep_pool = make_training_timestep_pool(args.num_train_timesteps, train_step_choices, device)

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
        progress = tqdm(loader, desc=f"latent epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            source = batch["source"].to(device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)
            noise = torch.randn_like(target)
            pool_indices = torch.randint(0, timestep_pool.numel(), (target.shape[0],), device=device)
            timesteps = timestep_pool[pool_indices]
            noisy = add_noise(target, noise, timesteps, alpha_bars)

            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                pred = model(noisy, source, timesteps)
                loss = F.mse_loss(pred, noise)

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
            save_latent_samples(
                path=sample_dir / f"epoch-{epoch + 1:04d}.png",
                model=model,
                vae=vae,
                batch=sample_batch,
                alpha_bars=alpha_bars,
                sample_steps=args.sample_steps,
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
