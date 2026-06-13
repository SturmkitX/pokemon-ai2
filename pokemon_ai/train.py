from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig, save_config
from .dataset import PairedImageDataset
from .losses import CharbonnierLoss, PerceptualLoss
from .model import PatchDiscriminator, StylizerUNet
from .utils import save_sample_grid, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    defaults = TrainConfig()
    for field_name, field_def in defaults.__dataclass_fields__.items():
        default = getattr(defaults, field_name)
        arg_name = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(arg_name, type=type(default) if default is not None else str, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, checkpoint_config: dict[str, Any] | None = None) -> TrainConfig:
    base = TrainConfig.from_dict(checkpoint_config) if checkpoint_config is not None else TrainConfig()
    data = base.to_dict()
    for key, value in vars(args).items():
        if value is not None:
            data[key] = value
    return TrainConfig.from_dict(data)


def make_perceptual_loss(device: torch.device, enabled: bool) -> nn.Module | None:
    if not enabled:
        return None
    try:
        return PerceptualLoss().to(device)
    except Exception as exc:
        print(f"Perceptual loss disabled because VGG weights are unavailable: {exc}")
        return None


def main() -> None:
    args = parse_args()
    resume_state: dict[str, Any] | None = None
    if args.resume:
        resume_state = load_checkpoint(args.resume, map_location="cpu")

    config = config_from_args(args, resume_state.get("config") if resume_state else None)
    seed_everything(config.seed)

    run_dir = Path(config.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    save_config(config, run_dir / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedImageDataset(config.input_dir, config.target_dir, config.cache_dir, config.image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    generator = StylizerUNet(config.base_channels).to(device)
    discriminator = PatchDiscriminator(config.base_channels).to(device)
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    start_step = 0
    start_epoch = 0
    if resume_state is not None:
        generator.load_state_dict(resume_state["generator"])
        discriminator.load_state_dict(resume_state["discriminator"])
        optimizer_g.load_state_dict(resume_state["optimizer_g"])
        optimizer_d.load_state_dict(resume_state["optimizer_d"])
        if resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        start_step = int(resume_state.get("step", 0))
        start_epoch = int(resume_state.get("epoch", 0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}, step {start_step}")

    recon_loss = CharbonnierLoss()
    gan_loss = nn.MSELoss()
    perceptual_loss = make_perceptual_loss(device, config.lambda_perceptual > 0)

    step = start_step
    for epoch in range(start_epoch, config.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for batch in progress:
            source = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            use_gan = epoch >= config.gan_start_epoch and config.lambda_gan > 0

            if use_gan:
                with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                    fake = generator(source)
                    real_logits = discriminator(source, target)
                    fake_logits = discriminator(source, fake.detach())
                    loss_d = 0.5 * (
                        gan_loss(real_logits, torch.ones_like(real_logits))
                        + gan_loss(fake_logits, torch.zeros_like(fake_logits))
                    )

                optimizer_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
            else:
                loss_d = torch.zeros((), device=device)

            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                fake = generator(source)
                loss_l1 = recon_loss(fake, target)
                if use_gan:
                    fake_logits_for_g = discriminator(source, fake)
                    loss_gan = gan_loss(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
                else:
                    loss_gan = torch.zeros((), device=device)
                loss_perc = (
                    perceptual_loss(fake, target)
                    if perceptual_loss is not None
                    else torch.zeros((), device=device)
                )
                loss_g = (
                    config.lambda_l1 * loss_l1
                    + config.lambda_gan * loss_gan
                    + config.lambda_perceptual * loss_perc
                )

            optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            step += 1
            progress.set_postfix(
                {
                    "g": f"{loss_g.item():.3f}",
                    "d": f"{loss_d.item():.3f}",
                    "l1": f"{loss_l1.item():.3f}",
                }
            )

            if step % config.sample_every_steps == 0:
                generator.eval()
                with torch.no_grad():
                    sample_fake = generator(source)
                save_sample_grid(
                    sample_dir / f"step-{step:08d}.png",
                    source.detach().cpu(),
                    sample_fake.detach().cpu(),
                    target.detach().cpu(),
                    config.max_samples,
                )
                generator.train()

            if step % config.save_every_steps == 0:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scaler=scaler,
                    step=step,
                    epoch=epoch,
                    config=config.to_dict(),
                )

            if config.max_steps is not None and step >= config.max_steps:
                save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scaler=scaler,
                    step=step,
                    epoch=epoch,
                    config=config.to_dict(),
                )
                return

        save_checkpoint(
            checkpoint_dir / f"epoch-{epoch + 1:04d}.pt",
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scaler=scaler,
            step=step,
            epoch=epoch + 1,
            config=config.to_dict(),
        )
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scaler=scaler,
            step=step,
            epoch=epoch + 1,
            config=config.to_dict(),
        )


if __name__ == "__main__":
    main()
