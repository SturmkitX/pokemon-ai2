from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataset import PairedImageDataset


class LatentPairDataset(Dataset):
    def __init__(self, names: list[str], latent_dir: str | Path) -> None:
        self.names = names
        self.latent_dir = Path(latent_dir)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        return {
            "source": torch.load(self.latent_dir / "source" / f"{name}.pt", map_location="cpu", weights_only=True),
            "target": torch.load(self.latent_dir / "target" / f"{name}.pt", map_location="cpu", weights_only=True),
            "name": name,
        }


@torch.no_grad()
def cache_latents(
    *,
    vae: torch.nn.Module,
    dataset: PairedImageDataset,
    latent_dir: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[str]:
    latent_dir = Path(latent_dir)
    source_dir = latent_dir / "source"
    target_dir = latent_dir / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    names = [input_path.stem for input_path, _ in dataset.pairs]
    missing = [
        name
        for name in names
        if not (source_dir / f"{name}.pt").exists() or not (target_dir / f"{name}.pt").exists()
    ]
    if not missing:
        return names

    missing_set = set(missing)
    indices = [i for i, name in enumerate(names) if name in missing_set]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    vae_dtype = next(vae.parameters()).dtype
    vae.eval()
    for batch in tqdm(loader, desc="cache latents"):
        source = batch["input"].to(device, dtype=vae_dtype, non_blocking=True)
        target = batch["target"].to(device, dtype=vae_dtype, non_blocking=True)
        source_latent = vae.encode(source).latent_dist.mean * scaling
        target_latent = vae.encode(target).latent_dist.mean * scaling
        for item_index, name in enumerate(batch["name"]):
            torch.save(source_latent[item_index].detach().float().cpu(), source_dir / f"{name}.pt")
            torch.save(target_latent[item_index].detach().float().cpu(), target_dir / f"{name}.pt")
    return names


def make_beta_schedule(num_train_timesteps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(0.00085, 0.012, num_train_timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def add_noise(clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor) -> torch.Tensor:
    a = alpha_bars[timesteps].view(-1, 1, 1, 1)
    return a.sqrt() * clean + (1.0 - a).sqrt() * noise


def parse_step_choices(value: str) -> list[int]:
    choices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not choices:
        raise ValueError("Expected at least one step choice")
    return choices


def make_inference_timesteps(
    num_train_timesteps: int,
    num_steps: int,
    device: torch.device,
    noise_strength: float = 1.0,
) -> torch.Tensor:
    start = max(1, min(num_train_timesteps - 1, int((num_train_timesteps - 1) * noise_strength)))
    return torch.linspace(start, 0, num_steps, device=device).long()


def make_training_timestep_pool(
    num_train_timesteps: int,
    step_choices: list[int],
    device: torch.device,
    noise_strength: float = 1.0,
) -> torch.Tensor:
    pools = [
        make_inference_timesteps(num_train_timesteps, steps, device, noise_strength)
        for steps in step_choices
    ]
    return torch.unique(torch.cat(pools)).long()


@torch.no_grad()
def sample_latents(
    *,
    model: torch.nn.Module,
    source_latent: torch.Tensor,
    shape: tuple[int, int, int, int],
    alpha_bars: torch.Tensor,
    num_steps: int,
    seed: int,
    guidance_scale: float,
    noise_strength: float,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=device)
    timesteps = make_inference_timesteps(alpha_bars.numel(), num_steps, device, noise_strength)
    start_a = alpha_bars[timesteps[0]].view(1, 1, 1, 1)
    latent = start_a.sqrt() * source_latent + (1.0 - start_a).sqrt() * noise
    for index, timestep in enumerate(timesteps):
        t = torch.full((shape[0],), int(timestep.item()), device=device, dtype=torch.long)
        if guidance_scale == 1.0:
            pred_noise = model(latent, source_latent, t)
        else:
            pred_uncond = model(latent, torch.zeros_like(source_latent), t)
            pred_cond = model(latent, source_latent, t)
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        a = alpha_bars[timestep].view(1, 1, 1, 1)
        x0 = (latent - (1.0 - a).sqrt() * pred_noise) / a.sqrt()
        if index == len(timesteps) - 1:
            latent = x0
        else:
            prev_t = timesteps[index + 1]
            prev_a = alpha_bars[prev_t].view(1, 1, 1, 1)
            latent = prev_a.sqrt() * x0 + (1.0 - prev_a).sqrt() * pred_noise
    return latent
