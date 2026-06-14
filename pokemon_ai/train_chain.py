from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .chain_dataset import ChainDataset
from .losses import CharbonnierLoss, PerceptualLoss
from .staged_model import StageUNet
from .staged_preprocess import source_condition
from .utils import denormalize, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["rough", "refine"], required=True)
    parser.add_argument("--source-dir", default="data/pairs-pokemon-chain-v1/source")
    parser.add_argument("--rough-dir", default="data/pairs-pokemon-chain-v1/rough")
    parser.add_argument("--final-dir", default="data/pairs-pokemon-chain-v1/final")
    parser.add_argument("--pair-name-regex", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--blur-factor", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=10.0)
    parser.add_argument("--lambda-perceptual", type=float, default=4.0)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--sample-every-epochs", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def make_batch(batch, stage: str, blur_factor: int, device: torch.device):
    source = batch["source"].to(device, non_blocking=True)
    rough = batch["rough"].to(device, non_blocking=True)
    final = batch["final"].to(device, non_blocking=True)
    cond = source_condition(source, blur_factor)
    if stage == "rough":
        return cond, rough, source
    return torch.cat([cond, rough], dim=1), final, source


def save_samples(path: Path, model: StageUNet, batch, args: argparse.Namespace, device: torch.device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        stage_input, target, source = make_batch(batch, args.stage, args.blur_factor, device)
        pred = model(stage_input)
    grid = torch.cat([source.cpu(), pred.cpu(), target.cpu()], dim=0)
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
    dataset = ChainDataset(args.source_dir, args.rough_dir, args.final_dir, args.image_size, args.pair_name_regex)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    in_channels = 7 if args.stage == "rough" else 10
    model = StageUNet(in_channels, 3, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    l1 = CharbonnierLoss()
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
            csv.writer(handle).writerow(["epoch", "loss", "l1", "perceptual"])

    for epoch in range(start_epoch, args.epochs):
        sums = {"loss": 0.0, "l1": 0.0, "perceptual": 0.0}
        batches = 0
        progress = tqdm(loader, desc=f"{args.stage} epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            stage_input, target, _ = make_batch(batch, args.stage, args.blur_factor, device)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                pred = model(stage_input)
                loss_l1 = l1(pred, target)
                loss_perc = perceptual(pred, target) if perceptual is not None else torch.zeros((), device=device)
                loss = args.lambda_l1 * loss_l1 + args.lambda_perceptual * loss_perc

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            sums["loss"] += loss.item()
            sums["l1"] += loss_l1.item()
            sums["perceptual"] += loss_perc.item()
            progress.set_postfix({"loss": f"{loss.item():.3f}", "l1": f"{loss_l1.item():.3f}"})

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    f"{sums['loss'] / max(batches, 1):.6f}",
                    f"{sums['l1'] / max(batches, 1):.6f}",
                    f"{sums['perceptual'] / max(batches, 1):.6f}",
                ]
            )
        if (epoch + 1) % args.sample_every_epochs == 0:
            save_samples(sample_dir / f"epoch-{epoch + 1:04d}.png", model, next(iter(loader)), args, device)

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1, "args": vars(args)}
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_dir / "latest.pt")
        if (epoch + 1) % args.save_every_epochs == 0:
            torch.save(state, checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")


if __name__ == "__main__":
    main()
