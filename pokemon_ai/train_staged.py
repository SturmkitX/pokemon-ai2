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
from .losses import CharbonnierLoss, PerceptualLoss
from .staged_model import StageUNet, stage_channels
from .staged_preprocess import make_stage_tensors
from .utils import denormalize, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["layout", "edge", "refine"], required=True)
    parser.add_argument("--input-dir", default="data/pairs-pokemon-sd15-v3/input")
    parser.add_argument("--target-dir", default="data/pairs-pokemon-sd15-v3/target")
    parser.add_argument("--pair-name-regex", default="_v000$")
    parser.add_argument("--cache-dir", default="cache/staged-images-512")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--blur-factor", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=20.0)
    parser.add_argument("--lambda-perceptual", type=float, default=0.0)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--sample-every-epochs", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def save_samples(path: Path, model: StageUNet, batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        source = batch["input"].to(device)
        target = batch["target"].to(device)
        stage_input, stage_target = make_stage_tensors(source, target, args.stage, args.blur_factor)
        pred = model(stage_input)
    if pred.shape[1] == 1:
        pred_vis = pred.repeat(1, 3, 1, 1)
        target_vis = stage_target.repeat(1, 3, 1, 1)
    else:
        pred_vis = pred
        target_vis = stage_target
    grid = torch.cat([source.cpu(), pred_vis.cpu(), target_vis.cpu()], dim=0)
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
    dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        args.cache_dir,
        args.image_size,
        args.pair_name_regex,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    in_channels, out_channels = stage_channels(args.stage)
    model = StageUNet(in_channels, out_channels, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    l1_loss = CharbonnierLoss()
    perceptual_loss = PerceptualLoss().to(device) if args.lambda_perceptual > 0 and out_channels == 3 else None

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
            source = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            stage_input, stage_target = make_stage_tensors(source, target, args.stage, args.blur_factor)

            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                pred = model(stage_input)
                loss_l1 = l1_loss(pred, stage_target)
                loss_perc = (
                    perceptual_loss(pred, stage_target)
                    if perceptual_loss is not None
                    else torch.zeros((), device=device)
                )
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

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch + 1,
            "args": vars(args),
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_dir / "latest.pt")
        if (epoch + 1) % args.save_every_epochs == 0:
            torch.save(state, checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")


if __name__ == "__main__":
    main()
