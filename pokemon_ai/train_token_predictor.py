from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .chain_dataset import ChainDataset
from .token_dataset import ChainTokenDataset
from .token_model import MaskedTokenPredictor, TokenPredictorConfig, build_token_predictor_from_state
from .token_preprocess import token_condition_channels, token_source_condition
from .utils import denormalize, seed_everything
from .vq_tokenizer import build_vq_tokenizer_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["rough", "refine"], required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--pair-name-regex", default="")
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--rough-predictor-checkpoint", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--token-cache-dir", default="")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--architecture", choices=["resnet", "unet"], default="resnet")
    parser.add_argument("--model-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--blur-factor", type=int, default=16)
    parser.add_argument("--condition-mode", choices=["safe", "rgb"], default="safe")
    parser.add_argument("--mask-schedule", choices=["cosine", "uniform"], default="cosine")
    parser.add_argument("--min-mask-ratio", type=float, default=0.05)
    parser.add_argument("--full-mask-prob", type=float, default=0.15)
    parser.add_argument("--foreground-loss-weight", type=float, default=4.0)
    parser.add_argument("--edge-loss-weight", type=float, default=2.0)
    parser.add_argument("--refine-conditioning", choices=["teacher-rough", "source-vq"], default="teacher-rough")
    parser.add_argument("--teacher-forcing-rough-prob", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--sample-every-epochs", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-k", type=int, default=64)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def mask_ratio(batch_size: int, device: torch.device, schedule: str, min_mask_ratio: float) -> torch.Tensor:
    r = torch.rand(batch_size, device=device)
    if schedule == "cosine":
        return torch.cos(r * torch.pi * 0.5).clamp(min_mask_ratio, 1.0)
    return r.clamp(min_mask_ratio, 1.0)


def make_masked_tokens(
    target: torch.Tensor,
    mask_token_id: int,
    schedule: str,
    min_mask_ratio: float,
    full_mask_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, w = target.shape
    ratios = mask_ratio(b, target.device, schedule, min_mask_ratio)
    if full_mask_prob > 0:
        full_mask_items = torch.rand(b, device=target.device) < full_mask_prob
        ratios = torch.where(full_mask_items, torch.ones_like(ratios), ratios)
    ratios = ratios.view(b, 1, 1)
    mask = torch.rand(b, h, w, device=target.device) < ratios
    flat_mask = mask.view(b, -1)
    empty = flat_mask.sum(dim=1) == 0
    if empty.any():
        flat_mask[empty, torch.randint(0, h * w, (int(empty.sum().item()),), device=target.device)] = True
    mask = flat_mask.view(b, h, w)
    masked = target.clone()
    masked[mask] = mask_token_id
    return masked, mask


@torch.no_grad()
def generate_tokens(
    model: MaskedTokenPredictor,
    condition: torch.Tensor,
    steps: int,
    rough_tokens: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int = 64,
) -> torch.Tensor:
    b = condition.shape[0]
    grid = model.config.grid_size
    tokens = torch.full((b, grid, grid), model.mask_token_id, device=condition.device, dtype=torch.long)
    unknown = torch.ones_like(tokens, dtype=torch.bool)
    total = grid * grid
    for step in range(max(steps, 1)):
        logits = model(tokens, condition, rough_tokens)
        if top_k > 0 and top_k < logits.shape[1]:
            top_values, top_indices = logits.topk(top_k, dim=1)
            probs_top = (top_values / max(temperature, 1e-6)).softmax(dim=1)
            sampled = torch.multinomial(probs_top.permute(0, 2, 3, 1).reshape(-1, top_k), 1).view(b, grid, grid)
            pred = top_indices.permute(0, 2, 3, 1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            confidence = probs_top.permute(0, 2, 3, 1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            probs = (logits / max(temperature, 1e-6)).softmax(dim=1)
            pred = torch.multinomial(probs.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]), 1).view(b, grid, grid)
            confidence = probs.permute(0, 2, 3, 1).gather(-1, pred.unsqueeze(-1)).squeeze(-1)
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


@torch.no_grad()
def build_token_cache(args: argparse.Namespace, tokenizer, device: torch.device) -> Path:
    cache_dir = Path(args.token_cache_dir) if args.token_cache_dir else Path(args.run_dir) / "token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = ChainDataset(args.source_dir, args.rough_dir, args.final_dir, args.image_size, args.pair_name_regex)
    missing_indices = []
    for i, item in enumerate(dataset.items):
        cache_path = cache_dir / f"{item[0].stem}.pt"
        if not cache_path.exists():
            missing_indices.append(i)
            continue
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception:
            missing_indices.append(i)
            continue
        if "source" not in cached or "rough" not in cached or "final" not in cached:
            missing_indices.append(i)
    if not missing_indices:
        return cache_dir

    subset = torch.utils.data.Subset(dataset, missing_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    tokenizer.eval()
    progress = tqdm(loader, desc="token cache")
    for batch in progress:
        source = batch["source"].to(device, non_blocking=True)
        rough = batch["rough"].to(device, non_blocking=True)
        final = batch["final"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            source_tokens = tokenizer.encode_indices(source).cpu()
            rough_tokens = tokenizer.encode_indices(rough).cpu()
            final_tokens = tokenizer.encode_indices(final).cpu()
        for name, source_item, rough_item, final_item in zip(batch["name"], source_tokens, rough_tokens, final_tokens):
            torch.save({"source": source_item, "rough": rough_item, "final": final_item}, cache_dir / f"{name}.pt")
    return cache_dir


def target_weight_map(target_image: torch.Tensor, foreground_weight: float, edge_weight: float) -> torch.Tensor:
    image = (target_image.float().clamp(-1, 1) + 1.0) * 0.5
    saturation = image.max(dim=1).values - image.min(dim=1).values
    not_white = (1.0 - image.mean(dim=1)).clamp(0, 1)
    foreground = ((saturation > 0.08) | (not_white > 0.18)).float()

    gray = image.mean(dim=1, keepdim=True)
    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    edges = (dx + dy).amax(dim=1).clamp(0, 1)

    weights = 1.0 + foreground * max(foreground_weight - 1.0, 0.0) + edges * edge_weight
    return weights


def token_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction="none")
    effective = mask.float()
    if weights is not None:
        if weights.shape[-2:] != target.shape[-2:]:
            weights = F.interpolate(weights.unsqueeze(1), size=target.shape[-2:], mode="area").squeeze(1)
        effective = effective * weights
    return (loss * effective).sum() / effective.sum().clamp_min(1.0)


def atomic_torch_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


@torch.no_grad()
def save_samples(path: Path, model, tokenizer, batch, args, device: torch.device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    tokenizer.eval()
    source = batch["source"].to(device, non_blocking=True)
    cond = token_source_condition(source, args.blur_factor, args.condition_mode)
    source_tokens = batch["source_tokens"].to(device, non_blocking=True)
    rough_tokens = batch["rough_tokens"].to(device, non_blocking=True)
    final_tokens = batch["final_tokens"].to(device, non_blocking=True)
    if args.stage == "rough":
        pred_tokens = generate_tokens(model, cond, args.sample_steps, temperature=args.sample_temperature, top_k=args.sample_top_k)
        pred = tokenizer.decode_indices(pred_tokens)
        target = batch["rough"].to(device, non_blocking=True)
        grid = torch.cat([source, pred, target], dim=0)
    else:
        rough_for_cond = source_tokens if args.refine_conditioning == "source-vq" else rough_tokens
        pred_tokens = generate_tokens(
            model,
            cond,
            args.sample_steps,
            rough_for_cond,
            temperature=args.sample_temperature,
            top_k=args.sample_top_k,
        )
        rough_image = tokenizer.decode_indices(rough_for_cond)
        pred = tokenizer.decode_indices(pred_tokens)
        target = batch["final"].to(device, non_blocking=True)
        grid = torch.cat([source, rough_image, pred, target], dim=0)
    save_image(denormalize(grid.cpu()), path, nrow=source.shape[0])
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
    tokenizer_state = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    tokenizer = build_vq_tokenizer_from_state(tokenizer_state, device).eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)

    token_cache_dir = build_token_cache(args, tokenizer, device)
    dataset = ChainTokenDataset(args.source_dir, args.rough_dir, args.final_dir, args.image_size, token_cache_dir, args.pair_name_regex)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    sample_loader = DataLoader(dataset, batch_size=min(args.batch_size, 8), shuffle=False, num_workers=0)

    vq_config = tokenizer.config
    grid_size = args.image_size // vq_config.downsample_factor
    config = TokenPredictorConfig(
        codebook_size=vq_config.codebook_size,
        token_dim=vq_config.embedding_dim,
        model_dim=args.model_dim,
        layers=args.layers,
        downsample_factor=vq_config.downsample_factor,
        condition_channels=token_condition_channels(args.condition_mode),
        grid_size=grid_size,
        stage=args.stage,
        dropout=args.dropout,
        architecture=args.architecture,
    )
    model = MaskedTokenPredictor(config).to(device)
    rough_condition_model = None
    if args.stage == "refine" and args.rough_predictor_checkpoint:
        rough_condition_model = build_token_predictor_from_state(
            torch.load(args.rough_predictor_checkpoint, map_location=device, weights_only=False),
            device,
        ).eval()
        for param in rough_condition_model.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
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
            csv.writer(handle).writerow(["epoch", "loss", "accuracy_masked"])

    for epoch in range(start_epoch, args.epochs):
        sums = {"loss": 0.0, "accuracy": 0.0}
        batches = 0
        progress = tqdm(loader, desc=f"{args.stage} token epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            source = batch["source"].to(device, non_blocking=True)
            cond = token_source_condition(source, args.blur_factor, args.condition_mode)
            source_tokens = batch["source_tokens"].to(device, non_blocking=True)
            rough_tokens = batch["rough_tokens"].to(device, non_blocking=True)
            final_tokens = batch["final_tokens"].to(device, non_blocking=True)
            target_tokens = rough_tokens if args.stage == "rough" else final_tokens
            target_image = batch["rough" if args.stage == "rough" else "final"].to(device, non_blocking=True)
            weights = target_weight_map(target_image, args.foreground_loss_weight, args.edge_loss_weight)
            masked_tokens, mask = make_masked_tokens(
                target_tokens,
                model.mask_token_id,
                args.mask_schedule,
                args.min_mask_ratio,
                args.full_mask_prob,
            )

            rough_condition_tokens = source_tokens if args.refine_conditioning == "source-vq" else rough_tokens
            if args.stage == "refine" and rough_condition_model is not None and torch.rand((), device=device).item() > args.teacher_forcing_rough_prob:
                with torch.no_grad():
                    rough_condition_tokens = generate_tokens(
                        rough_condition_model,
                        cond,
                        args.sample_steps,
                        temperature=args.sample_temperature,
                        top_k=args.sample_top_k,
                    )

            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(masked_tokens, cond, rough_condition_tokens if args.stage == "refine" else None)
                loss = token_loss(logits, target_tokens, mask, weights)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = ((pred == target_tokens) & mask).float().sum() / mask.float().sum().clamp_min(1.0)
            batches += 1
            sums["loss"] += loss.item()
            sums["accuracy"] += acc.item()
            progress.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc.item():.3f}"})

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([epoch + 1, f"{sums['loss'] / max(batches, 1):.6f}", f"{sums['accuracy'] / max(batches, 1):.6f}"])

        if (epoch + 1) % args.sample_every_epochs == 0:
            save_samples(sample_dir / f"epoch-{epoch + 1:04d}.png", model, tokenizer, next(iter(sample_loader)), args, device)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "config": vars(config),
            "args": vars(args),
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(state, checkpoint_dir / "latest.pt")
        if (epoch + 1) % args.save_every_epochs == 0:
            atomic_torch_save(state, checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")


if __name__ == "__main__":
    main()
