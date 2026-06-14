from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from .latent_model import ConditionalLatentUNet
from .train_flow import sample_flow
from .utils import denormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--noise-strength", type=float, default=None)
    parser.add_argument("--sampler", choices=["euler", "heun"], default="heun")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = state["args"]
    noise_strength = args.noise_strength if args.noise_strength is not None else float(train_args["noise_strength"])

    from diffusers import AutoencoderKL

    vae_kwargs = {"torch_dtype": torch.float16 if device.type == "cuda" else torch.float32}
    if train_args.get("vae_subfolder"):
        vae_kwargs["subfolder"] = train_args["vae_subfolder"]
    vae = AutoencoderKL.from_pretrained(train_args["vae_model"], **vae_kwargs).to(device)
    vae.eval()
    vae_dtype = next(vae.parameters()).dtype
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))

    model = ConditionalLatentUNet(base_channels=int(train_args["base_channels"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    image_size = int(train_args["image_size"])
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = Image.open(args.input).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        source = (vae.encode(tensor.to(dtype=vae_dtype)).latent_dist.mean * scaling).float()
        generated = sample_flow(
            model=model,
            source=source,
            steps=args.sample_steps,
            noise_strength=noise_strength,
            sampler=args.sampler,
            seed=args.seed,
            device=device,
        )
        output = vae.decode((generated / scaling).to(dtype=vae_dtype)).sample.squeeze(0).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(denormalize(output)).save(output_path)


if __name__ == "__main__":
    main()
