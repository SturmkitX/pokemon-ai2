from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from .dataset import VALID_EXTS


DEFAULT_PROMPT = (
    "a full-body original collectible monster creature inspired by the input person, "
    "non-human creature anatomy, expressive creature face matching the person's expression, "
    "wearing simplified versions of the person's clothes and accessories, same watch if visible, "
    "same dominant clothing colors, same pose and body attitude, charming game creature design, "
    "clean bold silhouette, colorful hand-painted anime game art, polished concept art, "
    "not a human, not a person in costume"
)

DEFAULT_NEGATIVE_PROMPT = (
    "human, humanoid, realistic person, person with animal ears, person with horns, cosplay, "
    "human face, human skin texture, human body, tail attached to a person, horns on a person, "
    "ordinary portrait, photorealistic, scary, horror, low quality, blurry, text, watermark"
)


@dataclass
class TeacherConfig:
    raw_dir: str = "data/raw_humans"
    pair_input_dir: str = "data/pairs/input"
    pair_target_dir: str = "data/pairs/target"
    cache_dir: str = "cache/teacher-sdxl"
    state_path: str = "runs/teacher-sdxl/state.jsonl"
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model: str = "xinsir/controlnet-openpose-sdxl-1.0"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight: str = "ip-adapter_sdxl_vit-h.safetensors"
    pose_detector_repo: str = "lllyasviel/ControlNet"
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    image_size: int = 1024
    num_variants: int = 1
    num_inference_steps: int = 24
    guidance_scale: float = 6.5
    strength: float = 0.82
    controlnet_scale: float = 0.7
    ip_adapter_scale: float = 0.45
    seed: int = 1337
    save_every: int = 2
    torch_dtype: str = "float16"
    device: str = "cuda"
    overwrite: bool = False
    enable_xformers: bool = False
    cpu_offload: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    defaults = TeacherConfig()
    for field_name in defaults.__dataclass_fields__:
        default = getattr(defaults, field_name)
        arg_name = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(arg_name, type=type(default), default=default)
    return parser.parse_args()


def image_fingerprint(path: Path, image_size: int) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{image_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def fit_rgb(path: Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS)


def discover_images(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    )


def load_completed(state_path: Path) -> set[str]:
    completed: set[str] = set()
    if not state_path.exists():
        return completed
    for line in state_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "done":
            completed.add(str(record.get("pair_name")))
    return completed


def append_state(state_path: Path, record: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def save_progress(cache_dir: Path, config: TeacherConfig, processed: int, total: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "processed": processed,
        "total": total,
        "config": asdict(config),
    }
    (cache_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch dtype: {name}")


def build_pipeline(config: TeacherConfig):
    try:
        from controlnet_aux import OpenposeDetector
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
    except ImportError as exc:
        raise ImportError(
            "Teacher generation requires extra packages. Install with: "
            "pip install diffusers transformers accelerate controlnet-aux opencv-python safetensors"
        ) from exc

    dtype = dtype_from_name(config.torch_dtype)
    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        config.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.load_ip_adapter(
        config.ip_adapter_repo,
        subfolder=config.ip_adapter_subfolder,
        weight_name=config.ip_adapter_weight,
    )
    pipe.set_ip_adapter_scale(config.ip_adapter_scale)

    if config.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(config.device)

    if config.enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except AttributeError:
        pass

    pose_detector = OpenposeDetector.from_pretrained(config.pose_detector_repo)
    return pipe, pose_detector


def cached_pose(
    pose_detector: Any,
    image: Image.Image,
    source_path: Path,
    cache_dir: Path,
    image_size: int,
) -> Image.Image:
    pose_dir = cache_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    pose_path = pose_dir / f"{image_fingerprint(source_path, image_size)}.png"
    if pose_path.exists():
        return Image.open(pose_path).convert("RGB")
    pose = pose_detector(image)
    pose = pose.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    pose.save(pose_path)
    return pose


def output_name(source: Path, variant: int, num_variants: int) -> str:
    if num_variants == 1:
        return source.stem
    return f"{source.stem}_v{variant:03d}"


def main() -> None:
    config = TeacherConfig(**vars(parse_args()))
    raw_dir = Path(config.raw_dir)
    pair_input_dir = Path(config.pair_input_dir)
    pair_target_dir = Path(config.pair_target_dir)
    cache_dir = Path(config.cache_dir)
    state_path = Path(config.state_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw image directory does not exist: {raw_dir}")

    pair_input_dir.mkdir(parents=True, exist_ok=True)
    pair_target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(raw_dir)
    if not images:
        raise RuntimeError(f"No source images found in {raw_dir}")

    completed = load_completed(state_path) if not config.overwrite else set()
    pipe, pose_detector = build_pipeline(config)

    total = len(images) * config.num_variants
    processed = 0
    for image_index, source_path in enumerate(tqdm(images, desc="teacher pairs")):
        source_image = fit_rgb(source_path, config.image_size)
        pose_image = cached_pose(pose_detector, source_image, source_path, cache_dir, config.image_size)

        for variant in range(config.num_variants):
            pair_name = output_name(source_path, variant, config.num_variants)
            input_path = pair_input_dir / f"{pair_name}.png"
            target_path = pair_target_dir / f"{pair_name}.png"
            processed += 1

            if pair_name in completed and input_path.exists() and target_path.exists():
                continue
            if target_path.exists() and not config.overwrite:
                append_state(
                    state_path,
                    {"pair_name": pair_name, "source": str(source_path), "target": str(target_path), "status": "done"},
                )
                continue

            generator = torch.Generator(device=config.device).manual_seed(
                config.seed + image_index * 1009 + variant
            )
            result = pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=source_image,
                control_image=pose_image,
                ip_adapter_image=source_image,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                strength=config.strength,
                controlnet_conditioning_scale=config.controlnet_scale,
                generator=generator,
            ).images[0]

            source_image.save(input_path)
            result.save(target_path)
            append_state(
                state_path,
                {
                    "pair_name": pair_name,
                    "source": str(source_path),
                    "input": str(input_path),
                    "target": str(target_path),
                    "pose": str(cache_dir / "poses" / f"{image_fingerprint(source_path, config.image_size)}.png"),
                    "variant": variant,
                    "status": "done",
                },
            )

            if processed % config.save_every == 0:
                save_progress(cache_dir, config, processed, total)

    save_progress(cache_dir, config, processed, total)


if __name__ == "__main__":
    main()
