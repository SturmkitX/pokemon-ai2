from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

from .dataset import VALID_EXTS
from .hf_validate import HfValidationTarget, validate_hf_references


DEFAULT_PROMPT = (
    "a pokemon creature, non-human monster, same pose, matching expression, "
    "wearing simplified clothes and accessories from input, same clothing colors, "
    "cute creature design, finished clean game art, crisp outlines, sharp edges, "
    "clean black contour linework, smooth cel shading, solid color regions, "
    "complete polished character concept, high quality creature sprite art"
)

DEFAULT_NEGATIVE_PROMPT = (
    "human, humanoid, realistic person, person with animal ears, person with horns, cosplay, "
    "human face, human skin texture, human body, tail attached to a person, horns on a person, "
    "ordinary portrait, photorealistic, scary, horror, low quality, blurry, soft edges, "
    "unfinished, incomplete, missing outline, sketch, messy linework, jagged contours, "
    "muddy shading, rough coloring, noisy texture, text, watermark"
)


@dataclass
class TeacherConfig:
    hf_dataset: str = "detection-datasets/fashionpedia"
    hf_config: str = ""
    hf_split: str = "train"
    hf_image_column: str = "image"
    hf_caption_column: str = ""
    hf_caption_filter: str = ""
    hf_objects_column: str = "objects"
    hf_object_category_column: str = "category"
    hf_required_categories: str = ""
    hf_required_category_min_count: int = 0
    hf_required_category_max_count: int = -1
    hf_streaming: bool = False
    max_source_images: int = 200
    raw_dir: str = ""
    pair_input_dir: str = "data/pairs/input"
    pair_target_dir: str = "data/pairs/target"
    cache_dir: str = "cache/teacher-sdxl"
    state_path: str = "runs/teacher-sdxl/state.jsonl"
    model_family: str = "sd15"
    base_model: str = "lambda/sd-pokemon-diffusers"
    controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose"
    base_use_safetensors: bool = False
    controlnet_use_safetensors: bool = True
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_image_encoder_folder: str = "models/image_encoder"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight: str = "ip-adapter_sd15.safetensors"
    pose_detector_repo: str = "lllyasviel/ControlNet"
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    image_size: int = 512
    pose_detect_resolution: int = 384
    num_variants: int = 1
    generation_batch_size: int = 1
    num_inference_steps: int = 16
    guidance_scale: float = 8.5
    strength: float = 0.82
    controlnet_scale: float = 0.75
    ip_adapter_scale: float = 0.45
    scheduler: str = "dpm"
    detail_pass: bool = False
    detail_pass_steps: int = 8
    detail_pass_strength: float = 0.28
    detail_pass_guidance_scale: float = 8.5
    detail_pass_controlnet_scale: float = 0.55
    sharpen_outputs: bool = True
    sharpen_radius: float = 1.0
    sharpen_percent: int = 125
    sharpen_threshold: int = 3
    seed: int = 1337
    save_every: int = 2
    torch_dtype: str = "float16"
    device: str = "cuda"
    overwrite: bool = False
    enable_xformers: bool = False
    cpu_offload: bool = False
    vae_slicing: bool = False
    vae_tiling: bool = False
    validate_hf_refs: bool = True
    diffusers_progress: bool = False


@dataclass
class SourceImage:
    name: str
    image: Image.Image
    source: str
    cache_key: str


@dataclass
class PendingJob:
    pair_name: str
    source: SourceImage
    source_image: Image.Image
    pose_image: Image.Image
    input_path: Path
    target_path: Path
    variant: int
    seed: int
    pose_seconds: float
    queued_at: float


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


def file_fingerprint(path: Path, image_size: int) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{image_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def stable_fingerprint(raw: str, image_size: int) -> str:
    return hashlib.sha1(f"{raw}:{image_size}".encode("utf-8")).hexdigest()


def fit_rgb_image(image: Image.Image, image_size: int) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), (image_size, image_size), method=Image.Resampling.LANCZOS)


def fit_rgb_path(path: Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return fit_rgb_image(image, image_size)


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


def caption_matches(row: dict[str, Any], column: str, filters: list[str]) -> bool:
    if not filters or not column:
        return True
    value = row.get(column)
    if value is None:
        return True
    if isinstance(value, list):
        text = " ".join(str(item) for item in value).lower()
    else:
        text = str(value).lower()
    return any(item in text for item in filters)


def object_categories_match(
    row: dict[str, Any],
    objects_column: str,
    category_column: str,
    required: set[int],
    min_count: int,
    max_count: int,
) -> bool:
    if not required or not objects_column:
        return True
    objects = row.get(objects_column)
    if not objects:
        return False
    categories = objects.get(category_column) if isinstance(objects, dict) else None
    if categories is None:
        return False
    count = sum(1 for category in categories if int(category) in required)
    if count < min_count:
        return False
    if max_count >= 0 and count > max_count:
        return False
    return True


def pil_from_hf_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported Hugging Face image value type: {type(value)!r}")


def cache_source_image(cache_dir: Path, source: SourceImage) -> Path:
    source_dir = cache_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{source.cache_key}.png"
    if not path.exists():
        source.image.save(path)
    return path


def iter_hf_sources(config: TeacherConfig, cache_dir: Path) -> Iterator[SourceImage]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the datasets package. Install with: pip install datasets"
        ) from exc

    dataset_args: list[str] = [config.hf_dataset]
    if config.hf_config:
        dataset_args.append(config.hf_config)
    dataset = load_dataset(*dataset_args, split=config.hf_split, streaming=config.hf_streaming)
    filters = [item.strip().lower() for item in config.hf_caption_filter.split(",") if item.strip()]
    required_categories = {
        int(item.strip()) for item in config.hf_required_categories.split(",") if item.strip()
    }

    count = 0
    for row_index, row in enumerate(dataset):
        if count >= config.max_source_images:
            break
        if not caption_matches(row, config.hf_caption_column, filters):
            continue
        if not object_categories_match(
            row,
            config.hf_objects_column,
            config.hf_object_category_column,
            required_categories,
            config.hf_required_category_min_count,
            config.hf_required_category_max_count,
        ):
            continue
        try:
            image = pil_from_hf_value(row[config.hf_image_column])
        except Exception:
            continue
        source_id = f"hf:{config.hf_dataset}:{config.hf_config}:{config.hf_split}:{row_index}"
        cache_key = stable_fingerprint(source_id, config.image_size)
        source = SourceImage(
            name=f"hf_{count:06d}",
            image=fit_rgb_image(image, config.image_size),
            source=source_id,
            cache_key=cache_key,
        )
        cache_source_image(cache_dir, source)
        yield source
        count += 1


def iter_local_sources(config: TeacherConfig, cache_dir: Path) -> Iterator[SourceImage]:
    raw_dir = Path(config.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw image directory does not exist: {raw_dir}")
    images = discover_images(raw_dir)
    if not images:
        raise RuntimeError(f"No source images found in {raw_dir}")
    for source_path in images[: config.max_source_images]:
        source = SourceImage(
            name=source_path.stem,
            image=fit_rgb_path(source_path, config.image_size),
            source=str(source_path),
            cache_key=file_fingerprint(source_path, config.image_size),
        )
        cache_source_image(cache_dir, source)
        yield source


def iter_sources(config: TeacherConfig, cache_dir: Path) -> Iterator[SourceImage]:
    if config.hf_dataset:
        yield from iter_hf_sources(config, cache_dir)
        return
    if config.raw_dir:
        yield from iter_local_sources(config, cache_dir)
        return
    raise ValueError("Set --hf-dataset or --raw-dir before generating teacher pairs")


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
        from diffusers import (
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            UniPCMultistepScheduler,
        )
    except ImportError as exc:
        raise ImportError(
            "Teacher generation requires extra packages. Install with: "
            "pip install diffusers transformers accelerate controlnet-aux opencv-python safetensors"
        ) from exc

    dtype = dtype_from_name(config.torch_dtype)
    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_model,
        torch_dtype=dtype,
        use_safetensors=config.controlnet_use_safetensors,
    )
    pipeline_cls = (
        StableDiffusionXLControlNetImg2ImgPipeline
        if config.model_family == "sdxl"
        else StableDiffusionControlNetImg2ImgPipeline
    )
    pipeline_kwargs: dict[str, Any] = {
        "controlnet": controlnet,
        "torch_dtype": dtype,
        "use_safetensors": config.base_use_safetensors,
    }
    if config.model_family == "sd15":
        pipeline_kwargs["safety_checker"] = None
        pipeline_kwargs["requires_safety_checker"] = False
    pipe = pipeline_cls.from_pretrained(config.base_model, **pipeline_kwargs)
    if config.scheduler == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    elif config.scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.load_ip_adapter(
        config.ip_adapter_repo,
        subfolder=config.ip_adapter_subfolder,
        weight_name=config.ip_adapter_weight,
        image_encoder_folder=config.ip_adapter_image_encoder_folder,
    )
    pipe.set_ip_adapter_scale(config.ip_adapter_scale)

    if config.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(config.device)

    if config.enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    pipe.set_progress_bar_config(disable=not config.diffusers_progress)

    if config.vae_slicing:
        try:
            pipe.vae.enable_slicing()
        except AttributeError:
            pipe.enable_vae_slicing()

    if config.vae_tiling:
        try:
            pipe.vae.enable_tiling()
        except AttributeError:
            pipe.enable_vae_tiling()

    pose_detector = OpenposeDetector.from_pretrained(config.pose_detector_repo)
    if config.device != "cpu" and not config.cpu_offload and hasattr(pose_detector, "to"):
        pose_detector = pose_detector.to(config.device)
    return pipe, pose_detector


def cached_pose(
    pose_detector: Any,
    image: Image.Image,
    cache_key: str,
    cache_dir: Path,
    detect_resolution: int,
) -> Image.Image:
    pose_dir = cache_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    pose_path = pose_dir / f"{cache_key}.png"
    if pose_path.exists():
        return Image.open(pose_path).convert("RGB")
    try:
        pose = pose_detector(
            image,
            detect_resolution=detect_resolution,
            image_resolution=max(image.size),
            include_hand=False,
            include_face=False,
        )
    except TypeError:
        pose = pose_detector(image)
    pose = pose.convert("RGB").resize(image.size, Image.Resampling.BICUBIC)
    pose.save(pose_path)
    return pose


def output_name(source_name: str, variant: int, num_variants: int) -> str:
    if num_variants == 1:
        return source_name
    return f"{source_name}_v{variant:03d}"


def postprocess_target(image: Image.Image, config: TeacherConfig) -> Image.Image:
    if not config.sharpen_outputs:
        return image
    return image.filter(
        ImageFilter.UnsharpMask(
            radius=config.sharpen_radius,
            percent=config.sharpen_percent,
            threshold=config.sharpen_threshold,
        )
    )


def generate_batch(
    pipe: Any,
    config: TeacherConfig,
    jobs: list[PendingJob],
    state_path: Path,
    cache_dir: Path,
) -> None:
    if not jobs:
        return

    diffusion_start = time.perf_counter()
    generators = [
        torch.Generator(device=config.device).manual_seed(job.seed)
        for job in jobs
    ]
    results = pipe(
        prompt=[config.prompt] * len(jobs),
        negative_prompt=[config.negative_prompt] * len(jobs),
        image=[job.source_image for job in jobs],
        control_image=[job.pose_image for job in jobs],
        ip_adapter_image=[[job.source_image for job in jobs]],
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        strength=config.strength,
        controlnet_conditioning_scale=config.controlnet_scale,
        generator=generators,
    ).images

    if config.detail_pass:
        detail_generators = [
            torch.Generator(device=config.device).manual_seed(job.seed + 1_000_003)
            for job in jobs
        ]
        results = pipe(
            prompt=[config.prompt] * len(jobs),
            negative_prompt=[config.negative_prompt] * len(jobs),
            image=results,
            control_image=[job.pose_image for job in jobs],
            ip_adapter_image=[[job.source_image for job in jobs]],
            num_inference_steps=config.detail_pass_steps,
            guidance_scale=config.detail_pass_guidance_scale,
            strength=config.detail_pass_strength,
            controlnet_conditioning_scale=config.detail_pass_controlnet_scale,
            generator=detail_generators,
        ).images

    diffusion_seconds = time.perf_counter() - diffusion_start
    per_image_diffusion_seconds = diffusion_seconds / max(len(jobs), 1)

    for job, result in zip(jobs, results, strict=True):
        result = postprocess_target(result, config)
        save_start = time.perf_counter()
        job.source_image.save(job.input_path)
        result.save(job.target_path)
        save_seconds = time.perf_counter() - save_start
        append_state(
            state_path,
            {
                "pair_name": job.pair_name,
                "source": job.source.source,
                "input": str(job.input_path),
                "target": str(job.target_path),
                "source_cache": str(cache_dir / "sources" / f"{job.source.cache_key}.png"),
                "pose": str(cache_dir / "poses" / f"{job.source.cache_key}.png"),
                "variant": job.variant,
                "batch_size": len(jobs),
                "seconds_total": round(time.perf_counter() - job.queued_at, 3),
                "seconds_pose": round(job.pose_seconds, 3),
                "seconds_diffusion": round(per_image_diffusion_seconds, 3),
                "seconds_save": round(save_seconds, 3),
                "status": "done",
            },
        )


def main() -> None:
    config = TeacherConfig(**vars(parse_args()))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    pair_input_dir = Path(config.pair_input_dir)
    pair_target_dir = Path(config.pair_target_dir)
    cache_dir = Path(config.cache_dir)
    state_path = Path(config.state_path)

    pair_input_dir.mkdir(parents=True, exist_ok=True)
    pair_target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if config.validate_hf_refs:
        validate_hf_references(
            HfValidationTarget(
                hf_dataset=config.hf_dataset,
                hf_config=config.hf_config,
                hf_split=config.hf_split,
                hf_streaming=config.hf_streaming,
                base_model=config.base_model,
                controlnet_model=config.controlnet_model,
                ip_adapter_repo=config.ip_adapter_repo,
                ip_adapter_image_encoder_folder=config.ip_adapter_image_encoder_folder,
                ip_adapter_subfolder=config.ip_adapter_subfolder,
                ip_adapter_weight=config.ip_adapter_weight,
                pose_detector_repo=config.pose_detector_repo,
            )
        )

    completed = load_completed(state_path) if not config.overwrite else set()
    pipe, pose_detector = build_pipeline(config)

    total = config.max_source_images * config.num_variants
    processed = 0
    source_count = 0
    pending_jobs: list[PendingJob] = []
    for image_index, source in enumerate(tqdm(iter_sources(config, cache_dir), total=config.max_source_images, desc="teacher pairs")):
        pair_start = time.perf_counter()
        source_count += 1
        source_image = source.image
        pose_start = time.perf_counter()
        pose_image = cached_pose(
            pose_detector,
            source_image,
            source.cache_key,
            cache_dir,
            config.pose_detect_resolution,
        )
        pose_seconds = time.perf_counter() - pose_start

        for variant in range(config.num_variants):
            pair_name = output_name(source.name, variant, config.num_variants)
            input_path = pair_input_dir / f"{pair_name}.png"
            target_path = pair_target_dir / f"{pair_name}.png"
            processed += 1

            if pair_name in completed and input_path.exists() and target_path.exists():
                continue
            if target_path.exists() and not config.overwrite:
                append_state(
                    state_path,
                    {"pair_name": pair_name, "source": source.source, "target": str(target_path), "status": "done"},
                )
                continue

            pending_jobs.append(
                PendingJob(
                    pair_name=pair_name,
                    source=source,
                    source_image=source_image,
                    pose_image=pose_image,
                    input_path=input_path,
                    target_path=target_path,
                    variant=variant,
                    seed=config.seed + image_index * 1009 + variant,
                    pose_seconds=pose_seconds,
                    queued_at=pair_start,
                )
            )
            if len(pending_jobs) >= config.generation_batch_size:
                generate_batch(pipe, config, pending_jobs, state_path, cache_dir)
                pending_jobs.clear()

            if processed % config.save_every == 0:
                save_progress(cache_dir, config, processed, total)

    generate_batch(pipe, config, pending_jobs, state_path, cache_dir)

    if source_count == 0:
        raise RuntimeError(
            "No usable source images were found. Try a different --hf-dataset, --hf-split, "
            "--hf-image-column, or loosen --hf-caption-filter."
        )
    save_progress(cache_dir, config, processed, total)


if __name__ == "__main__":
    main()
