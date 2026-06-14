from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .teacher import (
    TeacherConfig,
    append_state,
    build_pipeline,
    cached_pose,
    iter_sources,
    load_completed,
    output_name,
    save_progress,
)


ROUGH_PROMPT = (
    "single centered pokemon creature rough design, full body, one character, "
    "simple silhouette, large readable shapes, flat colors from source clothes, "
    "plain background, no details, clean concept thumbnail"
)

FINAL_PROMPT = (
    "single centered pokemon creature, full body, one character, polished sprite art, "
    "clean black outlines, smooth cel shading, source clothing colors, cute creature design, "
    "simple background, high quality"
)

CHAIN_NEGATIVE_PROMPT = (
    "human, humanoid, cosplay, crowd, multiple characters, sidekick, busy background, "
    "cropped, blurry, sketch, messy linework, text, watermark"
)


def parse_args() -> argparse.Namespace:
    defaults = TeacherConfig()
    parser = argparse.ArgumentParser()

    for field_name in defaults.__dataclass_fields__:
        default = getattr(defaults, field_name)
        arg_name = "--" + field_name.replace("_", "-")
        if field_name in {"prompt", "negative_prompt"}:
            continue
        if isinstance(default, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(arg_name, type=type(default), default=default)

    parser.add_argument("--source-output-dir", default="data/pairs-pokemon-chain-v1/source")
    parser.add_argument("--rough-output-dir", default="data/pairs-pokemon-chain-v1/rough")
    parser.add_argument("--final-output-dir", default="data/pairs-pokemon-chain-v1/final")
    parser.add_argument("--rough-prompt", default=ROUGH_PROMPT)
    parser.add_argument("--final-prompt", default=FINAL_PROMPT)
    parser.add_argument("--negative-prompt", default=CHAIN_NEGATIVE_PROMPT)
    parser.add_argument("--rough-strength", type=float, default=0.86)
    parser.add_argument("--final-strength", type=float, default=0.42)
    parser.add_argument("--rough-steps", type=int, default=18)
    parser.add_argument("--final-steps", type=int, default=14)
    parser.add_argument("--rough-guidance-scale", type=float, default=8.5)
    parser.add_argument("--final-guidance-scale", type=float, default=8.0)
    parser.add_argument("--rough-controlnet-scale", type=float, default=0.72)
    parser.add_argument("--final-controlnet-scale", type=float, default=0.55)
    parser.add_argument("--rough-ip-adapter-scale", type=float, default=0.30)
    parser.add_argument("--final-ip-adapter-scale", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TeacherConfig(**{key: value for key, value in vars(args).items() if key in TeacherConfig.__dataclass_fields__})
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cache_dir = Path(config.cache_dir)
    state_path = Path(config.state_path)
    source_dir = Path(args.source_output_dir)
    rough_dir = Path(args.rough_output_dir)
    final_dir = Path(args.final_output_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    rough_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed(state_path) if not config.overwrite else set()
    pipe, pose_detector = build_pipeline(config)

    total = config.max_source_images * config.num_variants
    processed = 0
    source_count = 0
    for image_index, source in enumerate(tqdm(iter_sources(config, cache_dir), total=config.max_source_images, desc="teacher chain")):
        source_count += 1
        source_image = source.image
        pose_image = cached_pose(
            pose_detector,
            source_image,
            source.cache_key,
            cache_dir,
            config.pose_detect_resolution,
        )

        for variant in range(config.num_variants):
            pair_name = output_name(source.name, variant, config.num_variants)
            source_path = source_dir / f"{pair_name}.png"
            rough_path = rough_dir / f"{pair_name}.png"
            final_path = final_dir / f"{pair_name}.png"
            processed += 1

            if pair_name in completed and source_path.exists() and rough_path.exists() and final_path.exists():
                continue
            if final_path.exists() and rough_path.exists() and not config.overwrite:
                append_state(state_path, {"pair_name": pair_name, "source": source.source, "status": "done"})
                continue

            seed = config.seed + image_index * 1009 + variant
            rough_generator = torch.Generator(device=config.device).manual_seed(seed)
            pipe.set_ip_adapter_scale(args.rough_ip_adapter_scale)
            rough = pipe(
                prompt=args.rough_prompt,
                negative_prompt=args.negative_prompt,
                image=source_image,
                control_image=pose_image,
                ip_adapter_image=source_image,
                num_inference_steps=args.rough_steps,
                guidance_scale=args.rough_guidance_scale,
                strength=args.rough_strength,
                controlnet_conditioning_scale=args.rough_controlnet_scale,
                generator=rough_generator,
            ).images[0]

            final_generator = torch.Generator(device=config.device).manual_seed(seed + 1_000_003)
            pipe.set_ip_adapter_scale(args.final_ip_adapter_scale)
            final = pipe(
                prompt=args.final_prompt,
                negative_prompt=args.negative_prompt,
                image=rough,
                control_image=pose_image,
                ip_adapter_image=source_image,
                num_inference_steps=args.final_steps,
                guidance_scale=args.final_guidance_scale,
                strength=args.final_strength,
                controlnet_conditioning_scale=args.final_controlnet_scale,
                generator=final_generator,
            ).images[0]

            source_image.save(source_path)
            rough.save(rough_path)
            final.save(final_path)
            append_state(
                state_path,
                {
                    "pair_name": pair_name,
                    "source": source.source,
                    "source_image": str(source_path),
                    "rough": str(rough_path),
                    "final": str(final_path),
                    "status": "done",
                },
            )
            if processed % config.save_every == 0:
                save_progress(cache_dir, config, processed, total)

    if source_count == 0:
        raise RuntimeError("No usable source images were found.")
    save_progress(cache_dir, config, processed, total)


if __name__ == "__main__":
    main()
