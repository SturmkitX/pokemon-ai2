# Pokemon Human Stylizer

PyTorch starter implementation for a fast one-pass image-to-image student model:

```text
human image -> compact U-Net generator -> stylized Pokemon-like output
```

The intended workflow is:

1. Pull source human/person images from a Hugging Face dataset.
2. Use a stronger offline teacher model to create paired examples.
3. Write human inputs to `data/pairs/input`.
4. Write matching stylized targets to `data/pairs/target`.
5. Train this compact student for fast inference.

Matching is filename-based. For example:

```text
data/pairs/input/alex.png
data/pairs/target/alex.png
```

## Install

```powershell
pip install -r requirements.txt
```

## Generate Teacher Pairs

Generate paired training targets from a Hugging Face image dataset:

```powershell
python -m pokemon_ai.validate_hf
```

That preflight checks the default dataset, model repos, dataset split, and IP-Adapter weight path.

```powershell
python -m pokemon_ai.teacher `
  --hf-dataset detection-datasets/fashionpedia `
  --hf-split train `
  --hf-image-column image `
  --hf-objects-column objects `
  --hf-object-category-column category `
  --hf-required-categories "" `
  --max-source-images 1000 `
  --pair-input-dir data/pairs/input `
  --pair-target-dir data/pairs/target `
  --cache-dir cache/teacher-pokemon-sd15 `
  --state-path runs/teacher-pokemon-sd15/state.jsonl `
  --model-family sd15 `
  --base-model lambda/sd-pokemon-diffusers `
  --controlnet-model lllyasviel/control_v11p_sd15_openpose `
  --ip-adapter-subfolder models `
  --ip-adapter-weight ip-adapter_sd15.safetensors `
  --no-base-use-safetensors `
  --no-vae-slicing `
  --no-vae-tiling `
  --image-size 512 `
  --pose-detect-resolution 384 `
  --num-variants 2 `
  --generation-batch-size 4 `
  --num-inference-steps 20 `
  --guidance-scale 8.5 `
  --strength 0.82 `
  --controlnet-scale 0.75 `
  --ip-adapter-scale 0.45 `
  --sharpen-outputs `
  --save-every 2
```

For cleaner but slower targets, add a low-strength polish pass:

```powershell
  --detail-pass `
  --detail-pass-steps 8 `
  --detail-pass-strength 0.28 `
  --detail-pass-controlnet-scale 0.55
```

Diffusers' inner denoising progress bars are disabled by default so the outer `teacher pairs` bar is readable. Add `--diffusers-progress` if you want to see each denoising step. Per-pair timing is written into `runs/teacher-pokemon-sd15/state.jsonl`.

`detection-datasets/fashionpedia` is the default because it is script-free/parquet on Hugging Face and is fashion/person-centered instead of general scene photography. That gives better single-subject framing and stronger clothing/accessory signals than COCO.

COCO is still usable as a fallback if you want strict person-count filtering:

```powershell
python -m pokemon_ai.teacher `
  --hf-dataset detection-datasets/coco `
  --hf-split train `
  --hf-image-column image `
  --hf-objects-column objects `
  --hf-object-category-column category `
  --hf-required-categories 0 `
  --hf-required-category-min-count 1 `
  --hf-required-category-max-count 1
```

For a dataset without captions, pass an empty caption filter:

```powershell
python -m pokemon_ai.teacher `
  --hf-dataset your/dataset `
  --hf-split train `
  --hf-image-column image `
  --hf-caption-column "" `
  --hf-caption-filter "" `
  --hf-objects-column "" `
  --hf-required-categories "" `
  --max-source-images 500
```

Manual local photos are still supported by disabling HF and setting `--raw-dir`:

```powershell
python -m pokemon_ai.teacher --hf-dataset "" --raw-dir data/raw_humans
```

The default teacher stack is the faster Pokemon-specific SD 1.5 path:

```text
Base model:        lambda/sd-pokemon-diffusers
Pose ControlNet:   lllyasviel/control_v11p_sd15_openpose
IP-Adapter:        h94/IP-Adapter, models/ip-adapter_sd15.safetensors
Image encoder:     h94/IP-Adapter, models/image_encoder
Pose detector:     lllyasviel/ControlNet OpenPose annotator
```

The prompt is biased toward a non-human collectible creature that keeps the person's pose, expression, clothing colors, accessories, and watch if visible. It explicitly pushes away from "human with horns/tail" outputs.

## Train

The training loop caches resized tensors and saves checkpoints frequently.

Recommended next path: train the few-step latent student. This is better suited to the human-to-creature geometry change than the one-pass pixel student.

```powershell
python -m pokemon_ai.train_latent `
  --input-dir data/pairs-pokemon-sd15-v3/input `
  --target-dir data/pairs-pokemon-sd15-v3/target `
  --pair-name-regex "_v000$" `
  --image-cache-dir cache/pairs-pokemon-sd15-v3-256-latent-images `
  --latent-cache-dir cache/pairs-pokemon-sd15-v3-256-latents `
  --run-dir runs/latent-student-pokemon-sd15-v3-256 `
  --image-size 256 `
  --batch-size 32 `
  --latent-cache-batch-size 16 `
  --epochs 80 `
  --base-channels 128 `
  --sample-steps 8 `
  --train-step-choices 4,8,12 `
  --save-every-epochs 5 `
  --sample-every-epochs 2 `
  --amp
```

Latent inference:

```powershell
python -m pokemon_ai.infer_latent `
  --checkpoint runs/latent-student-pokemon-sd15-v3-256/checkpoints/latest.pt `
  --input path/to/human.png `
  --output out/latent-pokemon-human.png `
  --sample-steps 8
```

Older one-pass pixel student:

```powershell
python -m pokemon_ai.train `
  --input-dir data/pairs/input `
  --target-dir data/pairs/target `
  --cache-dir cache/pairs-256 `
  --run-dir runs/student-256 `
  --image-size 256 `
  --batch-size 16 `
  --save-every-steps 2
```

Tiny CPU-safe smoke run, useful before moving to the 3090:

```powershell
python -m pokemon_ai.train `
  --input-dir data/pairs/input `
  --target-dir data/pairs/target `
  --cache-dir cache/smoke-64 `
  --run-dir runs/smoke-64 `
  --image-size 64 `
  --batch-size 1 `
  --num-workers 0 `
  --base-channels 8 `
  --lambda-perceptual 0 `
  --max-steps 2 `
  --no-amp
```

Resume from the newest checkpoint:

```powershell
python -m pokemon_ai.train --resume runs/student-256/checkpoints/latest.pt
```

## Infer

```powershell
python -m pokemon_ai.infer `
  --checkpoint runs/student-256/checkpoints/latest.pt `
  --input path/to/human.png `
  --output out/pokemon-human.png
```

## Export

For low-performance inference hardware, export only the generator:

```powershell
python -m pokemon_ai.export `
  --checkpoint runs/student-256/checkpoints/latest.pt `
  --output runs/student-256/export/generator.torchscript
```

## Notes

- This is deliberately not diffusion at inference time.
- The student is lightweight enough to optimize, quantize, or export later.
- Dataset preprocessing is cached as `.pt` tensors.
- Full training state is checkpointed, including model, optimizer, scaler, step, and config.
