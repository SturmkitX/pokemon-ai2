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
  --hf-dataset detection-datasets/coco `
  --hf-split train `
  --hf-image-column image `
  --hf-objects-column objects `
  --hf-object-category-column category `
  --hf-required-categories 0 `
  --max-source-images 200 `
  --pair-input-dir data/pairs/input `
  --pair-target-dir data/pairs/target `
  --cache-dir cache/teacher-sdxl `
  --state-path runs/teacher-sdxl/state.jsonl `
  --image-size 1024 `
  --num-variants 1 `
  --num-inference-steps 24 `
  --strength 0.82 `
  --controlnet-scale 0.7 `
  --ip-adapter-scale 0.45 `
  --save-every 2
```

`detection-datasets/coco` is the default because it is script-free/parquet on Hugging Face and has object annotations. Category `0` is COCO's `person` class, so the script filters toward images that actually contain people.

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

The default teacher stack is:

```text
Base model:        stabilityai/stable-diffusion-xl-base-1.0
Pose ControlNet:   xinsir/controlnet-openpose-sdxl-1.0
IP-Adapter:        h94/IP-Adapter, sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
Image encoder:     h94/IP-Adapter, models/image_encoder
Pose detector:     lllyasviel/ControlNet OpenPose annotator
```

The prompt is biased toward a non-human collectible creature that keeps the person's pose, expression, clothing colors, accessories, and watch if visible. It explicitly pushes away from "human with horns/tail" outputs.

## Train

The training loop caches resized tensors and saves checkpoints frequently.

```powershell
python -m pokemon_ai.train `
  --input-dir data/pairs/input `
  --target-dir data/pairs/target `
  --cache-dir cache/pairs-256 `
  --run-dir runs/student-256 `
  --image-size 256 `
  --batch-size 4 `
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
