from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from .dataset import VALID_EXTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--rough-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--output-csv", default="runs/multi_character_candidates.csv")
    parser.add_argument("--review-dir", default="runs/multi_character_review")
    parser.add_argument("--min-component-area", type=int, default=900)
    parser.add_argument("--max-components", type=int, default=1)
    parser.add_argument("--background-margin", type=int, default=12)
    parser.add_argument("--diff-threshold", type=int, default=34)
    parser.add_argument("--move-flagged-to", default="")
    return parser.parse_args()


def list_names(directory: Path) -> set[str]:
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTS
    }


def estimate_background(image: Image.Image, margin: int) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    h, w, _ = arr.shape
    strips = [
        arr[:margin, :, :],
        arr[h - margin :, :, :],
        arr[:, :margin, :],
        arr[:, w - margin :, :],
    ]
    border = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    return np.median(border, axis=0)


def foreground_mask(image: Image.Image, margin: int, threshold: int) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    bg = estimate_background(image, margin)
    diff = np.abs(arr - bg).mean(axis=2)
    mask = diff > threshold
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def count_components(mask: np.ndarray, min_area: int) -> tuple[int, list[int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if area >= min_area:
                areas.append(area)
    return len(areas), sorted(areas, reverse=True)


def make_sheet(source: Image.Image, rough: Image.Image, final: Image.Image) -> Image.Image:
    images = [ImageOps.contain(image.convert("RGB"), (256, 256)) for image in [source, rough, final]]
    sheet = Image.new("RGB", (256 * 3, 256), (20, 20, 20))
    for index, image in enumerate(images):
        x = index * 256 + (256 - image.width) // 2
        y = (256 - image.height) // 2
        sheet.paste(image, (x, y))
    return sheet


def maybe_move_triplet(name: str, args: argparse.Namespace) -> None:
    if not args.move_flagged_to:
        return
    root = Path(args.move_flagged_to)
    for role, directory in [
        ("source", Path(args.source_dir)),
        ("rough", Path(args.rough_dir)),
        ("final", Path(args.final_dir)),
    ]:
        src = directory / f"{name}.png"
        if not src.exists():
            continue
        dst = root / role / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    rough_dir = Path(args.rough_dir)
    final_dir = Path(args.final_dir)
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    names = sorted(list_names(source_dir) & list_names(rough_dir) & list_names(final_dir))
    rows = []
    for name in tqdm(names, desc="scan multi-character"):
        rough = Image.open(rough_dir / f"{name}.png").convert("RGB")
        final = Image.open(final_dir / f"{name}.png").convert("RGB")
        rough_count, rough_areas = count_components(
            foreground_mask(rough, args.background_margin, args.diff_threshold),
            args.min_component_area,
        )
        final_count, final_areas = count_components(
            foreground_mask(final, args.background_margin, args.diff_threshold),
            args.min_component_area,
        )
        flagged = rough_count > args.max_components or final_count > args.max_components
        if flagged:
            source = Image.open(source_dir / f"{name}.png").convert("RGB")
            make_sheet(source, rough, final).save(review_dir / f"{name}.png")
            maybe_move_triplet(name, args)
        rows.append(
            {
                "name": name,
                "flagged": int(flagged),
                "rough_components": rough_count,
                "rough_areas": " ".join(str(area) for area in rough_areas[:8]),
                "final_components": final_count,
                "final_areas": " ".join(str(area) for area in final_areas[:8]),
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "flagged",
                "rough_components",
                "rough_areas",
                "final_components",
                "final_areas",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    flagged_count = sum(row["flagged"] for row in rows)
    print(f"Scanned {len(rows)} pairs. Flagged {flagged_count}. Review: {review_dir}")


if __name__ == "__main__":
    main()
