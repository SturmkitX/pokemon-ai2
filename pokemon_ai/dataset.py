from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _fingerprint(path: Path, image_size: int) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{image_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(image)


class PairedImageDataset(Dataset):
    """Filename-matched image pairs with persistent tensor caching."""

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        cache_dir: str | Path,
        image_size: int = 256,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.cache_dir = Path(cache_dir)
        self.image_size = image_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory does not exist: {self.target_dir}")

        self.pairs = self._discover_pairs()
        if not self.pairs:
            raise RuntimeError(
                f"No filename-matched pairs found in {self.input_dir} and {self.target_dir}"
            )

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        targets_by_stem = {
            path.stem: path
            for path in self.target_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTS
        }
        pairs: list[tuple[Path, Path]] = []
        for input_path in sorted(self.input_dir.iterdir()):
            if not input_path.is_file() or input_path.suffix.lower() not in VALID_EXTS:
                continue
            target_path = targets_by_stem.get(input_path.stem)
            if target_path is not None:
                pairs.append((input_path, target_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _cached_tensor(self, path: Path, role: str) -> torch.Tensor:
        digest = _fingerprint(path, self.image_size)
        cache_path = self.cache_dir / role / f"{digest}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        tensor = _load_rgb(path, self.image_size)
        torch.save(tensor, cache_path)
        return tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        input_path, target_path = self.pairs[index]
        return {
            "input": self._cached_tensor(input_path, "input"),
            "target": self._cached_tensor(target_path, "target"),
            "name": input_path.stem,
        }
