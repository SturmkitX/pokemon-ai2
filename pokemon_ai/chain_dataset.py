from __future__ import annotations

from pathlib import Path
import re

from torch.utils.data import Dataset

from .dataset import VALID_EXTS, _load_rgb


class ChainDataset(Dataset):
    def __init__(
        self,
        source_dir: str | Path,
        rough_dir: str | Path,
        final_dir: str | Path,
        image_size: int,
        pair_name_regex: str = "",
    ) -> None:
        self.source_dir = Path(source_dir)
        self.rough_dir = Path(rough_dir)
        self.final_dir = Path(final_dir)
        self.image_size = image_size
        pattern = re.compile(pair_name_regex) if pair_name_regex else None

        rough_by_stem = {path.stem: path for path in self.rough_dir.iterdir() if path.suffix.lower() in VALID_EXTS}
        final_by_stem = {path.stem: path for path in self.final_dir.iterdir() if path.suffix.lower() in VALID_EXTS}
        self.items = []
        for source_path in sorted(self.source_dir.iterdir()):
            if source_path.suffix.lower() not in VALID_EXTS:
                continue
            if pattern is not None and not pattern.search(source_path.stem):
                continue
            rough_path = rough_by_stem.get(source_path.stem)
            final_path = final_by_stem.get(source_path.stem)
            if rough_path is not None and final_path is not None:
                self.items.append((source_path, rough_path, final_path))
        if not self.items:
            raise RuntimeError("No complete source/rough/final chain items found.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        source_path, rough_path, final_path = self.items[index]
        return {
            "source": _load_rgb(source_path, self.image_size),
            "rough": _load_rgb(rough_path, self.image_size),
            "final": _load_rgb(final_path, self.image_size),
            "name": source_path.stem,
        }
