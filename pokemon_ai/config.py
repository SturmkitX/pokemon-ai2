from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    input_dir: str = "data/pairs/input"
    target_dir: str = "data/pairs/target"
    cache_dir: str = "cache/pairs-256"
    run_dir: str = "runs/student-256"
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 100
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    base_channels: int = 48
    lambda_l1: float = 40.0
    lambda_perceptual: float = 4.0
    lambda_gan: float = 1.0
    save_every_steps: int = 2
    sample_every_steps: int = 50
    max_steps: int | None = None
    max_samples: int = 8
    amp: bool = True
    seed: int = 1337
    resume: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in valid})


def save_config(config: TrainConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def load_config(path: str | Path) -> TrainConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainConfig.from_dict(data)
