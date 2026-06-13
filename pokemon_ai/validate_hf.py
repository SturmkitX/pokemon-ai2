from __future__ import annotations

import argparse

from .hf_validate import HfValidationTarget, validate_hf_references
from .teacher import TeacherConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    defaults = TeacherConfig()
    fields = [
        "hf_dataset",
        "hf_config",
        "hf_split",
        "base_model",
        "controlnet_model",
        "ip_adapter_repo",
        "ip_adapter_subfolder",
        "ip_adapter_weight",
        "pose_detector_repo",
    ]
    for field_name in fields:
        parser.add_argument(
            "--" + field_name.replace("_", "-"),
            default=getattr(defaults, field_name),
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_hf_references(HfValidationTarget(**vars(args)))
    print("HF references ok")


if __name__ == "__main__":
    main()
