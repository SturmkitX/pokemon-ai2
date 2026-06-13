from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HfValidationTarget:
    hf_dataset: str
    hf_config: str
    hf_split: str
    hf_streaming: bool
    base_model: str
    controlnet_model: str
    ip_adapter_repo: str
    ip_adapter_subfolder: str
    ip_adapter_weight: str
    pose_detector_repo: str


def validate_hf_references(target: HfValidationTarget) -> None:
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder
        from huggingface_hub import file_exists, model_info
    except ImportError as exc:
        raise ImportError(
            "HF reference validation requires datasets and huggingface_hub. "
            "Install the project requirements first."
        ) from exc

    model_info(target.base_model)
    model_info(target.controlnet_model)
    model_info(target.pose_detector_repo)
    model_info(target.ip_adapter_repo)

    ip_adapter_path = f"{target.ip_adapter_subfolder}/{target.ip_adapter_weight}".strip("/")
    if not file_exists(target.ip_adapter_repo, ip_adapter_path, repo_type="model"):
        raise FileNotFoundError(
            f"Missing IP-Adapter weight on Hugging Face: {target.ip_adapter_repo}/{ip_adapter_path}"
        )

    if target.hf_dataset:
        try:
            config_names = get_dataset_config_names(target.hf_dataset)
        except RuntimeError as exc:
            if "Dataset scripts are no longer supported" in str(exc):
                raise RuntimeError(
                    f"{target.hf_dataset} uses an old dataset loading script that this datasets "
                    "version refuses. Use a parquet/native dataset such as detection-datasets/coco."
                ) from exc
            raise

        if target.hf_config and target.hf_config not in config_names:
            raise ValueError(
                f"Dataset config {target.hf_config!r} was not found in {target.hf_dataset}. "
                f"Available configs: {config_names}"
            )

        split_config = target.hf_config or (config_names[0] if len(config_names) == 1 else None)
        split_names = get_dataset_split_names(target.hf_dataset, split_config) if split_config else []
        if split_names and target.hf_split not in split_names:
            raise ValueError(
                f"Dataset split {target.hf_split!r} was not found in {target.hf_dataset}. "
                f"Available splits: {split_names}"
            )

        if not target.hf_streaming:
            load_dataset_builder(target.hf_dataset, split_config)
