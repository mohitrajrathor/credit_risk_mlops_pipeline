"""Helpers for loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dictionary."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    return config


def load_data_config() -> dict[str, Any]:
    """Load the data configuration."""
    return load_yaml_config(CONFIG_DIR / "data_config.yaml")


def load_model_config() -> dict[str, Any]:
    """Load the model configuration."""
    return load_yaml_config(CONFIG_DIR / "model_config.yaml")


def load_training_config() -> dict[str, Any]:
    """Load the training configuration."""
    return load_yaml_config(CONFIG_DIR / "traning_config.yaml")


if __name__ == "__main__":
    print(load_data_config().keys())
