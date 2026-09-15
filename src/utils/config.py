import os
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_config(cfg: Dict[str, Any], save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
