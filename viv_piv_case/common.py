from __future__ import annotations

import json
import os
import pathlib
import platform
import sys
from typing import Any

import numpy as np


def load_config(path: pathlib.Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if os.environ.get("VIV_PIV_DATA_ROOT"):
        config["data_root"] = os.environ["VIV_PIV_DATA_ROOT"]
    if os.environ.get("VIV_PIV_OUTPUT_ROOT"):
        config["output_root"] = os.environ["VIV_PIV_OUTPUT_ROOT"]
    replacements = {
        "<PUBLIC_DATA_ROOT>": os.environ.get("PUBLIC_DATA_ROOT", "<PUBLIC_DATA_ROOT>"),
        "<HILDA_RESULTS_ROOT>": os.environ.get("HILDA_RESULTS_ROOT", "<HILDA_RESULTS_ROOT>"),
        "<EXTERNAL_DATA_ROOT>": os.environ.get("EXTERNAL_DATA_ROOT", "<EXTERNAL_DATA_ROOT>"),
        "<PRIVATE_DATA_ROOT>": os.environ.get("PRIVATE_DATA_ROOT", "<PRIVATE_DATA_ROOT>"),
        "<LOCAL_PATH>": os.environ.get("LOCAL_PATH", "<LOCAL_PATH>"),
    }

    def expand(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            expanded = os.path.expandvars(os.path.expanduser(value))
            for marker, replacement in replacements.items():
                expanded = expanded.replace(marker, replacement)
            return expanded
        return value

    config = expand(config)
    return config


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def software_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    for name in ("numpy", "scipy", "torch", "matplotlib"):
        try:
            module = __import__(name)
            result[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - environment audit
            result[name] = f"unavailable:{type(exc).__name__}"
    return result
