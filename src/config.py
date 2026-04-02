import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_env_yaml(path: Path, override: bool = False) -> Tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return True, "missing"

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, str(exc)

    try:
        parsed = yaml.safe_load(raw)
    except Exception as exc:
        return False, str(exc)

    data: Any = parsed
    if isinstance(data, dict) and "env" in data and isinstance(data.get("env"), dict):
        data = data.get("env")

    if not isinstance(data, dict):
        return False, "invalid_yaml_root"

    applied = 0
    for key, value in data.items():
        if key is None:
            continue
        k = str(key).strip()
        if not k:
            continue
        if value is None:
            continue
        if (not override) and (k in os.environ):
            continue
        os.environ[k] = str(value)
        applied += 1

    _ = applied
    return True, "ok"


def config_path_from_env(default: str = "conf/config.yaml") -> Path:
    value = str(os.environ.get("WEBOTS_AGENT_CONFIG") or "").strip()
    if value:
        return Path(value)
    return Path(default)
