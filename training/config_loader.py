"""
Simple YAML config loader with environment variable substitution.
Replaces HYDRA for basic configuration management.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in config values.
    Supports ${oc.env:VAR_NAME,default_value} syntax.
    """
    if isinstance(value, str):
        # Pattern: ${oc.env:VAR_NAME,default} or ${oc.env:VAR_NAME}
        pattern = r'\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}'

        full_match = re.fullmatch(pattern, value)
        if full_match:
            var_name, default_val = full_match.groups()
            resolved = os.environ.get(var_name, default_val)
            return None if resolved is None or resolved.lower() == "null" else resolved

        def replace_env(match):
            var_name = match.group(1)
            default_val = match.group(2)
            return os.environ.get(var_name, default_val if default_val is not None else '')

        return re.sub(pattern, replace_env, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    else:
        return value


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML config file and perform environment variable substitution.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Recursively substitute environment variables
    config = _substitute_env_vars(config)

    return config


def dict_to_namespace(d: Dict[str, Any]):
    """
    Convert a dictionary to a namespace-like object for dot notation access.
    Recursively converts nested dicts and supports dict-like get() method and unpacking with **.
    """
    class Namespace:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    setattr(self, key, dict_to_namespace(value))
                else:
                    setattr(self, key, value)

        def __repr__(self):
            items = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
            return f"Namespace({items})"

        def get(self, key: str, default: Any = None) -> Any:
            """Dict-like get method for compatibility"""
            return getattr(self, key, default)

        def keys(self):
            """Dict-like keys method for unpacking with **"""
            return self.__dict__.keys()

        def values(self):
            """Dict-like values method"""
            return self.__dict__.values()

        def items(self):
            """Dict-like items method for unpacking with **"""
            return self.__dict__.items()

    return Namespace(**d)


def namespace_to_dict(ns) -> Dict[str, Any]:
    """
    Convert a Namespace object back to a dictionary recursively.
    Used for saving hyperparameters to checkpoints.
    """
    if isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    elif hasattr(ns, '__dict__'):
        return {k: namespace_to_dict(v) for k, v in ns.__dict__.items()}
    else:
        return ns
