"""
Configuration Loader for the Domain Q&A RAG Assistant.

Reads config.yaml once at import time and exposes a frozen
dictionary-like object to every module that does:

    from src.config import cfg

    model_name = cfg.llm.model
    top_k      = cfg.retrieval.top_k

Environment variable overrides follow the pattern:
    QA_<SECTION>_<KEY>   →   e.g.  QA_LLM_MODEL=microsoft/phi-2

Professional Commit Message:
    feat(config): add YAML config loader with env-var overrides
"""

import os
import yaml
from types import SimpleNamespace
from pathlib import Path


def _find_config_path() -> Path:
    """Walk up from this file's directory to find config.yaml."""
    current = Path(__file__).resolve().parent.parent  # project root
    config_path = current / "config.yaml"
    if config_path.exists():
        return config_path
    # Fallback: check working directory
    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config
    raise FileNotFoundError(
        "config.yaml not found. Expected at project root or working directory."
    )


def _apply_env_overrides(data: dict) -> dict:
    """
    Override any config value with an environment variable.
    Pattern: QA_<SECTION>_<KEY> in uppercase.
    Example: QA_LLM_MODEL=microsoft/phi-2
    """
    for section_key, section_val in data.items():
        if isinstance(section_val, dict):
            for key, default_val in section_val.items():
                env_name = f"QA_{section_key.upper()}_{key.upper()}"
                env_val = os.environ.get(env_name)
                if env_val is not None:
                    # Cast to the original type
                    if isinstance(default_val, int):
                        section_val[key] = int(env_val)
                    elif isinstance(default_val, float):
                        section_val[key] = float(env_val)
                    elif isinstance(default_val, bool):
                        section_val[key] = env_val.lower() in ("true", "1", "yes")
                    else:
                        section_val[key] = env_val
    return data


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert nested dicts to dot-accessible namespaces."""
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = _dict_to_namespace(val)
    return SimpleNamespace(**d)


def load_config() -> SimpleNamespace:
    """Load config.yaml, apply env overrides, return as namespace."""
    config_path = _find_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _apply_env_overrides(raw)
    return _dict_to_namespace(raw)


# ------------------------------------------------------------------
# Singleton: imported once, shared everywhere via `from src.config import cfg`
# ------------------------------------------------------------------
cfg = load_config()
