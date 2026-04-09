"""Application configuration — first-run detection, credential storage, preferences."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG_DIR = Path(os.environ.get("CYBERFORGE_CONFIG_DIR", "data"))
_CONFIG_FILE = "config.yaml"


class AppConfig(BaseModel):
    """User-facing configuration persisted across sessions."""

    # First-run flag
    onboarded: bool = False

    # Credentials (stored locally — never sent to third parties beyond intended APIs)
    hf_token: Optional[str] = None
    openrouter_key: Optional[str] = None

    # Directories
    cache_dir: str = "data/cache"
    saved_models_dir: str = "data/saved_models"
    reports_dir: str = "reports"

    # Preferences
    privacy: str = "local_only"  # local_only | allow_remote
    default_task_mode: str = "general"  # general | coding | cyber
    allow_benchmark_downloads: bool = True
    auto_cleanup_cache: bool = True

    # Hardware overrides (optional)
    force_cpu_only: bool = False

    # Database
    db_path: str = "data/cyberforge.db"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # OpenRouter
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


def config_path(config_dir: str | Path | None = None) -> Path:
    d = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR
    return d / _CONFIG_FILE


def load_config(config_dir: str | Path | None = None) -> AppConfig:
    p = config_path(config_dir)
    if not p.exists():
        return AppConfig()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AppConfig(**raw)


def save_config(cfg: AppConfig, config_dir: str | Path | None = None) -> Path:
    p = config_path(config_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return p


def is_first_run(config_dir: str | Path | None = None) -> bool:
    return not config_path(config_dir).exists()
