"""Model registry — loads, queries, and manages the curated model catalogue."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelEntry(BaseModel):
    id: str
    display_name: str
    family: str = ""
    task_modes: list[str] = Field(default_factory=list)
    source: str = "huggingface"  # huggingface | ollama | byok
    hf_repo: Optional[str] = None
    ollama_tag: Optional[str] = None
    format: str = "safetensors"  # safetensors | gguf | adapter
    params: float = 0  # billions
    context_length: int = 4096
    license: str = ""
    recommended_backend: str = "ollama"
    quant_options: list[str] = Field(default_factory=list)
    min_vram: int = 0  # MiB
    min_ram: int = 0  # MiB
    supports_lora: bool = False
    supports_qlora: bool = False
    supports_ollama_import: bool = False
    supports_vllm: bool = False
    supports_tensorrt_llm: bool = False
    cyber_role: Optional[str] = None  # analyst | coder | guardrail | classifier
    notes: str = ""


_DEFAULT_REGISTRY = Path(__file__).with_name("registry.yaml")


class ModelRegistry:
    """In-memory model catalogue backed by a YAML file."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else _DEFAULT_REGISTRY
        self._models: list[ModelEntry] = []
        self.reload()

    def reload(self) -> None:
        if not self._path.exists():
            self._models = []
            return
        raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        self._models = [ModelEntry(**m) for m in raw.get("models", [])]

    @property
    def models(self) -> list[ModelEntry]:
        return list(self._models)

    def get(self, model_id: str) -> Optional[ModelEntry]:
        for m in self._models:
            if m.id == model_id:
                return m
        return None

    def search(
        self,
        task_mode: Optional[str] = None,
        source: Optional[str] = None,
        max_params: Optional[float] = None,
        max_vram: Optional[int] = None,
        max_ram: Optional[int] = None,
        cyber_role: Optional[str] = None,
    ) -> list[ModelEntry]:
        results = self._models
        if task_mode:
            results = [m for m in results if task_mode in m.task_modes]
        if source:
            results = [m for m in results if m.source == source]
        if max_params is not None:
            results = [m for m in results if m.params <= max_params]
        if max_vram is not None:
            results = [m for m in results if m.min_vram <= max_vram]
        if max_ram is not None:
            results = [m for m in results if m.min_ram <= max_ram]
        if cyber_role:
            results = [m for m in results if m.cyber_role == cyber_role]
        return results

    def compatible_with(
        self,
        vram_mb: int,
        ram_mb: int,
        task_mode: Optional[str] = None,
    ) -> list[ModelEntry]:
        """Return models the current hardware can run."""
        results = self._models
        if task_mode:
            results = [m for m in results if task_mode in m.task_modes]
        return [
            m
            for m in results
            if m.min_vram <= max(vram_mb, 0) or m.min_ram <= max(ram_mb, 0)
        ]

    def add(self, entry: ModelEntry) -> None:
        existing = self.get(entry.id)
        if existing:
            self._models.remove(existing)
        self._models.append(entry)

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self._path
        data = {"models": [m.model_dump() for m in self._models]}
        target.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
