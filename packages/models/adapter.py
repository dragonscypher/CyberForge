"""Adapter manager — tracks LoRA/QLoRA adapters, handles merge/export stubs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AdapterInfo(BaseModel):
    name: str
    base_model_id: str
    adapter_path: str
    method: str = "lora"  # lora | qlora
    task_mode: str = "general"
    created_at: str = ""
    saved: bool = False  # True = permanently saved, False = ephemeral cache


class AdapterManager:
    """Track and manage LoRA/QLoRA adapter artifacts."""

    def __init__(self, adapters_dir: str = "data/saved_models/adapters"):
        self._dir = Path(adapters_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._adapters: dict[str, AdapterInfo] = {}

    def register(self, info: AdapterInfo) -> None:
        self._adapters[info.name] = info

    def get(self, name: str) -> Optional[AdapterInfo]:
        return self._adapters.get(name)

    def list_all(self) -> list[AdapterInfo]:
        return list(self._adapters.values())

    def mark_saved(self, name: str) -> bool:
        if name in self._adapters:
            self._adapters[name].saved = True
            return True
        return False

    def delete(self, name: str) -> bool:
        import shutil

        info = self._adapters.pop(name, None)
        if info and Path(info.adapter_path).exists():
            shutil.rmtree(info.adapter_path, ignore_errors=True)
            return True
        return False

    def cleanup_ephemeral(self) -> int:
        """Remove all unsaved adapters. Returns count of deleted."""
        to_delete = [n for n, a in self._adapters.items() if not a.saved]
        for n in to_delete:
            self.delete(n)
        return len(to_delete)
