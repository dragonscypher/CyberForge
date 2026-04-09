"""Lifecycle manager — cache / permanent / volatile storage and cleanup."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ArtifactInfo(BaseModel):
    name: str
    path: str
    storage_class: str  # permanent | cache | volatile
    size_mb: float = 0.0
    created_ts: float = 0.0


class LifecycleManager:
    """Manage three-tier storage: permanent, cache, volatile."""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        saved_dir: str = "data/saved_models",
        volatile_dir: str = "data/cache/_volatile",
    ):
        self._cache = Path(cache_dir)
        self._saved = Path(saved_dir)
        self._volatile = Path(volatile_dir)
        for d in (self._cache, self._saved, self._volatile):
            d.mkdir(parents=True, exist_ok=True)

    # ── queries ──────────────────────────────────────────────────
    def list_cache(self) -> list[ArtifactInfo]:
        return self._list_dir(self._cache, "cache")

    def list_saved(self) -> list[ArtifactInfo]:
        return self._list_dir(self._saved, "permanent")

    def list_volatile(self) -> list[ArtifactInfo]:
        return self._list_dir(self._volatile, "volatile")

    # ── mutations ────────────────────────────────────────────────
    def promote_to_saved(self, cache_name: str, new_name: Optional[str] = None) -> Optional[str]:
        """Move a cached artifact to permanent storage."""
        src = self._cache / cache_name
        if not src.exists():
            return None
        dest_name = new_name or cache_name
        dest = self._saved / dest_name
        shutil.move(str(src), str(dest))
        return str(dest)

    def delete_artifact(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return True

    def cleanup_volatile(self) -> int:
        """Remove everything in volatile dir. Returns count of items deleted."""
        count = 0
        if self._volatile.exists():
            for item in self._volatile.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                count += 1
        return count

    def cleanup_stale_cache(self, max_age_hours: float = 24) -> int:
        """Remove cache items older than max_age_hours. Returns count deleted."""
        cutoff = time.time() - max_age_hours * 3600
        count = 0
        for item in self._cache.iterdir():
            if item.name.startswith("_"):
                continue
            if item.stat().st_mtime < cutoff:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                count += 1
        return count

    def disk_usage_mb(self) -> dict[str, float]:
        return {
            "cache": _dir_size_mb(self._cache),
            "saved": _dir_size_mb(self._saved),
            "volatile": _dir_size_mb(self._volatile),
        }

    # ── internals ────────────────────────────────────────────────
    @staticmethod
    def _list_dir(root: Path, storage_class: str) -> list[ArtifactInfo]:
        if not root.exists():
            return []
        items: list[ArtifactInfo] = []
        for p in root.iterdir():
            if p.name.startswith(".") or p.name.startswith("_"):
                continue
            items.append(
                ArtifactInfo(
                    name=p.name,
                    path=str(p),
                    storage_class=storage_class,
                    size_mb=_dir_size_mb(p) if p.is_dir() else round(p.stat().st_size / 1e6, 2),
                    created_ts=p.stat().st_mtime,
                )
            )
        return items


def _dir_size_mb(p: Path) -> float:
    if not p.is_dir():
        return round(p.stat().st_size / 1e6, 2) if p.exists() else 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1e6, 2)
