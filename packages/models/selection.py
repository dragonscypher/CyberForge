"""Model selection and cache cleanup — when user picks a model, clean up the rest.

Ticket: OPT-012
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


class SelectionResult(BaseModel):
    selected_model: str
    deleted_count: int = 0
    deleted_names: list[str] = []
    freed_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None


def select_model_and_cleanup(
    selected_model: str,
    cache_dir: str = "data/cache",
    keep_patterns: list[str] | None = None,
    ollama_client: Any = None,
) -> SelectionResult:
    """Select a model and delete other cached models.

    Args:
        selected_model: Name/id of the model to keep.
        cache_dir: Path to the cache directory.
        keep_patterns: Additional name patterns to keep (e.g. ["_volatile"]).
        ollama_client: Optional OllamaClient for Ollama model cleanup.

    Returns:
        SelectionResult with details of what was cleaned up.
    """
    keep = set(keep_patterns or [])
    keep.add("_volatile")  # Never delete volatile dir

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return SelectionResult(selected_model=selected_model)

    deleted_names: list[str] = []
    freed_bytes = 0

    for item in cache_path.iterdir():
        name = item.name

        # Skip the selected model
        if selected_model.lower().replace("/", "_") in name.lower():
            continue

        # Skip protected patterns
        if any(p in name for p in keep):
            continue

        # Calculate size before deletion
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            shutil.rmtree(item)
        elif item.is_file():
            size = item.stat().st_size
            item.unlink()
        else:
            continue

        freed_bytes += size
        deleted_names.append(name)
        log.info("Deleted cached model: %s (%.1f MB)", name, size / 1e6)

    return SelectionResult(
        selected_model=selected_model,
        deleted_count=len(deleted_names),
        deleted_names=deleted_names,
        freed_mb=round(freed_bytes / (1024 * 1024), 1),
        success=True,
    )


async def select_model_async(
    selected_model: str,
    cache_dir: str = "data/cache",
    keep_patterns: list[str] | None = None,
    ollama_client: Any = None,
    delete_ollama_others: bool = False,
) -> SelectionResult:
    """Async wrapper for model selection.

    If delete_ollama_others is True and ollama_client is provided,
    also deletes non-selected Ollama models (use with caution).
    """
    import asyncio

    result = await asyncio.to_thread(
        select_model_and_cleanup,
        selected_model,
        cache_dir,
        keep_patterns,
    )

    # Optionally clean Ollama models too
    if delete_ollama_others and ollama_client is not None:
        try:
            models = await ollama_client.list_models()
            for m in models:
                if selected_model.lower() not in m.lower():
                    try:
                        await ollama_client.delete(m)
                        result.deleted_names.append(f"ollama:{m}")
                        result.deleted_count += 1
                        log.info("Deleted Ollama model: %s", m)
                    except Exception as e:
                        log.warning("Failed to delete Ollama model %s: %s", m, e)
        except Exception as e:
            log.warning("Failed to list Ollama models for cleanup: %s", e)

    return result
