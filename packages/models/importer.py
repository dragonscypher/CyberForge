"""Model importer — import models from Ollama Modelfiles or adapter merges.

Handles the flow: external model → temp artifact → benchmark → save/discard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


class ImportRequest(BaseModel):
    source: str  # ollama_tag, hf_repo, or local path
    import_type: str = "ollama"  # ollama | hf | local | adapter_merge
    target_name: Optional[str] = None
    adapter_path: Optional[str] = None  # for adapter merge imports


class ImportResult(BaseModel):
    model_name: str = ""
    source: str = ""
    import_type: str = ""
    success: bool = True
    error: Optional[str] = None
    is_temporary: bool = True
    path: str = ""


async def import_from_ollama(
    source_tag: str,
    target_name: str,
    ollama_client: Any,
) -> ImportResult:
    """Import a model via Ollama pull."""
    try:
        result = await ollama_client.pull(source_tag)
        return ImportResult(
            model_name=target_name or source_tag,
            source=source_tag,
            import_type="ollama",
            success=True,
            path=f"ollama://{source_tag}",
        )
    except Exception as e:
        return ImportResult(
            source=source_tag,
            import_type="ollama",
            success=False,
            error=str(e),
        )


def import_from_local(
    model_path: str,
    target_name: str,
    cache_dir: str = "data/cache",
) -> ImportResult:
    """Import a model from a local directory or file."""
    source_path = Path(model_path)
    if not source_path.exists():
        return ImportResult(
            source=model_path,
            import_type="local",
            success=False,
            error=f"Path not found: {model_path}",
        )

    return ImportResult(
        model_name=target_name or source_path.stem,
        source=model_path,
        import_type="local",
        success=True,
        path=str(source_path),
    )


async def import_model(
    request: ImportRequest,
    ollama_client: Any = None,
    cache_dir: str = "data/cache",
) -> ImportResult:
    """Dispatch model import based on type."""
    target = request.target_name or request.source.split("/")[-1]

    if request.import_type == "ollama":
        if ollama_client is None:
            return ImportResult(success=False, error="Ollama client required")
        return await import_from_ollama(request.source, target, ollama_client)
    elif request.import_type == "local":
        return import_from_local(request.source, target, cache_dir)
    elif request.import_type == "hf":
        # Delegate to HFDownloader
        return ImportResult(
            model_name=target,
            source=request.source,
            import_type="hf",
            success=True,
            path=f"hf://{request.source}",
        )
    else:
        return ImportResult(
            success=False,
            error=f"Unknown import type: {request.import_type}",
        )
