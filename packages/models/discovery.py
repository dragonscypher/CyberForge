"""Model discovery — scan Ollama and local folders for available models."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

log = logging.getLogger(__name__)


class DiscoveredModel(BaseModel):
    name: str
    source: str  # ollama | local_folder
    format: str = ""  # gguf | safetensors | pytorch
    size_bytes: int = 0
    quantization: str = ""
    path: str = ""
    family: str = ""


async def discover_ollama(
    ollama_client: "OllamaClient",  # type: ignore[name-defined]
) -> list[DiscoveredModel]:
    """Query Ollama for installed models."""
    try:
        if not await ollama_client.is_available():
            log.warning("Ollama is not available")
            return []

        infos = await ollama_client.list_model_info()
        return [
            DiscoveredModel(
                name=m.name,
                source="ollama",
                format=m.format or "gguf",
                size_bytes=m.size,
                quantization=m.quantization_level,
                path=f"ollama://{m.name}",
                family=m.family,
            )
            for m in infos
        ]
    except Exception:
        log.exception("Ollama discovery failed")
        return []


def discover_local_folder(
    folder: str | Path,
    extensions: tuple[str, ...] = (".gguf", ".safetensors", ".bin"),
) -> list[DiscoveredModel]:
    """Scan a local directory for model files."""
    folder = Path(folder)
    if not folder.exists():
        return []

    results: list[DiscoveredModel] = []
    seen_dirs: set[str] = set()

    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in extensions:
            continue

        # Group by parent directory for multi-file models (safetensors shards)
        model_dir = str(f.parent)
        if f.suffix == ".safetensors" and model_dir in seen_dirs:
            continue

        fmt = "gguf" if f.suffix == ".gguf" else "safetensors" if f.suffix == ".safetensors" else "pytorch"

        if f.suffix == ".safetensors":
            seen_dirs.add(model_dir)
            # Sum all safetensors in the same directory
            size = sum(sf.stat().st_size for sf in f.parent.glob("*.safetensors"))
            name = f.parent.name
            path = str(f.parent)
        else:
            size = f.stat().st_size
            name = f.stem
            path = str(f)

        results.append(
            DiscoveredModel(
                name=name,
                source="local_folder",
                format=fmt,
                size_bytes=size,
                path=path,
            )
        )

    return results


async def discover_all(
    ollama_client: "OllamaClient | None" = None,  # type: ignore[name-defined]
    local_folders: list[str | Path] | None = None,
) -> list[DiscoveredModel]:
    """Run all discovery sources and deduplicate."""
    models: list[DiscoveredModel] = []

    if ollama_client is not None:
        models.extend(await discover_ollama(ollama_client))

    if local_folders:
        for folder in local_folders:
            models.extend(discover_local_folder(folder))

    # Deduplicate by name (prefer ollama over local)
    seen: dict[str, DiscoveredModel] = {}
    for m in models:
        key = m.name.lower().replace(":", "_").replace("/", "_")
        if key not in seen or m.source == "ollama":
            seen[key] = m

    return list(seen.values())
