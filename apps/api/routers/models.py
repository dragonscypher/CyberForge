"""Model registry, discovery, download, import, web discovery & selection endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from packages.models.discovery import DiscoveredModel, discover_all
from packages.models.downloader import DownloadResult
from packages.models.importer import ImportRequest, ImportResult, import_model
from packages.models.registry import ModelEntry
from packages.models.selection import SelectionResult, select_model_async
from packages.models.web_discovery import WebDiscoveryResult, web_discover

router = APIRouter()


@router.get("/registry", response_model=list[ModelEntry])
async def list_registry(request: Request):
    return request.app.state.registry.models


@router.get("/registry/{model_id}", response_model=ModelEntry)
async def get_model(model_id: str, request: Request):
    entry = request.app.state.registry.get(model_id)
    if not entry:
        raise HTTPException(404, f"Model '{model_id}' not found in registry")
    return entry


@router.get("/registry/search/", response_model=list[ModelEntry])
async def search_models(
    request: Request,
    task_mode: Optional[str] = None,
    source: Optional[str] = None,
    max_params: Optional[float] = None,
    cyber_role: Optional[str] = None,
):
    return request.app.state.registry.search(
        task_mode=task_mode,
        source=source,
        max_params=max_params,
        cyber_role=cyber_role,
    )


@router.get("/discover", response_model=list[DiscoveredModel])
async def discover_models(request: Request):
    """Discover installed models from Ollama and local folders (plan.md §5.3)."""
    ollama = getattr(request.app.state, "ollama", None)
    config = request.app.state.config
    local_folders = [config.cache_dir] if hasattr(config, "cache_dir") else []
    return await discover_all(ollama_client=ollama, local_folders=local_folders)


class DownloadRequest(BaseModel):
    repo_id: str
    revision: str = "main"
    allow_patterns: Optional[list[str]] = None


@router.post("/download", response_model=DownloadResult)
async def download_model(body: DownloadRequest, request: Request):
    dl = request.app.state.downloader
    return dl.download(body.repo_id, body.revision, body.allow_patterns)


@router.post("/import", response_model=ImportResult)
async def import_model_endpoint(body: ImportRequest, request: Request):
    """Import a model from Ollama, HuggingFace, or local path (plan.md §5.3)."""
    ollama = getattr(request.app.state, "ollama", None)
    config = request.app.state.config
    cache_dir = config.cache_dir if hasattr(config, "cache_dir") else "data/cache"
    return await import_model(body, ollama_client=ollama, cache_dir=cache_dir)


@router.get("/cached", response_model=list[str])
async def list_cached(request: Request):
    return request.app.state.downloader.list_cached()


@router.delete("/cached/{repo_id:path}")
async def delete_cached(repo_id: str, request: Request):
    ok = request.app.state.downloader.delete_cached(repo_id)
    if not ok:
        raise HTTPException(404, "Cached model not found")
    return {"deleted": repo_id}


class WebDiscoverRequest(BaseModel):
    task_mode: str = "general"
    limit: int = 30


@router.post("/web-discover", response_model=WebDiscoveryResult)
async def web_discover_models(body: WebDiscoverRequest, request: Request):
    """Discover models from HuggingFace Hub based on hardware profile (DISC-001)."""
    profile = request.app.state.profiler.profile()
    vram = sum(g.vram_total_mb for g in profile.gpus)
    ram = profile.ram_total_mb

    # Include locally installed models
    ollama = getattr(request.app.state, "ollama", None)
    config = request.app.state.config
    local_folders = [config.cache_dir] if hasattr(config, "cache_dir") else []
    installed = await discover_all(ollama_client=ollama, local_folders=local_folders)
    installed_dicts = [m.model_dump() for m in installed]

    return await web_discover(
        vram_mb=vram,
        ram_mb=ram,
        task_mode=body.task_mode,
        installed_models=installed_dicts,
        limit=body.limit,
    )


class SelectRequest(BaseModel):
    selected_model: str
    delete_ollama_others: bool = False


@router.post("/select", response_model=SelectionResult)
async def select_model(body: SelectRequest, request: Request):
    """Select a model and clean up other cached models (OPT-012)."""
    config = request.app.state.config
    cache_dir = config.cache_dir if hasattr(config, "cache_dir") else "data/cache"
    ollama = getattr(request.app.state, "ollama", None)
    return await select_model_async(
        selected_model=body.selected_model,
        cache_dir=cache_dir,
        ollama_client=ollama,
        delete_ollama_others=body.delete_ollama_others,
    )
