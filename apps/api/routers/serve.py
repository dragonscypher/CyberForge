"""Serve endpoints — Ollama discovery, chat, OpenRouter chat."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from packages.serve.base import ChatMessage, ChatRequest, ChatResponse
from packages.serve.ollama import OllamaModelInfo

router = APIRouter()


# ── Ollama ───────────────────────────────────────────────────────
@router.get("/ollama/status")
async def ollama_status(request: Request):
    available = await request.app.state.ollama.is_available()
    return {"available": available}


@router.get("/ollama/models", response_model=list[OllamaModelInfo])
async def ollama_models(request: Request):
    client = request.app.state.ollama
    if not await client.is_available():
        raise HTTPException(503, "Ollama is not running")
    return await client.list_model_info()


@router.get("/ollama/running")
async def ollama_running(request: Request):
    client = request.app.state.ollama
    if not await client.is_available():
        raise HTTPException(503, "Ollama is not running")
    return await client.list_running()


@router.post("/ollama/chat", response_model=ChatResponse)
async def ollama_chat(body: ChatRequest, request: Request):
    client = request.app.state.ollama
    if not await client.is_available():
        raise HTTPException(503, "Ollama is not running")
    try:
        return await client.chat(body)
    except Exception as exc:
        raise HTTPException(502, f"Ollama chat error: {exc}") from exc


class PullRequest(BaseModel):
    model_name: str


@router.post("/ollama/pull")
async def ollama_pull(body: PullRequest, request: Request):
    client = request.app.state.ollama
    if not await client.is_available():
        raise HTTPException(503, "Ollama is not running")
    return await client.pull(body.model_name)


class CreateModelRequest(BaseModel):
    model_name: str
    from_model: str = ""
    quantize: str = ""
    system: str = ""
    modelfile: str = ""


class ModelFitRequest(BaseModel):
    model_name: str


@router.post("/ollama/model-fit")
async def ollama_model_fit(body: ModelFitRequest, request: Request):
    """Check if a model fits in available VRAM before attempting to run it.

    Uses Ollama /api/show to get model size, then compares against the
    hardware profiler's GPU VRAM.  Returns a recommendation: fits, tight,
    or too_large.
    """
    client = request.app.state.ollama
    if not await client.is_available():
        return {"fits": False, "reason": "Ollama is not running", "category": "service_down"}
    try:
        info = await client.show(body.model_name)
    except Exception:
        # Model not pulled yet — can't check size
        return {"fits": True, "reason": "Model not found locally — cannot estimate size", "category": "unknown"}

    model_size_bytes = info.get("size", 0)
    # Also check model_info for parameter count
    details = info.get("details", {})
    param_size = details.get("parameter_size", "")

    profiler = request.app.state.profiler
    profile = profiler.profile()
    vram_total = sum(g.vram_total_mb for g in profile.gpus)
    vram_free = sum(g.vram_free_mb for g in profile.gpus)
    ram_total = profile.ram_total_mb

    model_size_mb = model_size_bytes / (1024 * 1024) if model_size_bytes else 0

    # Ollama needs ~1.2x model size for runtime overhead (KV cache, etc.)
    estimated_runtime_mb = model_size_mb * 1.2

    result = {
        "model_name": body.model_name,
        "model_size_mb": round(model_size_mb),
        "estimated_runtime_mb": round(estimated_runtime_mb),
        "vram_total_mb": vram_total,
        "vram_free_mb": vram_free,
        "ram_total_mb": ram_total,
        "param_size": param_size,
    }

    if vram_total == 0:
        # CPU-only inference is possible but slow
        if estimated_runtime_mb < ram_total * 0.7:
            result.update({"fits": True, "category": "cpu_only",
                           "reason": f"No GPU — will run on CPU ({ram_total} MB RAM). May be slow."})
        else:
            result.update({"fits": False, "category": "too_large",
                           "reason": f"Model needs ~{round(estimated_runtime_mb)} MB but only {ram_total} MB RAM available (no GPU)."})
    elif estimated_runtime_mb <= vram_free * 0.95:
        result.update({"fits": True, "category": "fits",
                       "reason": f"Model fits in GPU VRAM ({round(estimated_runtime_mb)} MB needed, {vram_free} MB free)."})
    elif estimated_runtime_mb <= vram_total:
        result.update({"fits": True, "category": "tight",
                       "reason": f"Model may fit but VRAM is tight ({round(estimated_runtime_mb)} MB needed, {vram_free} MB free of {vram_total} MB total). Other processes may need to free VRAM."})
    elif estimated_runtime_mb <= vram_total + ram_total * 0.5:
        result.update({"fits": True, "category": "partial_offload",
                       "reason": f"Model too large for VRAM alone ({round(estimated_runtime_mb)} MB needed, {vram_total} MB VRAM). Ollama will offload to RAM — expect slower inference."})
    else:
        result.update({"fits": False, "category": "too_large",
                       "reason": f"Model needs ~{round(estimated_runtime_mb)} MB but system has {vram_total} MB VRAM + {ram_total} MB RAM. Choose a smaller model or quantize first."})
    return result


@router.post("/ollama/create")
async def ollama_create(body: CreateModelRequest, request: Request):
    client = request.app.state.ollama
    if not await client.is_available():
        raise HTTPException(503, "Ollama is not running")
    try:
        return await client.create(
            body.model_name,
            from_model=body.from_model or None,
            quantize=body.quantize or None,
            system=body.system or None,
            modelfile=body.modelfile or None,
        )
    except Exception as exc:
        raise HTTPException(502, f"Ollama create error: {exc}") from exc


# ── OpenRouter ───────────────────────────────────────────────────
@router.get("/openrouter/status")
async def openrouter_status(request: Request):
    available = await request.app.state.openrouter.is_available()
    return {"available": available}


@router.get("/openrouter/models")
async def openrouter_models(request: Request):
    client = request.app.state.openrouter
    if not await client.is_available():
        raise HTTPException(503, "OpenRouter API key not configured or unreachable")
    return await client.list_models()


@router.post("/openrouter/chat", response_model=ChatResponse)
async def openrouter_chat(body: ChatRequest, request: Request):
    client = request.app.state.openrouter
    if not await client.is_available():
        raise HTTPException(503, "OpenRouter API key not configured or unreachable")
    return await client.chat(body)
