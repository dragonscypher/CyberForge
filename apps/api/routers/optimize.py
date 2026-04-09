"""Optimization router — quantize, compare, prune, distill, edit, smart-route endpoints."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from packages.core.smart_router import SmartRouter
from packages.models.distillation import (DistillConfig, DistillMethod,
                                          DistillResult, distill_model,
                                          list_distill_methods,
                                          suggest_distillation)
from packages.models.model_editor import (EditConfig, EditOperation,
                                          EditResult, MergeMethod, edit_model,
                                          list_edit_operations, suggest_edits)
from packages.models.pruning import (IterativePruneConfig,
                                     IterativePruneResult, PruneConfig,
                                     PruneMethod, PruneResult, iterative_prune,
                                     list_prune_methods, prune_model,
                                     suggest_pruning)
from packages.models.quantization import (QuantComparisonResult, QuantConfig,
                                          QuantMethod,
                                          check_method_availability,
                                          compare_quantization_methods,
                                          quantize)

router = APIRouter()


@router.get("/quantize/status")
async def quantize_status():
    """Pre-flight check: which quantization methods are available."""
    return await check_method_availability()


@router.post("/quantize/preflight")
async def quantize_preflight(req: "PreflightRequest"):
    """Hardware-aware preflight: Ollama status, VRAM, RAM, eligible methods."""
    avail = await check_method_availability()
    hw = avail.pop("_hardware", {})
    methods_summary = {}
    for method, info in avail.items():
        methods_summary[method] = {
            "available": info.get("available", False),
            "reason": info.get("reason", ""),
            "status": info.get("status", "ok"),
        }
    return {
        "hardware": hw,
        "methods": methods_summary,
        "source_model": req.source_model,
        "eligible_methods": [m for m, i in methods_summary.items() if i["available"]],
    }


class PreflightRequest(BaseModel):
    source_model: str = ""


class QuantizeRequest(BaseModel):
    source_model: str
    backend: str = "ollama"  # ollama | bnb_8bit | bnb_4bit | awq | gptq
    quant_method: str = "q4_k_m"
    temporary: bool = True
    run_benchmark_after: bool = False
    hf_token: Optional[str] = None


class QuantizeResponse(BaseModel):
    output_model: str = ""
    output_path: str = ""
    method: str = ""
    size_bytes: int = 0
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0


@router.post("/quantize", response_model=QuantizeResponse)
async def quantize_model(req: QuantizeRequest, request: Request):
    """Quantize a model via Ollama GGUF, bitsandbytes, AWQ, or GPTQ."""
    method_map = {
        "ollama": QuantMethod.OLLAMA_GGUF,
        "bnb_8bit": QuantMethod.BNB_8BIT,
        "bnb_4bit": QuantMethod.BNB_4BIT,
        "awq": QuantMethod.AWQ,
        "gptq": QuantMethod.GPTQ,
    }
    method = method_map.get(req.backend, QuantMethod.OLLAMA_GGUF)

    config = QuantConfig(
        source_model=req.source_model,
        method=method,
        quant_level=req.quant_method,
        temporary=req.temporary,
    )

    # If user provided an HF token, set it for this request
    if req.hf_token:
        import os
        os.environ["HF_TOKEN"] = req.hf_token

    ollama = getattr(request.app.state, "ollama", None)
    result = await quantize(config, ollama_client=ollama)

    return QuantizeResponse(
        output_model=result.output_model,
        output_path=result.output_path,
        method=result.method,
        size_bytes=result.size_bytes,
        success=result.success,
        error=result.error,
        duration_seconds=result.duration_seconds,
    )


class CompareQuantRequest(BaseModel):
    source_model: str
    methods: list[str] = ["ollama_gguf", "bnb_4bit", "awq", "gptq"]
    reference_size_bytes: int = 0


@router.post("/quantize/compare", response_model=QuantComparisonResult)
async def compare_quantization(req: CompareQuantRequest, request: Request):
    """Run multiple quantization methods on the same model and compare results."""
    ollama = getattr(request.app.state, "ollama", None)
    return await compare_quantization_methods(
        source_model=req.source_model,
        methods=req.methods,
        reference_size_bytes=req.reference_size_bytes,
        ollama_client=ollama,
    )


# ── Pruning ──────────────────────────────────────────────────────
@router.get("/prune/methods")
async def get_prune_methods():
    """List available pruning methods with descriptions."""
    return list_prune_methods()


class PruneRequest(BaseModel):
    source_model: str
    method: str = "magnitude"   # magnitude | l1_structured | random
    sparsity: float = 0.3       # 0.0–0.9
    make_permanent: bool = True


@router.post("/prune", response_model=PruneResult)
async def prune_model_endpoint(req: PruneRequest, request: Request):
    """Prune a model to reduce size while preserving quality."""
    method_map = {
        "magnitude": PruneMethod.MAGNITUDE,
        "l1_structured": PruneMethod.L1_STRUCTURED,
        "random": PruneMethod.RANDOM,
    }
    config = PruneConfig(
        source_model=req.source_model,
        method=method_map.get(req.method, PruneMethod.MAGNITUDE),
        sparsity=max(0.0, min(req.sparsity, 0.9)),
        make_permanent=req.make_permanent,
    )
    return await prune_model(config)


class PruneSuggestRequest(BaseModel):
    params_b: float
    target_vram_mb: int = 0


@router.post("/prune/suggest")
async def suggest_pruning_endpoint(req: PruneSuggestRequest, request: Request):
    """Suggest pruning strategy to fit a model into target VRAM."""
    profile = request.app.state.profiler.profile()
    vram = sum(g.vram_total_mb for g in profile.gpus)
    target = req.target_vram_mb if req.target_vram_mb > 0 else vram
    return suggest_pruning(req.params_b, vram, target)


# ── Iterative Pruning ──────────────────────────────────────────

class IterativePruneRequest(BaseModel):
    source_model: str
    method: str = "magnitude"
    start_sparsity: float = 0.1
    end_sparsity: float = 0.7
    step_size: float = 0.1
    max_perplexity_ratio: float = 1.5
    eval_text: str = ""
    make_permanent: bool = True


@router.post("/prune/iterative", response_model=IterativePruneResult)
async def iterative_prune_endpoint(req: IterativePruneRequest, request: Request):
    """Iterative pruning with quality verification at each sparsity step.

    Sweeps sparsity from start to end, measures perplexity at each step,
    stops when quality degrades beyond threshold.
    """
    method_map = {
        "magnitude": PruneMethod.MAGNITUDE,
        "l1_structured": PruneMethod.L1_STRUCTURED,
        "random": PruneMethod.RANDOM,
    }
    config = IterativePruneConfig(
        source_model=req.source_model,
        method=method_map.get(req.method, PruneMethod.MAGNITUDE),
        start_sparsity=max(0.05, min(req.start_sparsity, 0.9)),
        end_sparsity=max(0.1, min(req.end_sparsity, 0.9)),
        step_size=max(0.05, min(req.step_size, 0.3)),
        max_perplexity_ratio=max(1.1, req.max_perplexity_ratio),
        eval_text=req.eval_text,
        make_permanent=req.make_permanent,
    )
    return await iterative_prune(config)


# ── Distillation ────────────────────────────────────────────────

@router.get("/distill/methods")
async def get_distill_methods():
    """List available distillation methods with descriptions."""
    return list_distill_methods()


class DistillRequest(BaseModel):
    teacher_model: str
    student_model: str
    method: str = "logit"         # logit | hidden | progressive
    temperature: float = 2.0
    alpha: float = 0.5
    max_steps: int = 200
    learning_rate: float = 5e-5
    dataset_text: str = ""
    dataset_path: str = ""
    max_length: int = 256
    quality_threshold: float = 1.5


@router.post("/distill", response_model=DistillResult)
async def distill_model_endpoint(req: DistillRequest, request: Request):
    """Distill knowledge from a teacher model to a smaller student."""
    method_map = {
        "logit": DistillMethod.LOGIT,
        "hidden": DistillMethod.HIDDEN,
        "progressive": DistillMethod.PROGRESSIVE,
    }
    config = DistillConfig(
        teacher_model=req.teacher_model,
        student_model=req.student_model,
        method=method_map.get(req.method, DistillMethod.LOGIT),
        temperature=max(1.0, min(req.temperature, 10.0)),
        alpha=max(0.0, min(req.alpha, 1.0)),
        max_steps=max(10, min(req.max_steps, 5000)),
        learning_rate=req.learning_rate,
        dataset_text=req.dataset_text,
        dataset_path=req.dataset_path,
        max_length=max(64, min(req.max_length, 1024)),
        quality_threshold=max(1.1, req.quality_threshold),
    )
    return await distill_model(config)


class DistillSuggestRequest(BaseModel):
    teacher_params_b: float
    target_compression: float = 0.5


@router.post("/distill/suggest")
async def suggest_distillation_endpoint(req: DistillSuggestRequest, request: Request):
    """Suggest distillation configuration given hardware constraints."""
    profile = request.app.state.profiler.profile()
    vram = sum(g.vram_total_mb for g in profile.gpus)
    return suggest_distillation(req.teacher_params_b, vram, req.target_compression)


# ── Model Editing (OPT-014) ─────────────────────────────────────

@router.get("/edit/operations")
async def get_edit_operations():
    """List available model editing operations."""
    return list_edit_operations()


class EditRequest(BaseModel):
    source_model: str
    operation: str = "layer_remove"  # layer_remove | weight_merge | vocab_resize | head_prune
    layers_to_remove: list[int] = []
    merge_model: str = ""
    merge_method: str = "linear"     # linear | slerp
    merge_alpha: float = 0.5
    new_vocab_size: int = 0
    heads_to_prune: dict[int, list[int]] = {}
    num_heads_to_prune: int = 0


@router.post("/edit", response_model=EditResult)
async def edit_model_endpoint(req: EditRequest, request: Request):
    """Perform surgical model edits — layer removal, weight merging, etc."""
    op_map = {
        "layer_remove": EditOperation.LAYER_REMOVE,
        "weight_merge": EditOperation.WEIGHT_MERGE,
        "vocab_resize": EditOperation.VOCAB_RESIZE,
        "head_prune": EditOperation.HEAD_PRUNE,
    }
    merge_map = {
        "linear": MergeMethod.LINEAR,
        "slerp": MergeMethod.SLERP,
    }
    config = EditConfig(
        source_model=req.source_model,
        operation=op_map.get(req.operation, EditOperation.LAYER_REMOVE),
        layers_to_remove=req.layers_to_remove,
        merge_model=req.merge_model,
        merge_method=merge_map.get(req.merge_method, MergeMethod.LINEAR),
        merge_alpha=max(0.0, min(req.merge_alpha, 1.0)),
        new_vocab_size=max(0, req.new_vocab_size),
        heads_to_prune=req.heads_to_prune,
        num_heads_to_prune=max(0, req.num_heads_to_prune),
    )
    return await edit_model(config)


class EditSuggestRequest(BaseModel):
    params_b: float
    num_layers: int = 32
    target_vram_mb: int = 0


@router.post("/edit/suggest")
async def suggest_edit_endpoint(req: EditSuggestRequest, request: Request):
    """Suggest model editing operations to fit within target VRAM."""
    profile = request.app.state.profiler.profile()
    vram = sum(g.vram_total_mb for g in profile.gpus)
    target = req.target_vram_mb if req.target_vram_mb > 0 else vram
    return suggest_edits(req.params_b, req.num_layers, vram, target)


# ── Model-Info (for UI gating) ──────────────────────────────────

class ModelInfoRequest(BaseModel):
    model_name: str


class ModelInfoResponse(BaseModel):
    model_name: str
    is_ollama_tag: bool = False
    has_hf_mapping: bool = False
    hf_repo: str = ""
    is_quantized: bool = False
    quant_level: str = ""
    format: str = ""
    available_backends: list[str] = []
    warnings: list[str] = []


@router.post("/model-info", response_model=ModelInfoResponse)
async def model_info_endpoint(req: ModelInfoRequest, request: Request):
    """Check model type and return which optimization operations are valid.

    The UI uses this to gate invalid options (e.g. pruning an Ollama tag
    that has no HF mapping, or re-quantizing an already-quantized model).
    """
    import re
    from pathlib import Path

    name = req.model_name.strip()
    is_ollama = ":" in name
    has_hf = False
    hf_repo = ""
    is_quant = False
    quant_level = ""
    fmt = ""
    warnings: list[str] = []
    backends: list[str] = []

    # Check registry for HF mapping
    if is_ollama:
        try:
            import yaml
            reg = Path(__file__).resolve().parent.parent.parent.parent / "packages" / "models" / "registry.yaml"
            if reg.exists():
                data = yaml.safe_load(reg.read_text(encoding="utf-8")) or {}
                for m in data.get("models", []):
                    if m.get("ollama_tag") == name and m.get("hf_repo"):
                        has_hf = True
                        hf_repo = m["hf_repo"]
                        break
        except Exception:
            pass

    # Try to detect quantization via Ollama show
    if is_ollama:
        ollama = getattr(request.app.state, "ollama", None)
        if ollama:
            try:
                info = await ollama.show(name)
                details = info.get("details", {})
                quant_level = details.get("quantization_level", "")
                fmt = details.get("format", "")
                families = details.get("families", [])
                # A model is "already quantized" if it has a quant level like Q4_K_M
                if quant_level and re.search(r'[Qq]\d', quant_level):
                    is_quant = True
            except Exception:
                pass  # Model may not be pulled locally

    # Determine available backends
    if is_ollama and not is_quant:
        backends.append("ollama")
    if is_ollama and is_quant:
        # Re-quant via Ollama is risky — only allow if F16 source available
        backends.append("ollama")
        warnings.append(
            f"Model is already quantized ({quant_level}). Re-quantizing may degrade quality. "
            f"Ollama will attempt to find an FP16 source."
        )
    if has_hf or not is_ollama:
        backends.extend(["bnb_4bit", "bnb_8bit", "awq", "gptq"])

    # Pruning / distillation availability
    if not has_hf and is_ollama:
        warnings.append(
            f"Pruning, distillation, and editing require a HuggingFace model. "
            f"'{name}' has no HF mapping. Use a HuggingFace repo ID for these features."
        )

    return ModelInfoResponse(
        model_name=name,
        is_ollama_tag=is_ollama,
        has_hf_mapping=has_hf,
        hf_repo=hf_repo,
        is_quantized=is_quant,
        quant_level=quant_level,
        format=fmt,
        available_backends=backends,
        warnings=warnings,
    )


# ── Smart Routing (CORE-004) ────────────────────────────────────

@router.get("/route/status")
async def route_status(request: Request):
    """Get current smart router status with all backend health info."""
    router_obj = getattr(request.app.state, "smart_router", None)
    if not router_obj:
        router_obj = SmartRouter()
        profile = request.app.state.profiler.profile()
        vram = sum(g.vram_total_mb for g in profile.gpus)
        await router_obj.initialize(vram)
        request.app.state.smart_router = router_obj
    return router_obj.status()


class RouteRequest(BaseModel):
    task: str = "quantize"       # quantize | prune | distill | edit
    model_params_b: float = 0
    preferred_backend: str = ""
    strategy: str = "balanced"   # balanced | quality | speed


@router.post("/route")
async def route_task(req: RouteRequest, request: Request):
    """Route an optimization task to the best available backend (CORE-004)."""
    router_obj = getattr(request.app.state, "smart_router", None)
    if not router_obj:
        router_obj = SmartRouter(strategy=req.strategy)
        profile = request.app.state.profiler.profile()
        vram = sum(g.vram_total_mb for g in profile.gpus)
        await router_obj.initialize(vram)
        request.app.state.smart_router = router_obj

    if router_obj.strategy != req.strategy:
        router_obj.strategy = req.strategy

    profile = request.app.state.profiler.profile()
    vram = sum(g.vram_total_mb for g in profile.gpus)

    decision = router_obj.route(
        task=req.task,
        model_params_b=req.model_params_b,
        vram_mb=vram,
        preferred=req.preferred_backend,
    )
    return {
        "backend": decision.backend,
        "reason": decision.reason,
        "fallback_chain": decision.fallback_chain,
        "score": decision.score,
        "latency_ms": decision.latency_ms,
    }
