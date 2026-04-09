"""Hardware endpoints — profile system, get capabilities, preflight checks."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from packages.hardware.capability import (CapabilityMatrix, MachineClass,
                                          PreflightResult, SystemCapability,
                                          compute_capabilities,
                                          preflight_check)
from packages.hardware.profiler import HardwareProfile

router = APIRouter()


@router.get("/profile", response_model=HardwareProfile)
async def get_profile(request: Request):
    """Run a full hardware scan and return the profile."""
    profiler = request.app.state.profiler
    return profiler.profile()


@router.post("/profile/refresh", response_model=HardwareProfile)
async def refresh_profile(request: Request):
    """Force re-scan hardware and return updated profile (plan.md §5.2)."""
    profiler = request.app.state.profiler
    return profiler.profile(force=True)


@router.get("/capabilities", response_model=CapabilityMatrix)
async def get_capabilities(request: Request):
    """Compute capability matrix from current hardware."""
    profiler = request.app.state.profiler
    profile = profiler.profile()
    return compute_capabilities(profile)


@router.get("/machine-class", response_model=MachineClass)
async def get_machine_class(request: Request):
    """Classify machine into a generalized tier with derived limits."""
    profiler = request.app.state.profiler
    profile = profiler.profile()
    return MachineClass.classify(profile)


@router.get("/system-capability", response_model=SystemCapability)
async def get_system_capability(request: Request):
    """Per-backend capability snapshot — every backend assessed independently."""
    profiler = request.app.state.profiler
    profile = profiler.profile()
    return SystemCapability.from_profile(profile)


class PreflightRequest(BaseModel):
    action: str          # infer | chat | benchmark | quantize | train
    backend: str         # ollama | pytorch_cuda | pytorch_cpu | bitsandbytes | awq | gptq | vllm | tensorrt_llm | transformers
    model_params_b: float = 0.0
    model_size_mb: float = 0.0


@router.post("/preflight", response_model=PreflightResult)
async def run_preflight(body: PreflightRequest, request: Request):
    """Universal pre-action safety check.

    Checks backend availability AND model-size fit before
    allowing chat / benchmark / quantize / train.
    """
    profiler = request.app.state.profiler
    profile = profiler.profile()
    syscap = SystemCapability.from_profile(profile)
    return preflight_check(
        action=body.action,
        backend=body.backend,
        model_params_b=body.model_params_b,
        model_size_mb=body.model_size_mb,
        syscap=syscap,
    )
