"""Capability matrix — translates a HardwareProfile into actionable flags."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from packages.hardware.profiler import HardwareProfile

# ── Per-backend availability status ──────────────────────────────

AvailabilityStatus = Literal[
    "available",
    "unavailable_by_config",
    "unavailable_by_hardware",
    "unavailable_by_service",
    "not_implemented",
]


class BackendDetail(BaseModel):
    """Single backend's availability with machine-honest reasoning."""
    name: str
    status: AvailabilityStatus = "not_implemented"
    reason: str = ""
    can_infer: bool = False
    can_quantize: bool = False
    can_train: bool = False


class SystemCapability(BaseModel):
    """Full per-backend capability snapshot for the current machine.

    Every backend is assessed independently — no single global
    'supported / unsupported' decision.
    """
    tier: str = "cpu_only"
    gpu_name: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    ram_total_mb: int = 0
    cpu_cores: int = 0
    os: str = ""
    has_nvidia_gpu: bool = False
    nvidia_cuda_version: str = ""

    # Derived limits (carried from MachineClass for convenience)
    max_native_params_b: float = 0.0
    max_postquant_params_b: float = 0.0
    safe_context_length: int = 4096

    # Per-backend detail
    backends: list[BackendDetail] = Field(default_factory=list)

    # Convenience roll-ups
    any_inference_available: bool = False
    any_quantize_available: bool = False
    any_train_available: bool = False

    @staticmethod
    def from_profile(profile: HardwareProfile) -> "SystemCapability":
        mc = MachineClass.classify(profile)
        b = profile.backends
        vram = mc.vram_total_mb
        has_gpu = mc.has_nvidia_gpu
        cuda_ver = profile.gpus[0].cuda_version if profile.gpus and profile.gpus[0].cuda_version else ""

        backends: list[BackendDetail] = []

        # ── Ollama ──
        if b.ollama:
            backends.append(BackendDetail(
                name="ollama", status="available",
                reason="Ollama HTTP API is running",
                can_infer=True, can_quantize=True, can_train=False,
            ))
        else:
            backends.append(BackendDetail(
                name="ollama", status="unavailable_by_service",
                reason="Ollama is not running — start with `ollama serve`",
                can_infer=False, can_quantize=False, can_train=False,
            ))

        # ── PyTorch CUDA ──
        if b.pytorch_cuda:
            backends.append(BackendDetail(
                name="pytorch_cuda", status="available",
                reason=f"PyTorch sees CUDA on {mc.gpu_name}",
                can_infer=True, can_quantize=True, can_train=True,
            ))
        elif has_gpu and not b.pytorch_cuda:
            backends.append(BackendDetail(
                name="pytorch_cuda", status="unavailable_by_config",
                reason="GPU detected but PyTorch CUDA is not available — install torch with CUDA support",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        else:
            backends.append(BackendDetail(
                name="pytorch_cuda", status="unavailable_by_hardware",
                reason="No NVIDIA GPU detected",
                can_infer=False, can_quantize=False, can_train=False,
            ))

        # ── PyTorch CPU (always present as fallback) ──
        backends.append(BackendDetail(
            name="pytorch_cpu", status="available",
            reason="PyTorch CPU inference always available (slow)",
            can_infer=True, can_quantize=False, can_train=False,
        ))

        # ── bitsandbytes ──
        if b.bitsandbytes and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="bitsandbytes", status="available",
                reason="bitsandbytes quantization available with CUDA",
                can_infer=True, can_quantize=True, can_train=True,
            ))
        elif b.bitsandbytes and not b.pytorch_cuda:
            backends.append(BackendDetail(
                name="bitsandbytes", status="unavailable_by_config",
                reason="bitsandbytes installed but PyTorch CUDA is missing",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        elif not b.bitsandbytes and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="bitsandbytes", status="unavailable_by_config",
                reason="PyTorch CUDA available but bitsandbytes not installed — `pip install bitsandbytes`",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        else:
            if has_gpu:
                backends.append(BackendDetail(
                    name="bitsandbytes", status="unavailable_by_config",
                    reason="GPU detected but both bitsandbytes and PyTorch CUDA are missing — install both",
                    can_infer=False, can_quantize=False, can_train=False,
                ))
            else:
                backends.append(BackendDetail(
                    name="bitsandbytes", status="unavailable_by_hardware",
                    reason="Requires NVIDIA GPU with CUDA",
                    can_infer=False, can_quantize=False, can_train=False,
                ))

        # ── AWQ ──
        _awq_importable = _can_import("awq")
        if _awq_importable and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="awq", status="available",
                reason="AWQ quantization available",
                can_infer=True, can_quantize=True, can_train=False,
            ))
        elif _awq_importable and not b.pytorch_cuda:
            backends.append(BackendDetail(
                name="awq", status="unavailable_by_config",
                reason="AWQ installed but PyTorch CUDA missing",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        elif not _awq_importable and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="awq", status="unavailable_by_config",
                reason="PyTorch CUDA available but autoawq not installed — `pip install autoawq`",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        else:
            if has_gpu:
                backends.append(BackendDetail(
                    name="awq", status="unavailable_by_config",
                    reason="GPU detected but autoawq and PyTorch CUDA are missing",
                    can_infer=False, can_quantize=False, can_train=False,
                ))
            else:
                backends.append(BackendDetail(
                    name="awq", status="unavailable_by_hardware",
                    reason="Requires NVIDIA GPU with CUDA",
                    can_infer=False, can_quantize=False, can_train=False,
                ))

        # ── GPTQ ──
        _gptq_importable = _can_import("auto_gptq")
        if _gptq_importable and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="gptq", status="available",
                reason="GPTQ quantization available",
                can_infer=True, can_quantize=True, can_train=False,
            ))
        elif _gptq_importable and not b.pytorch_cuda:
            backends.append(BackendDetail(
                name="gptq", status="unavailable_by_config",
                reason="GPTQ installed but PyTorch CUDA missing",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        elif not _gptq_importable and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="gptq", status="unavailable_by_config",
                reason="PyTorch CUDA available but auto-gptq not installed — `pip install auto-gptq`",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        else:
            if has_gpu:
                backends.append(BackendDetail(
                    name="gptq", status="unavailable_by_config",
                    reason="GPU detected but auto-gptq and PyTorch CUDA are missing",
                    can_infer=False, can_quantize=False, can_train=False,
                ))
            else:
                backends.append(BackendDetail(
                    name="gptq", status="unavailable_by_hardware",
                    reason="Requires NVIDIA GPU with CUDA",
                    can_infer=False, can_quantize=False, can_train=False,
                ))

        # ── vLLM ──
        if b.vllm and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="vllm", status="available",
                reason="vLLM serving available",
                can_infer=True, can_quantize=False, can_train=False,
            ))
        elif b.vllm and not b.pytorch_cuda:
            backends.append(BackendDetail(
                name="vllm", status="unavailable_by_config",
                reason="vLLM installed but PyTorch CUDA missing",
                can_infer=False, can_quantize=False, can_train=False,
            ))
        else:
            backends.append(BackendDetail(
                name="vllm", status="not_implemented",
                reason="vLLM not installed",
                can_infer=False, can_quantize=False, can_train=False,
            ))

        # ── TensorRT-LLM ──
        if b.tensorrt_llm and b.pytorch_cuda:
            backends.append(BackendDetail(
                name="tensorrt_llm", status="available",
                reason="TensorRT-LLM available",
                can_infer=True, can_quantize=True, can_train=False,
            ))
        else:
            backends.append(BackendDetail(
                name="tensorrt_llm", status="not_implemented",
                reason="TensorRT-LLM not installed",
                can_infer=False, can_quantize=False, can_train=False,
            ))

        # ── Transformers (HF) ──
        if b.transformers:
            backends.append(BackendDetail(
                name="transformers", status="available",
                reason="HuggingFace Transformers available for pruning/editing/distillation",
                can_infer=True, can_quantize=False, can_train=True,
            ))
        else:
            backends.append(BackendDetail(
                name="transformers", status="unavailable_by_config",
                reason="transformers not installed — `pip install transformers`",
                can_infer=False, can_quantize=False, can_train=False,
            ))

        any_infer = any(bd.can_infer for bd in backends)
        any_quant = any(bd.can_quantize for bd in backends)
        any_train = any(bd.can_train for bd in backends)

        return SystemCapability(
            tier=mc.tier,
            gpu_name=mc.gpu_name,
            vram_total_mb=mc.vram_total_mb,
            vram_free_mb=mc.vram_free_mb,
            ram_total_mb=mc.ram_total_mb,
            cpu_cores=profile.cpu_cores,
            os=profile.os,
            has_nvidia_gpu=mc.has_nvidia_gpu,
            nvidia_cuda_version=cuda_ver,
            max_native_params_b=mc.max_native_params_b,
            max_postquant_params_b=mc.max_postquant_params_b,
            safe_context_length=mc.safe_context_length,
            backends=backends,
            any_inference_available=any_infer,
            any_quantize_available=any_quant,
            any_train_available=any_train,
        )

    def backend(self, name: str) -> BackendDetail | None:
        """Look up a single backend by name."""
        for bd in self.backends:
            if bd.name == name:
                return bd
        return None

    def available_backends_for(self, action: str) -> list[BackendDetail]:
        """Return backends supporting a given action (infer/quantize/train)."""
        attr = {"infer": "can_infer", "quantize": "can_quantize", "train": "can_train"}.get(action, "")
        if not attr:
            return []
        return [bd for bd in self.backends if getattr(bd, attr, False)]


def _can_import(module: str) -> bool:
    import importlib
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


class PreflightResult(BaseModel):
    """Result of a pre-action safety check."""
    allowed: bool = False
    reason: str = ""
    category: str = ""  # fits | tight | partial_offload | cpu_only | too_large | no_backend | service_down
    backend_status: AvailabilityStatus = "not_implemented"
    suggestion: str = ""


def preflight_check(
    action: str,
    backend: str,
    model_params_b: float,
    model_size_mb: float,
    syscap: SystemCapability,
) -> PreflightResult:
    """Universal preflight: can this action run on this backend for this model on this machine?

    Actions: infer, quantize, train, benchmark, chat
    """
    # Normalize backend aliases (UI uses bnb_4bit/bnb_8bit, capability system uses bitsandbytes)
    _backend_alias = {"bnb_4bit": "bitsandbytes", "bnb_8bit": "bitsandbytes"}
    backend = _backend_alias.get(backend, backend)

    bd = syscap.backend(backend)
    if not bd:
        return PreflightResult(
            allowed=False, reason=f"Unknown backend '{backend}'",
            category="no_backend", backend_status="not_implemented",
        )

    if bd.status != "available":
        suggestion = ""
        if bd.status == "unavailable_by_service":
            suggestion = "Start the service and retry"
        elif bd.status == "unavailable_by_config":
            suggestion = bd.reason  # reason already contains install instructions
        elif bd.status == "unavailable_by_hardware":
            suggestion = "This backend requires hardware not present on this machine"
        return PreflightResult(
            allowed=False, reason=bd.reason,
            category="no_backend", backend_status=bd.status,
            suggestion=suggestion,
        )

    action_map = {"infer": "can_infer", "chat": "can_infer", "benchmark": "can_infer",
                  "quantize": "can_quantize", "train": "can_train"}
    cap_attr = action_map.get(action, "")
    if cap_attr and not getattr(bd, cap_attr, False):
        return PreflightResult(
            allowed=False,
            reason=f"Backend '{backend}' does not support '{action}'",
            category="no_backend", backend_status=bd.status,
        )

    # ── Size check ──
    vram = syscap.vram_total_mb
    vram_free = syscap.vram_free_mb
    ram = syscap.ram_total_mb
    est_runtime_mb = model_size_mb * 1.2 if model_size_mb > 0 else model_params_b * 700

    # Transformers without CUDA loads models on CPU regardless of GPU presence
    if backend == "transformers":
        cuda_bd = syscap.backend("pytorch_cuda")
        if not cuda_bd or cuda_bd.status != "available":
            if est_runtime_mb > 0 and est_runtime_mb >= ram * 0.7:
                return PreflightResult(
                    allowed=False,
                    reason=f"Model needs ~{round(est_runtime_mb)} MB. PyTorch CUDA unavailable — Transformers will use CPU. Only {ram} MB RAM.",
                    category="too_large", backend_status="available",
                    suggestion="Install PyTorch with CUDA support or choose a smaller model",
                )
            return PreflightResult(
                allowed=True,
                reason="Transformers will use CPU (PyTorch CUDA unavailable). May be slow for large models."
                    if est_runtime_mb == 0
                    else f"Transformers CPU-only (~{round(est_runtime_mb)} MB needed, {ram} MB RAM). May be slow.",
                category="cpu_only", backend_status="available",
            )

    if backend in ("ollama", "pytorch_cpu") and vram == 0:
        # CPU-only path
        if est_runtime_mb < ram * 0.7:
            return PreflightResult(
                allowed=True,
                reason=f"CPU-only inference (~{round(est_runtime_mb)} MB needed, {ram} MB RAM). Will be slow.",
                category="cpu_only", backend_status="available",
            )
        else:
            return PreflightResult(
                allowed=False,
                reason=f"Model needs ~{round(est_runtime_mb)} MB but only {ram} MB RAM (no GPU)",
                category="too_large", backend_status="available",
                suggestion="Choose a smaller model or add a GPU",
            )

    if vram > 0:
        if est_runtime_mb <= vram_free * 0.95:
            return PreflightResult(
                allowed=True,
                reason=f"Fits in GPU VRAM ({round(est_runtime_mb)} MB needed, {vram_free} MB free)",
                category="fits", backend_status="available",
            )
        elif est_runtime_mb <= vram:
            return PreflightResult(
                allowed=True,
                reason=f"Tight fit ({round(est_runtime_mb)} MB needed, {vram_free}/{vram} MB free). May need to close other GPU apps.",
                category="tight", backend_status="available",
            )
        elif est_runtime_mb <= vram + ram * 0.5:
            return PreflightResult(
                allowed=True,
                reason=f"Partial CPU offload ({round(est_runtime_mb)} MB needed, {vram} MB VRAM). Slower inference.",
                category="partial_offload", backend_status="available",
            )
        else:
            return PreflightResult(
                allowed=False,
                reason=f"Model needs ~{round(est_runtime_mb)} MB but system has {vram} MB VRAM + {ram} MB RAM",
                category="too_large", backend_status="available",
                suggestion="Choose a smaller model or quantize first",
            )

    # Fallback
    return PreflightResult(
        allowed=True, reason="Size check inconclusive — proceeding",
        category="fits", backend_status="available",
    )


class MachineClass(BaseModel):
    """Generalized machine tier — drives all UI gating and recommendation logic.

    Thresholds are intentionally rule-based, not hardcoded to a single GPU.
    """
    tier: str = "cpu_only"  # cpu_only | low_vram | mid_vram | high_vram
    gpu_name: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    ram_total_mb: int = 0
    has_nvidia_gpu: bool = False
    ollama_available: bool = False
    pytorch_cuda: bool = False
    transformers_available: bool = False
    bitsandbytes_available: bool = False

    # Derived limits
    max_native_params_b: float = 0.0      # largest model runnable now (Q4)
    max_postquant_params_b: float = 0.0   # largest model after quantization
    safe_context_length: int = 4096       # recommended max context
    allowed_quant_backends: list[str] = []  # e.g. ["ollama", "bnb_4bit", ...]
    allowed_optimizations: list[str] = []   # e.g. ["quantize", "prune", "distill"]

    @staticmethod
    def classify(profile: HardwareProfile) -> "MachineClass":
        vram = sum(g.vram_total_mb for g in profile.gpus)
        vram_free = sum(g.vram_free_mb for g in profile.gpus)
        ram = profile.ram_total_mb
        has_gpu = len(profile.gpus) > 0
        gpu_name = profile.gpus[0].name if profile.gpus else ""
        b = profile.backends

        # ── Tier classification ──
        if not has_gpu or vram == 0:
            tier = "cpu_only"
        elif vram < 6_000:
            tier = "low_vram"   # e.g. GTX 1060 3GB, MX350
        elif vram < 12_000:
            tier = "mid_vram"   # e.g. RTX 3060 6GB, RTX 4060 8GB, RTX 3070 8GB
        else:
            tier = "high_vram"  # e.g. RTX 3090 24GB, RTX 4090 24GB, A100

        # ── Max model sizes (conservative Q4 estimates) ──
        if tier == "cpu_only":
            max_native = round(ram / 1200, 1)    # CPU-only: need ~1.2 GB/param-B at Q4
            max_postquant = max_native            # No GPU to do faster quant
            safe_ctx = 2048
        elif tier == "low_vram":
            max_native = round(vram / 800, 1)    # ~800 MB/param-B at Q4 with overhead
            max_postquant = round(max(vram, ram) / 600, 1)
            safe_ctx = 4096
        elif tier == "mid_vram":
            max_native = round(vram / 700, 1)    # ~700 MB/param-B at Q4
            max_postquant = round(max(vram, ram) / 560, 1)
            safe_ctx = 32768
        else:  # high_vram
            max_native = round(vram / 650, 1)
            max_postquant = round(vram / 500, 1)  # Can afford larger quant on GPU
            safe_ctx = 131072

        # ── Allowed quant backends ──
        quant_backends: list[str] = []
        if b.ollama:
            quant_backends.append("ollama")
        if has_gpu and b.pytorch_cuda and b.bitsandbytes:
            quant_backends.extend(["bnb_4bit", "bnb_8bit"])
        if has_gpu and b.pytorch_cuda and b.transformers:
            quant_backends.extend(["awq", "gptq"])

        # ── Allowed optimizations ──
        allowed_opts = ["quantize"]  # Always allowed (Ollama GGUF works everywhere)
        if b.transformers:
            # prune, distill, edit all work on CPU via device_map="cpu";
            # GPU accelerates but is not required. Preflight already warns
            # when running on CPU-only ("May be slow on CPU").
            allowed_opts.extend(["prune", "distill", "edit"])

        return MachineClass(
            tier=tier,
            gpu_name=gpu_name,
            vram_total_mb=vram,
            vram_free_mb=vram_free,
            ram_total_mb=ram,
            has_nvidia_gpu=has_gpu,
            ollama_available=b.ollama,
            pytorch_cuda=b.pytorch_cuda,
            transformers_available=b.transformers,
            bitsandbytes_available=b.bitsandbytes,
            max_native_params_b=max_native,
            max_postquant_params_b=max_postquant,
            safe_context_length=safe_ctx,
            allowed_quant_backends=quant_backends,
            allowed_optimizations=allowed_opts,
        )


class CapabilityMatrix(BaseModel):
    can_run_3b_q4: bool = False
    can_run_7b_q4: bool = False
    can_run_7b_fp16: bool = False
    can_run_13b_q4: bool = False
    can_run_13b_fp16: bool = False
    can_run_70b_q4: bool = False
    can_train_lora_7b_4bit: bool = False
    can_train_lora_13b_4bit: bool = False
    can_serve_vllm: bool = False
    can_serve_vllm_awq: bool = False
    can_serve_tensorrt: bool = False
    can_only_cpu_gguf: bool = False
    max_model_params_b: float = 0.0
    recommended_quant: str = "q4_K_M"


# Rough VRAM estimates (MiB) for common configs
_VRAM_MAP = {
    "3b_q4": 2_500,
    "7b_q4": 5_000,
    "7b_fp16": 14_500,
    "13b_q4": 8_500,
    "13b_fp16": 27_000,
    "70b_q4": 40_000,
}

# LoRA training overhead on top of inference (approx)
_LORA_OVERHEAD_4BIT = {
    "7b": 8_000,   # ~8 GB VRAM for 7B QLoRA
    "13b": 16_000,
}


def _total_vram(profile: HardwareProfile) -> int:
    return sum(g.vram_total_mb for g in profile.gpus)


def compute_capabilities(profile: HardwareProfile) -> CapabilityMatrix:
    vram = _total_vram(profile)
    ram = profile.ram_total_mb
    has_gpu = len(profile.gpus) > 0

    cap = CapabilityMatrix()

    # Inference flags
    cap.can_run_3b_q4 = vram >= _VRAM_MAP["3b_q4"] or ram >= 6_000
    cap.can_run_7b_q4 = vram >= _VRAM_MAP["7b_q4"] or ram >= 10_000
    cap.can_run_7b_fp16 = vram >= _VRAM_MAP["7b_fp16"]
    cap.can_run_13b_q4 = vram >= _VRAM_MAP["13b_q4"] or ram >= 18_000
    cap.can_run_13b_fp16 = vram >= _VRAM_MAP["13b_fp16"]
    cap.can_run_70b_q4 = vram >= _VRAM_MAP["70b_q4"] or ram >= 48_000

    # Training flags
    cap.can_train_lora_7b_4bit = (
        has_gpu
        and vram >= _LORA_OVERHEAD_4BIT["7b"]
        and profile.backends.peft
        and profile.backends.bitsandbytes
    )
    cap.can_train_lora_13b_4bit = (
        has_gpu
        and vram >= _LORA_OVERHEAD_4BIT["13b"]
        and profile.backends.peft
        and profile.backends.bitsandbytes
    )

    # Serving flags
    cap.can_serve_vllm = has_gpu and profile.backends.vllm
    cap.can_serve_vllm_awq = cap.can_serve_vllm
    cap.can_serve_tensorrt = has_gpu and profile.backends.tensorrt_llm

    # CPU-only fallback
    cap.can_only_cpu_gguf = not has_gpu and ram >= 6_000

    # Estimated max param count the system can run (conservative, q4)
    if vram > 0:
        cap.max_model_params_b = round(vram / 700, 1)  # ~700 MB per B params at Q4
    else:
        cap.max_model_params_b = round(ram / 1200, 1)

    # Recommended quantization
    if vram >= _VRAM_MAP["7b_fp16"]:
        cap.recommended_quant = "fp16"
    elif vram >= _VRAM_MAP["7b_q4"]:
        cap.recommended_quant = "q4_K_M"
    elif ram >= 10_000:
        cap.recommended_quant = "q4_K_M"
    else:
        cap.recommended_quant = "q4_0"

    return cap
