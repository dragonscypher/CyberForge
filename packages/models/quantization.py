"""Quantization strategy selector — Ollama GGUF, bitsandbytes 8/4-bit, AWQ, GPTQ.

Never overwrites source model. Always produces a temp artifact first.
Supports latest quantization methods for comprehensive comparison.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

log = logging.getLogger(__name__)

_MODEL_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_name(raw: str) -> str:
    """Sanitise a model id for use as a filesystem directory name."""
    return _MODEL_ID_RE.sub("_", raw)


def _resolve_hf_repo(source_model: str) -> str:
    """Resolve an Ollama tag (e.g. 'qwen2.5:14b-instruct') to an HF repo ID.

    Uses the YAML registry as source of truth.  Falls back to the raw
    string when no mapping is found (so genuine HF repo IDs pass through).
    """
    if ":" not in source_model:
        return source_model  # already looks like an HF repo

    # Try loading the registry YAML for an exact ollama_tag → hf_repo match
    try:
        import yaml
        registry_path = Path(__file__).with_name("registry.yaml")
        if registry_path.exists():
            data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
            for m in data.get("models", []):
                if m.get("ollama_tag") == source_model and m.get("hf_repo"):
                    log.info("Resolved '%s' → '%s' via registry", source_model, m["hf_repo"])
                    return m["hf_repo"]
    except Exception as exc:
        log.debug("Registry lookup failed: %s", exc)

    # No registry match — return as-is (caller should handle the error)
    return source_model


def _trust_remote() -> bool:
    return os.environ.get("CYBERFORGE_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")


# ── Pre-flight dependency checks ────────────────────────────────


def _check_importable(*modules: str) -> str | None:
    """Return first missing module name, or None if all present.

    Uses find_spec to avoid heavy module loading (e.g. torch takes 20s+).
    """
    import importlib.util
    for m in modules:
        if importlib.util.find_spec(m) is None:
            return m
    return None


def _ollama_base_url() -> str:
    """Resolve Ollama base URL from OLLAMA_HOST env or default."""
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if host:
        if not host.startswith("http"):
            host = f"http://{host}"
        return host.rstrip("/")
    return "http://localhost:11434"


def _is_ollama_url_safe(url: str) -> bool:
    """SSRF guard: only allow localhost URLs for Ollama."""
    parsed = urlparse(url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


async def _check_ollama_reachable() -> bool:
    """Quick probe of Ollama REST API."""
    url = _ollama_base_url()
    if not _is_ollama_url_safe(url):
        log.warning("OLLAMA_HOST blocked — only localhost allowed")
        return False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


_METHOD_DEPS: dict[str, list[str]] = {
    "ollama_gguf": [],  # no pip deps — just needs Ollama running
    "bnb_8bit": ["torch", "transformers", "bitsandbytes"],
    "bnb_4bit": ["torch", "transformers", "bitsandbytes"],
    "awq": ["torch", "awq", "transformers"],
    "gptq": ["torch", "transformers", "optimum"],
}

_METHOD_INSTALL: dict[str, str] = {
    "ollama_gguf": "Install & start Ollama: https://ollama.com/download",
    "bnb_8bit": 'pip install torch transformers bitsandbytes accelerate',
    "bnb_4bit": 'pip install torch transformers bitsandbytes accelerate',
    "awq": 'pip install torch autoawq transformers',
    "gptq": 'pip install torch transformers optimum auto-gptq',
}


def _get_hardware_info(profile: "HardwareProfile | None" = None) -> dict:
    """Collect VRAM and RAM info for hardware-aware method gating.

    If a *profile* (from :class:`HardwareProfiler`) is provided, convert it
    instead of re-probing the system.

    Detects GPUs via **two independent methods** (when probing live):
    1. ``nvidia-smi`` — sees the physical GPU regardless of Python env.
       This is what Ollama uses.  Always reports the real VRAM.
    2. ``torch.cuda`` — only available when PyTorch is built with CUDA.
       Required for bitsandbytes / GPTQ / AWQ (they use PyTorch CUDA).

    The distinction matters: a user can have an RTX 3060 detected by
    nvidia-smi (and thus by Ollama) while PyTorch is CPU-only in the
    current venv.  We must **never** say "No CUDA GPU detected" when
    the physical GPU exists — only "PyTorch CUDA not available".
    """
    if profile is not None:
        # Fast path: convert an existing HardwareProfile
        gpu = profile.gpus[0] if profile.gpus else None
        return {
            "ram_total_mb": profile.ram_total_mb,
            "ram_available_mb": profile.ram_available_mb,
            "vram_total_mb": gpu.vram_total_mb if gpu else 0,
            "vram_free_mb": gpu.vram_free_mb if gpu else 0,
            "gpu_name": gpu.name if gpu else "",
            "nvidia_gpu_detected": len(profile.gpus) > 0,
            "torch_cuda_available": profile.backends.pytorch_cuda,
            "os": profile.os,
        }
    import platform as _platform
    import subprocess as _sp
    info: dict = {
        "ram_total_mb": 0, "ram_available_mb": 0,
        "vram_total_mb": 0, "vram_free_mb": 0,
        "gpu_name": "",
        "nvidia_gpu_detected": False,   # True if nvidia-smi sees a GPU
        "torch_cuda_available": False,   # True if torch.cuda works
        "os": _platform.system(),
    }
    # ── RAM ──
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_mb"] = mem.total // (1024 * 1024)
        info["ram_available_mb"] = mem.available // (1024 * 1024)
    except Exception:
        pass

    # ── GPU via nvidia-smi (independent of Python/PyTorch) ──
    try:
        creation_flags = _sp.CREATE_NO_WINDOW if _platform.system() == "Windows" else 0
        out = _sp.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
            creationflags=creation_flags,
        )
        for line in out.strip().splitlines():
            parts = [s.strip() for s in line.split(",")]
            if len(parts) >= 3:
                info["nvidia_gpu_detected"] = True
                info["gpu_name"] = parts[0]
                info["vram_total_mb"] = int(float(parts[1]))
                info["vram_free_mb"] = int(float(parts[2]))
                break  # use first GPU
    except Exception:
        pass

    # ── PyTorch CUDA (needed for bnb/AWQ/GPTQ, NOT for Ollama) ──
    try:
        import torch
        if torch.cuda.is_available():
            info["torch_cuda_available"] = True
            # Prefer torch's own VRAM figures when available (more precise)
            props = torch.cuda.get_device_properties(0)
            info["vram_total_mb"] = props.total_mem // (1024 * 1024)
            free, _ = torch.cuda.mem_get_info(0)
            info["vram_free_mb"] = free // (1024 * 1024)
            if not info["gpu_name"]:
                info["gpu_name"] = props.name
            info["nvidia_gpu_detected"] = True
    except Exception:
        pass
    return info


async def check_method_availability() -> dict[str, dict]:
    """Return per-method availability status with actionable messages.

    Returns dict like:
        {"ollama_gguf": {"available": True}, "awq": {"available": False,
         "missing": "awq", "install": "pip install autoawq transformers"}}

    Also includes a "_hardware" key with VRAM/RAM info.
    """
    import httpx as _httpx  # noqa: already imported at module level

    results: dict[str, dict] = {}
    ollama_ok = await _check_ollama_reachable()
    hw = _get_hardware_info()
    results["_hardware"] = hw

    for method_name, deps in _METHOD_DEPS.items():
        missing = _check_importable(*deps) if deps else None
        if missing:
            results[method_name] = {
                "available": False,
                "reason": f"Missing Python package: {missing}",
                "install": _METHOD_INSTALL[method_name],
                "status": "dependency_missing",
            }
        elif method_name == "ollama_gguf" and not ollama_ok:
            results[method_name] = {
                "available": False,
                "reason": f"Ollama is not running on {_ollama_base_url()}",
                "install": _METHOD_INSTALL[method_name],
                "status": "service_unavailable",
            }
        elif method_name in ("bnb_4bit", "bnb_8bit") and not hw["torch_cuda_available"]:
            # bnb needs PyTorch CUDA — distinguish "no GPU at all" from "GPU exists but torch is CPU-only"
            if hw["nvidia_gpu_detected"]:
                results[method_name] = {
                    "available": False,
                    "reason": (f"GPU detected ({hw['gpu_name']}) but PyTorch CUDA is not available in this Python environment. "
                               "bitsandbytes requires the CUDA build of PyTorch."),
                    "install": "pip install torch --index-url https://download.pytorch.org/whl/cu121",
                    "status": "dependency_missing",
                }
            else:
                results[method_name] = {
                    "available": False,
                    "reason": "No NVIDIA GPU detected — bitsandbytes requires a CUDA-capable GPU",
                    "install": "Install an NVIDIA GPU with CUDA support",
                    "status": "unsupported_hardware",
                }
        elif method_name == "awq" and hw["ram_total_mb"] < 16_000:
            results[method_name] = {
                "available": False,
                "reason": f"Insufficient total RAM ({hw['ram_total_mb']} MB). AWQ needs ~16 GB+ for most models",
                "install": "Increase system RAM to at least 16 GB",
                "status": "unsupported_hardware",
            }
        elif method_name == "gptq" and not hw["torch_cuda_available"]:
            if hw["nvidia_gpu_detected"]:
                results[method_name] = {
                    "available": False,
                    "reason": (f"GPU detected ({hw['gpu_name']}) but PyTorch CUDA is not available in this Python environment. "
                               "GPTQ requires the CUDA build of PyTorch."),
                    "install": "pip install torch --index-url https://download.pytorch.org/whl/cu121",
                    "status": "dependency_missing",
                }
            else:
                results[method_name] = {
                    "available": False,
                    "reason": "No NVIDIA GPU detected — GPTQ requires a CUDA-capable GPU",
                    "install": "Install an NVIDIA GPU with CUDA support",
                    "status": "unsupported_hardware",
                }
        else:
            results[method_name] = {"available": True, "status": "ok"}
    return results


class QuantMethod(str, Enum):
    OLLAMA_GGUF = "ollama_gguf"
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT = "bnb_4bit"
    AWQ = "awq"
    GPTQ = "gptq"


class QuantConfig(BaseModel):
    source_model: str  # Ollama model name or HF repo id
    method: QuantMethod = QuantMethod.OLLAMA_GGUF
    quant_level: str = "q4_k_m"  # for GGUF: q4_0, q4_k_m, q5_k_m, q8_0, etc.
    output_name: Optional[str] = None
    output_dir: str = "data/cache/quantized"
    temporary: bool = True
    # AWQ/GPTQ-specific
    bits: int = 4
    group_size: int = 128
    calibration_samples: int = 128
    calibration_dataset: str = "wikitext2"


class QuantResult(BaseModel):
    output_model: str = ""
    output_path: str = ""
    method: str = ""
    size_bytes: int = 0
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0


class QuantComparisonEntry(BaseModel):
    method: str
    quant_level: str = ""
    size_bytes: int = 0
    size_reduction_pct: float = 0.0
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    output_path: str = ""
    # Status category for UI display:
    #   ok, unsupported_hardware, config_error, service_unavailable,
    #   dependency_missing, model_error
    status: str = "ok"


class QuantComparisonResult(BaseModel):
    source_model: str
    entries: list[QuantComparisonEntry] = []
    best_size: Optional[str] = None
    best_speed: Optional[str] = None


async def quantize_ollama(
    config: QuantConfig,
    ollama_client: "OllamaClient",  # type: ignore[name-defined]
) -> QuantResult:
    """Quantize via Ollama's CREATE endpoint (new JSON API).

    Ollama only supports quantizing from F16/F32 models.  When the
    source model is already quantized (which is the default for most
    downloaded models), this function automatically pulls the fp16
    variant and quantizes from that instead.
    """
    start = time.time()
    output_name = config.output_name or f"{config.source_model}-{config.quant_level}"

    try:
        # ── 1.  Ensure the source model is available locally ─────────
        available = await ollama_client.list_models()
        if config.source_model not in available:
            matches = [m for m in available if m.startswith(config.source_model)]
            if not matches:
                log.info("Model '%s' not found locally — pulling from Ollama...", config.source_model)
                try:
                    await ollama_client.pull(config.source_model)
                    log.info("Successfully pulled '%s'", config.source_model)
                except Exception as pull_err:
                    return QuantResult(
                        success=False,
                        error=f"Model '{config.source_model}' not found in Ollama and auto-pull failed: {pull_err}. "
                              f"Available: {', '.join(available) if available else 'none'}.",
                        method=QuantMethod.OLLAMA_GGUF,
                        duration_seconds=round(time.time() - start, 2),
                    )

        # ── 2.  Check if source is already quantized ────────────────
        from_model = config.source_model
        try:
            info = await ollama_client.show(config.source_model)
            current_quant = (info.get("details", {}).get("quantization_level") or "").upper()
            model_format = (info.get("details", {}).get("format") or "").lower()
            log.info("Source model %s: format=%s, quant=%s", config.source_model, model_format, current_quant)

            is_full_precision = current_quant in ("", "F16", "F32", "FP16", "FP32") or "f16" in model_format.lower()

            if not is_full_precision:
                # Model is already quantized — need F16 version
                # Construct the F16 tag:  "qwen2.5:14b-instruct" → "qwen2.5:14b-instruct-fp16"
                # Also try: "name:XBtag" → "name:XBtag" with "-fp16" appended
                base_name = config.source_model
                # Strip existing quant suffix if present (e.g. "model:7b-q4_k_m" → "model:7b")
                fp16_candidates = []
                if ":" in base_name:
                    parts = base_name.split(":", 1)
                    fp16_candidates = [
                        f"{parts[0]}:{parts[1]}-fp16",
                        f"{parts[0]}:fp16",
                    ]
                else:
                    fp16_candidates = [f"{base_name}:fp16"]

                log.info("Source is already %s — need F16 for re-quantization. Trying: %s", current_quant, fp16_candidates)

                fp16_found = False
                for fp16_tag in fp16_candidates:
                    try:
                        log.info("Pulling F16 variant: %s", fp16_tag)
                        await ollama_client.pull(fp16_tag)
                        from_model = fp16_tag
                        fp16_found = True
                        log.info("Successfully pulled F16: %s", fp16_tag)
                        break
                    except Exception:
                        log.debug("F16 tag '%s' not found, trying next", fp16_tag)
                        continue

                if not fp16_found:
                    return QuantResult(
                        success=False,
                        error=f"Model '{config.source_model}' is already quantized ({current_quant}). "
                              f"Ollama requires F16/F32 source for re-quantization. "
                              f"Tried pulling F16 variants {fp16_candidates} but none were available. "
                              f"You can manually pull an F16 version with: ollama pull {fp16_candidates[0]}",
                        method=QuantMethod.OLLAMA_GGUF,
                        duration_seconds=round(time.time() - start, 2),
                    )
        except RuntimeError:
            raise
        except Exception as show_err:
            log.warning("Could not check source model format: %s (proceeding anyway)", show_err)

        # ── 3.  Create the quantized model ───────────────────────────
        log.info("Creating quantized model %s via Ollama (from=%s, quantize=%s)", output_name, from_model, config.quant_level)
        result = await ollama_client.create(
            model_name=output_name,
            from_model=from_model,
            quantize=config.quant_level,
        )
        log.info("Ollama create result: %s", result)

        # ── 4.  Get output model size ────────────────────────────────
        size = 0
        try:
            infos = await ollama_client.list_model_info()
            for mi in infos:
                if mi.name == output_name or mi.name.startswith(output_name + ":"):
                    size = mi.size
                    break
            if not size:
                out_info = await ollama_client.show(output_name)
                size = out_info.get("size", 0)
        except Exception:
            pass

        return QuantResult(
            output_model=output_name,
            output_path=f"ollama://{output_name}",
            method=QuantMethod.OLLAMA_GGUF,
            size_bytes=size,
            success=True,
            duration_seconds=round(time.time() - start, 2),
        )
    except Exception as e:
        log.exception("Ollama quantization failed")
        return QuantResult(success=False, error=str(e), method=QuantMethod.OLLAMA_GGUF,
                           duration_seconds=round(time.time() - start, 2))


def _quantize_bnb_sync(config: QuantConfig, bits: int) -> QuantResult:
    """Blocking bitsandbytes quantization — load model in N-bit and re-save."""
    start = time.time()

    # Resolve Ollama tags (e.g. 'qwen2.5:14b-instruct') to HF repo IDs
    model_id = _resolve_hf_repo(config.source_model)
    if ":" in model_id:
        return QuantResult(
            success=False,
            error=f"Cannot use Ollama tag '{config.source_model}' with bitsandbytes. "
                  f"Provide a HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-14B-Instruct') "
                  f"or add the model to registry.yaml with its hf_repo field.",
            method=config.method,
            duration_seconds=round(time.time() - start, 2),
        )

    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
    except ImportError as e:
        return QuantResult(
            success=False,
            error=f"GPU dependencies required: {e}. Install with: pip install -e \".[gpu]\"",
            method=config.method,
            duration_seconds=round(time.time() - start, 2),
        )

    # ── Hardware pre-check ──
    if not torch.cuda.is_available():
        return QuantResult(
            success=False,
            error="PyTorch CUDA is not available in this Python environment. "
                  "bitsandbytes requires PyTorch built with CUDA support. "
                  "Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121",
            method=config.method,
            duration_seconds=round(time.time() - start, 2),
        )

    trc = _trust_remote()
    out_dir = Path(config.output_dir) / (config.output_name or f"{_safe_name(model_id)}-{config.method}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        log.info("Loading model %s in %d-bit mode", model_id, bits)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=trc,
        )

        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # ── VRAM-aware loading: set max_memory + CPU offload ──
        max_memory = None
        offload_folder = None
        try:
            free_vram, total_vram = torch.cuda.mem_get_info(0)
            # Reserve 512 MB headroom on GPU
            usable_vram = max(0, free_vram - 512 * 1024 * 1024)
            import psutil as _ps
            usable_ram = max(0, _ps.virtual_memory().available - 1024 * 1024 * 1024)  # 1 GB headroom
            max_memory = {
                0: f"{usable_vram // (1024**2)}MiB",
                "cpu": f"{usable_ram // (1024**2)}MiB",
            }
            offload_folder = str(Path(config.output_dir) / "_offload")
            Path(offload_folder).mkdir(parents=True, exist_ok=True)
            log.info("VRAM-aware loading: GPU0=%s, CPU=%s, offload=%s",
                     max_memory[0], max_memory["cpu"], offload_folder)
        except Exception as hw_err:
            log.warning("Could not detect VRAM/RAM for max_memory: %s (falling back to auto)", hw_err)

        load_kwargs: dict = {
            "quantization_config": bnb_config,
            "token": os.environ.get("HF_TOKEN"),
            "trust_remote_code": trc,
            "device_map": "auto",
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            load_kwargs["offload_folder"] = offload_folder

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())

        return QuantResult(
            output_model=str(out_dir.name),
            output_path=str(out_dir),
            method=config.method,
            size_bytes=size,
            success=True,
            duration_seconds=round(time.time() - start, 2),
        )
    except Exception as e:
        log.exception("bitsandbytes quantization failed")
        err_str = str(e).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            return QuantResult(
                success=False,
                error=f"GPU memory insufficient for {bits}-bit loading of {model_id}: {e}",
                method=config.method,
                duration_seconds=round(time.time() - start, 2),
            )
        return QuantResult(success=False, error=str(e), method=config.method,
                           duration_seconds=round(time.time() - start, 2))


async def quantize_bnb(config: QuantConfig) -> QuantResult:
    """Quantize via bitsandbytes — runs in thread pool."""
    bits = 4 if config.method == QuantMethod.BNB_4BIT else 8
    return await asyncio.to_thread(_quantize_bnb_sync, config, bits)


async def quantize(
    config: QuantConfig,
    ollama_client: "OllamaClient | None" = None,  # type: ignore[name-defined]
) -> QuantResult:
    """Dispatch quantization based on method."""
    if config.method == QuantMethod.OLLAMA_GGUF:
        if ollama_client is None:
            return QuantResult(success=False, error="Ollama client required for GGUF quantization")
        return await quantize_ollama(config, ollama_client)
    elif config.method in (QuantMethod.BNB_4BIT, QuantMethod.BNB_8BIT):
        return await quantize_bnb(config)
    elif config.method == QuantMethod.AWQ:
        return await quantize_awq(config)
    elif config.method == QuantMethod.GPTQ:
        return await quantize_gptq(config)
    else:
        return QuantResult(success=False, error=f"Unknown method: {config.method}")


# ── AWQ quantization ────────────────────────────────────────────


def _quantize_awq_sync(config: QuantConfig) -> QuantResult:
    """AWQ (Activation-aware Weight Quantization) — state-of-the-art 4-bit."""
    start = time.time()

    # Resolve Ollama tags to HF repo IDs
    model_id = _resolve_hf_repo(config.source_model)
    if ":" in model_id:
        return QuantResult(
            success=False,
            error=f"Cannot use Ollama tag '{config.source_model}' with AWQ. "
                  f"Provide a HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-14B-Instruct') "
                  f"or add the model to registry.yaml with its hf_repo field.",
            method=QuantMethod.AWQ,
            duration_seconds=round(time.time() - start, 2),
        )

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        return QuantResult(
            success=False,
            error=f"AWQ requires: pip install autoawq. {e}",
            method=QuantMethod.AWQ,
            duration_seconds=round(time.time() - start, 2),
        )

    # ── RAM pre-check: AWQ loads full FP16 model into RAM before quantizing ──
    import platform as _platform
    try:
        import psutil as _ps
        avail_mb = _ps.virtual_memory().available // (1024 * 1024)
        # Conservative threshold: need at least 8 GB free for small models
        if avail_mb < 8000:
            hint = ""
            if _platform.system() == "Windows":
                hint = " On Windows, also increase the system pagefile size."
            return QuantResult(
                success=False,
                error=f"Insufficient RAM for AWQ ({avail_mb} MB available). "
                      f"AWQ loads the full FP16 model into memory before quantizing, "
                      f"requiring ~2x the model weight size.{hint}",
                method=QuantMethod.AWQ,
                duration_seconds=round(time.time() - start, 2),
            )
    except Exception:
        pass  # psutil not available — proceed and let it crash naturally

    trc = _trust_remote()
    out_dir = Path(config.output_dir) / (
        config.output_name or f"{_safe_name(model_id)}-awq-w{config.bits}g{config.group_size}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        log.info("AWQ quantizing %s to %d-bit (group_size=%d)", model_id, config.bits, config.group_size)
        model = AutoAWQForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trc,
            token=os.environ.get("HF_TOKEN"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trc,
            token=os.environ.get("HF_TOKEN"),
        )

        quant_config = {
            "zero_point": True,
            "q_group_size": config.group_size,
            "w_bit": config.bits,
            "version": "GEMM",
        }
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())

        return QuantResult(
            output_model=str(out_dir.name),
            output_path=str(out_dir),
            method=QuantMethod.AWQ,
            size_bytes=size,
            success=True,
            duration_seconds=round(time.time() - start, 2),
        )
    except Exception as e:
        log.exception("AWQ quantization failed")
        err_str = str(e).lower()
        if "1455" in err_str or "paging" in err_str or "not enough memory" in err_str:
            return QuantResult(
                success=False,
                error=f"System memory exhausted during AWQ quantization (Windows paging file too small). "
                      f"Increase the pagefile or free RAM before retrying. Original error: {e}",
                method=QuantMethod.AWQ,
                duration_seconds=round(time.time() - start, 2),
            )
        return QuantResult(success=False, error=str(e), method=QuantMethod.AWQ,
                           duration_seconds=round(time.time() - start, 2))


async def quantize_awq(config: QuantConfig) -> QuantResult:
    """AWQ quantization — runs in thread pool."""
    return await asyncio.to_thread(_quantize_awq_sync, config)


# ── GPTQ quantization ───────────────────────────────────────────


def _quantize_gptq_sync(config: QuantConfig) -> QuantResult:
    """GPTQ quantization — calibration-based post-training quantization."""
    start = time.time()

    # Resolve Ollama tags to HF repo IDs
    model_id = _resolve_hf_repo(config.source_model)
    if ":" in model_id:
        return QuantResult(
            success=False,
            error=f"Cannot use Ollama tag '{config.source_model}' with GPTQ. "
                  f"Provide a HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-14B-Instruct') "
                  f"or add the model to registry.yaml with its hf_repo field.",
            method=QuantMethod.GPTQ,
            duration_seconds=round(time.time() - start, 2),
        )

    try:
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  GPTQConfig)
    except ImportError as e:
        return QuantResult(
            success=False,
            error=f"GPTQ requires: pip install optimum auto-gptq. {e}",
            method=QuantMethod.GPTQ,
            duration_seconds=round(time.time() - start, 2),
        )

    trc = _trust_remote()
    out_dir = Path(config.output_dir) / (
        config.output_name or f"{_safe_name(model_id)}-gptq-{config.bits}bit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        log.info("GPTQ quantizing %s to %d-bit (group_size=%d)", model_id, config.bits, config.group_size)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trc,
            token=os.environ.get("HF_TOKEN"),
        )

        gptq_config = GPTQConfig(
            bits=config.bits,
            group_size=config.group_size,
            dataset=config.calibration_dataset,
            tokenizer=tokenizer,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=gptq_config,
            trust_remote_code=trc,
            token=os.environ.get("HF_TOKEN"),
            device_map="auto",
        )

        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())

        return QuantResult(
            output_model=str(out_dir.name),
            output_path=str(out_dir),
            method=QuantMethod.GPTQ,
            size_bytes=size,
            success=True,
            duration_seconds=round(time.time() - start, 2),
        )
    except Exception as e:
        log.exception("GPTQ quantization failed")
        return QuantResult(success=False, error=str(e), method=QuantMethod.GPTQ,
                           duration_seconds=round(time.time() - start, 2))


async def quantize_gptq(config: QuantConfig) -> QuantResult:
    """GPTQ quantization — runs in thread pool."""
    return await asyncio.to_thread(_quantize_gptq_sync, config)


# ── Comparison flow ──────────────────────────────────────────────


def _classify_error(error: str | None, method_name: str, availability_status: dict) -> str:
    """Classify an error into a status category for UI display."""
    if error is None:
        return "ok"
    err = error.lower()
    # Pre-flight status from availability check
    avail_status = availability_status.get("status", "")
    if avail_status in ("dependency_missing", "service_unavailable", "unsupported_hardware"):
        return avail_status
    # Hardware / memory failures
    if any(k in err for k in ("out of memory", "cuda", "vram", "gpu memory", "no cuda")):
        return "unsupported_hardware"
    if any(k in err for k in ("1455", "paging", "not enough memory", "insufficient ram")):
        return "unsupported_hardware"
    # Service failures
    if any(k in err for k in ("ollama is not running", "connection refused", "service unavailable")):
        return "service_unavailable"
    # Model / config errors
    if any(k in err for k in ("cannot use ollama tag", "no huggingface mapping", "registry")):
        return "model_error"
    if any(k in err for k in ("calibration", "dataset", "config")):
        return "config_error"
    return "config_error"  # fallback for unknown errors


async def compare_quantization_methods(
    source_model: str,
    methods: list[str] | None = None,
    reference_size_bytes: int = 0,
    ollama_client: "OllamaClient | None" = None,  # type: ignore[name-defined]
    output_dir: str = "data/cache/quantized",
) -> QuantComparisonResult:
    """Run multiple quantization methods on the same model and compare results.

    Pre-checks each method for required dependencies/services before
    attempting quantization.  Unavailable methods are reported with
    actionable install instructions instead of raw tracebacks.
    """
    if methods is None:
        methods = ["ollama_gguf", "bnb_4bit", "awq", "gptq"]

    availability = await check_method_availability()
    entries: list[QuantComparisonEntry] = []

    for method_name in methods:
        # ── Unknown method guard ──
        try:
            method = QuantMethod(method_name)
        except ValueError:
            entries.append(QuantComparisonEntry(
                method=method_name, success=False,
                error=f"Unknown method: {method_name}",
                status="config_error",
            ))
            continue

        # ── Pre-flight: skip if deps/services/hardware missing ──
        status = availability.get(method_name, {})
        if not status.get("available", True):
            entries.append(QuantComparisonEntry(
                method=method_name,
                success=False,
                error=f"{status.get('reason', 'Unavailable')}. {status.get('install', '')}",
                status=status.get("status", "config_error"),
            ))
            continue

        config = QuantConfig(
            source_model=source_model,
            method=method,
            output_dir=output_dir,
        )

        result = await quantize(config, ollama_client=ollama_client)

        reduction = 0.0
        if reference_size_bytes > 0 and result.size_bytes > 0:
            reduction = round((1 - result.size_bytes / reference_size_bytes) * 100, 1)

        error_status = "ok" if result.success else _classify_error(result.error, method_name, status)

        entries.append(QuantComparisonEntry(
            method=method_name,
            quant_level=config.quant_level if method == QuantMethod.OLLAMA_GGUF else f"{config.bits}bit",
            size_bytes=result.size_bytes,
            size_reduction_pct=reduction,
            duration_seconds=result.duration_seconds,
            success=result.success,
            error=result.error,
            output_path=result.output_path,
            status=error_status,
        ))

    # Determine best by size and speed among successful entries
    successful = [e for e in entries if e.success and e.size_bytes > 0]
    best_size = min(successful, key=lambda e: e.size_bytes).method if successful else None
    best_speed = min(successful, key=lambda e: e.duration_seconds).method if successful else None

    return QuantComparisonResult(
        source_model=source_model,
        entries=entries,
        best_size=best_size,
        best_speed=best_speed,
    )
