"""Hardware profiler — detects CPU, RAM, GPU, CUDA, disk, and backend availability."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import re
import shutil
import subprocess
from typing import Optional

import psutil
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class GPUInfo(BaseModel):
    index: int = 0
    name: str = "unknown"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None


class BackendStatus(BaseModel):
    ollama: bool = False
    pytorch_cuda: bool = False
    bitsandbytes: bool = False
    vllm: bool = False
    tensorrt_llm: bool = False
    transformers: bool = False
    peft: bool = False


class HardwareProfile(BaseModel):
    os: str = ""
    os_version: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    gpus: list[GPUInfo] = Field(default_factory=list)
    disk_free_mb: int = 0
    disk_path: str = ""
    backends: BackendStatus = Field(default_factory=BackendStatus)


class HardwareProfiler:
    """Collects a full snapshot of the local system hardware and software backends.

    Results are cached with a TTL to avoid redundant subprocess/HTTP probes
    when multiple UI sections fetch capabilities on the same page load.
    Hardware (GPU, CPU, RAM) changes slowly — 30s TTL.
    Services (Ollama) change faster — 5s TTL for those fields only.
    """

    HARDWARE_TTL_S: float = 30.0
    SERVICE_TTL_S: float = 5.0

    def __init__(self, disk_path: str = "."):
        self._disk_path = disk_path
        self._cached_profile: HardwareProfile | None = None
        self._hw_timestamp: float = 0.0
        self._svc_timestamp: float = 0.0

    def profile(self, *, force: bool = False) -> HardwareProfile:
        import time as _time
        now = _time.monotonic()
        hw_stale = force or (now - self._hw_timestamp > self.HARDWARE_TTL_S)
        svc_stale = force or (now - self._svc_timestamp > self.SERVICE_TTL_S)

        if hw_stale or self._cached_profile is None:
            # Full re-scan
            p = HardwareProfile()
            self._detect_os(p)
            self._detect_cpu(p)
            self._detect_ram(p)
            self._detect_disk(p)
            self._detect_gpus(p)
            self._detect_backends(p)
            self._cached_profile = p
            done = _time.monotonic()
            self._hw_timestamp = done
            self._svc_timestamp = done
        elif svc_stale:
            # Only refresh volatile service checks (Ollama, RAM avail)
            p = self._cached_profile.model_copy(deep=True)
            p.backends.ollama = self._check_ollama()
            mem = psutil.virtual_memory()
            p.ram_available_mb = mem.available // (1024 * 1024)
            self._cached_profile = p
            self._svc_timestamp = _time.monotonic()

        return self._cached_profile

    def invalidate(self) -> None:
        """Force next profile() call to do a full re-scan."""
        self._hw_timestamp = 0.0
        self._svc_timestamp = 0.0

    # ------------------------------------------------------------------
    def _detect_os(self, p: HardwareProfile) -> None:
        p.os = platform.system()
        p.os_version = platform.version()

    def _detect_cpu(self, p: HardwareProfile) -> None:
        p.cpu_cores = psutil.cpu_count(logical=False) or 1
        p.cpu_threads = psutil.cpu_count(logical=True) or 1
        p.cpu_model = platform.processor() or "unknown"
        # Try to get a friendlier name on Windows
        if p.os == "Windows":
            try:
                out = subprocess.check_output(
                    ["wmic", "cpu", "get", "Name"],
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"]
                if lines:
                    p.cpu_model = lines[0]
            except Exception:
                pass

    def _detect_ram(self, p: HardwareProfile) -> None:
        mem = psutil.virtual_memory()
        p.ram_total_mb = mem.total // (1024 * 1024)
        p.ram_available_mb = mem.available // (1024 * 1024)

    def _detect_disk(self, p: HardwareProfile) -> None:
        p.disk_path = os.path.abspath(self._disk_path)
        usage = shutil.disk_usage(p.disk_path)
        p.disk_free_mb = usage.free // (1024 * 1024)

    # ------------------------------------------------------------------
    def _detect_gpus(self, p: HardwareProfile) -> None:
        """Use nvidia-smi to enumerate GPUs. Falls back gracefully."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
            for line in out.strip().splitlines():
                parts = [s.strip() for s in line.split(",")]
                if len(parts) < 5:
                    continue
                gpu = GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(float(parts[2])),
                    vram_free_mb=int(float(parts[3])),
                    driver_version=parts[4],
                )
                p.gpus.append(gpu)
        except FileNotFoundError:
            log.debug("nvidia-smi not found on PATH — no NVIDIA GPU detection")
        except subprocess.TimeoutExpired:
            log.warning("nvidia-smi timed out — GPU detection incomplete")
        except Exception as exc:
            log.debug("GPU detection failed: %s", exc)

        # Attempt CUDA version from nvcc or nvidia-smi
        cuda_ver = self._detect_cuda_version()
        for gpu in p.gpus:
            gpu.cuda_version = cuda_ver

    @staticmethod
    def _detect_cuda_version() -> Optional[str]:
        # Try nvcc first
        try:
            out = subprocess.check_output(
                ["nvcc", "--version"],
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
            m = re.search(r"release (\d+\.\d+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
        # Fallback: nvidia-smi header
        try:
            out = subprocess.check_output(
                ["nvidia-smi"],
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
            m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def _detect_backends(self, p: HardwareProfile) -> None:
        p.backends.ollama = self._check_ollama()
        p.backends.pytorch_cuda = self._check_pytorch_cuda()
        p.backends.transformers = self._can_import("transformers")
        p.backends.peft = self._can_import("peft")
        p.backends.bitsandbytes = self._can_import("bitsandbytes")
        p.backends.vllm = self._can_import("vllm")
        p.backends.tensorrt_llm = self._can_import("tensorrt_llm")

    @staticmethod
    def _ollama_url() -> str:
        """Resolve Ollama base URL from OLLAMA_HOST env or default."""
        host = os.environ.get("OLLAMA_HOST", "").strip()
        if host:
            if not host.startswith("http"):
                host = f"http://{host}"
            return host.rstrip("/")
        return "http://localhost:11434"

    @classmethod
    def _check_ollama(cls) -> bool:
        """Return True if Ollama HTTP API is reachable."""
        import requests

        url = cls._ollama_url()
        try:
            r = requests.get(f"{url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            log.debug("Ollama not reachable at %s", url)
            return False

    @staticmethod
    def _check_pytorch_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _can_import(module: str) -> bool:
        try:
            importlib.import_module(module)
            return True
        except Exception:
            return False
