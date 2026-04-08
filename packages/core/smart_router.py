"""Smart backend router — routes optimization tasks to the best available backend.

Inspired by openclaude's smart_router.py: health-check all backends, score by
latency/availability/quality, auto-fallback if preferred backend fails.

CyberForge backends: ollama_gguf, bnb_8bit, bnb_4bit, awq, gptq, cpu_pytorch.

Ticket: CORE-004
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class Backend:
    """A quantization/optimization backend with health tracking."""
    name: str
    display_name: str
    check_fn: str              # Async function name for health check
    cost_score: float          # 0.0 (free/fast) to 1.0 (expensive/slow)
    quality_score: float       # 0.0 (lossy) to 1.0 (near-lossless)
    healthy: bool = False
    latency_ms: float = 0.0
    error_count: int = 0
    request_count: int = 0
    last_error: str = ""
    supports_quant: bool = True
    supports_prune: bool = False
    supports_distill: bool = False
    supports_edit: bool = False
    min_vram_mb: int = 0       # Minimum VRAM needed for this backend


@dataclass
class RouteDecision:
    """Result of routing: which backend to use and why."""
    backend: str
    reason: str
    fallback_chain: list[str] = field(default_factory=list)
    score: float = 0.0
    latency_ms: float = 0.0


class SmartRouter:
    """Routes optimization tasks to the best available backend.

    Strategies:
      - balanced: weighted combination of quality, cost, and health
      - quality: prefer highest quality backend
      - speed: prefer fastest / lowest-cost backend
      - fallback: try preferred, fall back through chain
    """

    def __init__(
        self,
        strategy: str = "balanced",
        fallback_enabled: bool = True,
    ):
        self.strategy = strategy
        self.fallback_enabled = fallback_enabled
        self._backends: dict[str, Backend] = {}
        self._initialized = False
        self._init_backends()

    def _init_backends(self):
        """Register all known backends."""
        self._backends = {
            "ollama_gguf": Backend(
                name="ollama_gguf",
                display_name="Ollama GGUF",
                check_fn="_check_ollama",
                cost_score=0.1,   # Free, local
                quality_score=0.7,  # Q4/Q8 quality
                supports_quant=True,
                min_vram_mb=0,    # CPU fallback available
            ),
            "bnb_8bit": Backend(
                name="bnb_8bit",
                display_name="bitsandbytes 8-bit",
                check_fn="_check_bnb",
                cost_score=0.2,
                quality_score=0.85,
                supports_quant=True,
                min_vram_mb=2000,
            ),
            "bnb_4bit": Backend(
                name="bnb_4bit",
                display_name="bitsandbytes 4-bit (NF4)",
                check_fn="_check_bnb",
                cost_score=0.15,
                quality_score=0.75,
                supports_quant=True,
                min_vram_mb=1500,
            ),
            "awq": Backend(
                name="awq",
                display_name="AWQ (Activation-aware Weight Quantization)",
                check_fn="_check_awq",
                cost_score=0.4,    # Needs calibration
                quality_score=0.9,
                supports_quant=True,
                min_vram_mb=4000,  # Calibration is VRAM-heavy
            ),
            "gptq": Backend(
                name="gptq",
                display_name="GPTQ (via Optimum)",
                check_fn="_check_gptq",
                cost_score=0.5,    # Slowest calibration
                quality_score=0.9,
                supports_quant=True,
                min_vram_mb=4000,
            ),
            "cpu_pytorch": Backend(
                name="cpu_pytorch",
                display_name="CPU PyTorch (Pruning/Editing/Distill)",
                check_fn="_check_pytorch",
                cost_score=0.3,
                quality_score=0.95,
                supports_quant=False,
                supports_prune=True,
                supports_distill=True,
                supports_edit=True,
                min_vram_mb=0,
            ),
        }

    async def initialize(self, vram_mb: int = 0) -> dict[str, Any]:
        """Health-check all backends and return status report."""
        start = time.monotonic()

        checks = [
            self._health_check(name, backend, vram_mb)
            for name, backend in self._backends.items()
        ]
        await asyncio.gather(*checks, return_exceptions=True)

        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000

        available = [b for b in self._backends.values() if b.healthy]
        log.info(
            "SmartRouter ready in %.0fms. Available: %s",
            elapsed, [b.name for b in available],
        )

        return self.status()

    async def _health_check(self, name: str, backend: Backend, vram_mb: int):
        """Check if a backend is available."""
        start = time.monotonic()

        # VRAM gate
        if backend.min_vram_mb > 0 and vram_mb > 0 and vram_mb < backend.min_vram_mb:
            backend.healthy = False
            backend.last_error = f"Needs {backend.min_vram_mb} MB VRAM, have {vram_mb} MB"
            return

        try:
            if name == "ollama_gguf":
                backend.healthy = await self._check_ollama()
            elif name in ("bnb_8bit", "bnb_4bit"):
                backend.healthy = self._check_bnb()
            elif name == "awq":
                backend.healthy = self._check_awq()
            elif name == "gptq":
                backend.healthy = self._check_gptq()
            elif name == "cpu_pytorch":
                backend.healthy = self._check_pytorch()
            else:
                backend.healthy = False

            backend.latency_ms = (time.monotonic() - start) * 1000

        except Exception as e:
            backend.healthy = False
            backend.last_error = str(e)
            backend.error_count += 1

    @staticmethod
    async def _check_ollama() -> bool:
        from urllib.parse import urlparse

        import httpx
        host = os.environ.get("OLLAMA_HOST", "").strip()
        url = host if host else "http://localhost:11434"
        if not url.startswith("http"):
            url = f"http://{url}"
        # SSRF protection: only allow localhost/127.0.0.1
        parsed = urlparse(url)
        allowed_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
        if parsed.hostname not in allowed_hosts:
            log.warning("OLLAMA_HOST blocked — only localhost allowed, got %s", parsed.hostname)
            return False
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{url.rstrip('/')}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _check_bnb() -> bool:
        import importlib.util
        return (
            importlib.util.find_spec("bitsandbytes") is not None
            and importlib.util.find_spec("torch") is not None
        )

    @staticmethod
    def _check_awq() -> bool:
        import importlib.util
        return importlib.util.find_spec("awq") is not None

    @staticmethod
    def _check_gptq() -> bool:
        import importlib.util
        return importlib.util.find_spec("optimum") is not None

    @staticmethod
    def _check_pytorch() -> bool:
        import importlib.util
        return importlib.util.find_spec("torch") is not None

    def route(
        self,
        task: str = "quantize",
        model_params_b: float = 0,
        vram_mb: int = 0,
        preferred: str = "",
    ) -> RouteDecision:
        """Route an optimization task to the best backend.

        Args:
            task: quantize | prune | distill | edit
            model_params_b: Model size in billions.
            vram_mb: Available VRAM.
            preferred: User-preferred backend (empty = auto).
        """
        # Filter backends by task capability
        candidates = []
        for b in self._backends.values():
            if not b.healthy:
                continue
            if task == "quantize" and not b.supports_quant:
                continue
            if task == "prune" and not b.supports_prune:
                continue
            if task == "distill" and not b.supports_distill:
                continue
            if task == "edit" and not b.supports_edit:
                continue
            # VRAM filter
            if b.min_vram_mb > 0 and vram_mb > 0 and vram_mb < b.min_vram_mb:
                continue
            candidates.append(b)

        if not candidates:
            return RouteDecision(
                backend="",
                reason=f"No healthy backends available for '{task}'",
                score=0.0,
            )

        # If user preferred is available, use it
        if preferred:
            for c in candidates:
                if c.name == preferred:
                    fallback = [b.name for b in candidates if b.name != preferred]
                    return RouteDecision(
                        backend=preferred,
                        reason=f"User preferred backend '{preferred}'",
                        fallback_chain=fallback,
                        score=1.0,
                        latency_ms=c.latency_ms,
                    )

        # Score candidates
        scored = []
        for c in candidates:
            if self.strategy == "quality":
                score = c.quality_score * 0.8 + (1 - c.cost_score) * 0.2
            elif self.strategy == "speed":
                score = (1 - c.cost_score) * 0.8 + c.quality_score * 0.2
            else:  # balanced
                health = 1.0 - (c.error_count / max(c.request_count, 1))
                score = (
                    c.quality_score * 0.4
                    + (1 - c.cost_score) * 0.3
                    + health * 0.3
                )
            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scored[0]
        fallback = [c.name for c, _ in scored[1:]]

        return RouteDecision(
            backend=best.name,
            reason=f"Best {self.strategy} score: quality={best.quality_score}, cost={best.cost_score}",
            fallback_chain=fallback if self.fallback_enabled else [],
            score=round(best_score, 3),
            latency_ms=best.latency_ms,
        )

    def record_result(self, backend: str, success: bool, error: str = ""):
        """Record a backend operation result for adaptive routing."""
        b = self._backends.get(backend)
        if b:
            b.request_count += 1
            if not success:
                b.error_count += 1
                b.last_error = error

    def status(self) -> dict[str, Any]:
        """Return current status of all backends."""
        return {
            "strategy": self.strategy,
            "fallback_enabled": self.fallback_enabled,
            "initialized": self._initialized,
            "backends": {
                name: {
                    "display_name": b.display_name,
                    "healthy": b.healthy,
                    "latency_ms": round(b.latency_ms, 1),
                    "quality_score": b.quality_score,
                    "cost_score": b.cost_score,
                    "error_count": b.error_count,
                    "request_count": b.request_count,
                    "last_error": b.last_error,
                    "supports": {
                        "quantize": b.supports_quant,
                        "prune": b.supports_prune,
                        "distill": b.supports_distill,
                        "edit": b.supports_edit,
                    },
                    "min_vram_mb": b.min_vram_mb,
                }
                for name, b in self._backends.items()
            },
        }
