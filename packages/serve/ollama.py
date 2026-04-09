"""Ollama client — local model discovery, inference, import, and quantization."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from packages.serve.base import (ChatMessage, ChatRequest, ChatResponse,
                                 InferenceBackend)

_DEFAULT_BASE = "http://localhost:11434"


class OllamaModelInfo(BaseModel):
    name: str
    size: int = 0  # bytes
    parameter_size: str = ""
    quantization_level: str = ""
    family: str = ""
    format: str = ""
    digest: str = ""
    modified_at: str = ""


class OllamaClient(InferenceBackend):
    """Wraps the Ollama HTTP API for model discovery, chat, pull, and create."""

    def __init__(self, base_url: str = _DEFAULT_BASE, timeout: float = 120.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # ── discovery ────────────────────────────────────────────────
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self._base}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        infos = await self.list_model_info()
        return [m.name for m in infos]

    async def list_model_info(self) -> list[OllamaModelInfo]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self._base}/api/tags")
            r.raise_for_status()
            data = r.json()
        models = []
        for m in data.get("models", []):
            details = m.get("details", {})
            models.append(
                OllamaModelInfo(
                    name=m.get("name", ""),
                    size=m.get("size", 0),
                    parameter_size=details.get("parameter_size", ""),
                    quantization_level=details.get("quantization_level", ""),
                    family=details.get("family", ""),
                    format=details.get("format", ""),
                    digest=m.get("digest", ""),
                    modified_at=m.get("modified_at", ""),
                )
            )
        return models

    async def list_running(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self._base}/api/ps")
            r.raise_for_status()
            return r.json().get("models", [])

    # ── inference ────────────────────────────────────────────────
    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        # Chat needs extra time on first call (model loading into VRAM)
        async with httpx.AsyncClient(timeout=max(self._timeout, 300)) as c:
            r = await c.post(f"{self._base}/api/chat", json=payload)
            if r.status_code != 200:
                try:
                    body = r.json()
                    err_msg = body.get("error", r.text)
                except Exception:
                    err_msg = r.text
                raise RuntimeError(f"Ollama chat failed ({r.status_code}): {err_msg}")
            data = r.json()
        msg = data.get("message", {})
        return ChatResponse(
            model=request.model,
            message=ChatMessage(role=msg.get("role", "assistant"), content=msg.get("content", "")),
            done=data.get("done", True),
            total_duration_ms=(data.get("total_duration", 0)) / 1e6,
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )

    async def generate(self, model: str, prompt: str, **kwargs: Any) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False, **kwargs}
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self._base}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")

    # ── model management ─────────────────────────────────────────
    async def pull(self, model_name: str, insecure: bool = False) -> dict[str, Any]:
        payload = {"name": model_name, "insecure": insecure, "stream": False}
        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{self._base}/api/pull", json=payload)
            if r.status_code != 200:
                try:
                    body = r.json()
                    err_msg = body.get("error", r.text)
                except Exception:
                    err_msg = r.text
                raise RuntimeError(f"Ollama pull failed ({r.status_code}): {err_msg}")
            return r.json()

    async def create(
        self,
        model_name: str,
        *,
        from_model: str | None = None,
        quantize: str | None = None,
        system: str | None = None,
        modelfile: str | None = None,
    ) -> dict[str, Any]:
        """Create a model via Ollama /api/create.

        Supports the new JSON API (``from`` / ``quantize``) and the legacy
        ``modelfile`` string approach as fallback.
        """
        payload: dict[str, Any] = {"model": model_name, "stream": False}
        if from_model:
            payload["from"] = from_model
        if quantize:
            payload["quantize"] = quantize
        if system:
            payload["system"] = system
        if modelfile and not from_model:
            # Legacy path – only used when callers still pass a raw Modelfile
            payload["modelfile"] = modelfile
        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{self._base}/api/create", json=payload)
            if r.status_code != 200:
                # Capture Ollama's actual error message instead of generic HTTP error
                try:
                    body = r.json()
                    err_msg = body.get("error", r.text)
                except Exception:
                    err_msg = r.text
                raise RuntimeError(f"Ollama create failed ({r.status_code}): {err_msg}")
            return r.json()

    async def delete(self, model_name: str) -> bool:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.request("DELETE", f"{self._base}/api/delete", json={"name": model_name})
            return r.status_code == 200

    async def show(self, model_name: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self._base}/api/show", json={"name": model_name})
            r.raise_for_status()
            return r.json()
