"""vLLM inference backend — optional high-throughput serving.

Requires: pip install -e ".[serve]"
Only available when vLLM is installed and a vLLM server is running.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from packages.serve.base import (ChatMessage, ChatRequest, ChatResponse,
                                 InferenceBackend)

log = logging.getLogger(__name__)

_DEFAULT_BASE = "http://localhost:8000"


class VLLMClient(InferenceBackend):
    """Wraps a running vLLM OpenAI-compatible API server."""

    def __init__(self, base_url: str = _DEFAULT_BASE, timeout: float = 120.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self._base}/v1/models")
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self._base}/v1/models")
            r.raise_for_status()
            data = r.json()
        return [m["id"] for m in data.get("data", [])]

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self._base}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})

        return ChatResponse(
            model=data.get("model", request.model),
            message=ChatMessage(
                role=msg.get("role", "assistant"),
                content=msg.get("content", ""),
            ),
            done=True,
            prompt_eval_count=usage.get("prompt_tokens"),
            eval_count=usage.get("completion_tokens"),
        )

    async def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Raw text completion via vLLM's completions endpoint."""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self._base}/v1/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        return data.get("choices", [{}])[0].get("text", "")
