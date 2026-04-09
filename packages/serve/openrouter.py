"""OpenRouter BYOK client — OpenAI-compatible inference via Bearer token."""

from __future__ import annotations

import os
from typing import Optional

import httpx
from pydantic import BaseModel

from packages.serve.base import (ChatMessage, ChatRequest, ChatResponse,
                                 InferenceBackend)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterClient(InferenceBackend):
    """Sends chat requests to OpenRouter using the OpenAI-like chat/completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _OPENROUTER_BASE,
        timeout: float = 120.0,
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # ── discovery ────────────────────────────────────────────────
    async def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(
                    f"{self._base}/models",
                    headers=self._headers(),
                )
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self._base}/models", headers=self._headers())
            r.raise_for_status()
            data = r.json()
        return [m["id"] for m in data.get("data", [])]

    # ── inference ────────────────────────────────────────────────
    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(
                f"{self._base}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})
        return ChatResponse(
            model=data.get("model", request.model),
            message=ChatMessage(role=msg.get("role", "assistant"), content=msg.get("content", "")),
            done=True,
            prompt_eval_count=usage.get("prompt_tokens"),
            eval_count=usage.get("completion_tokens"),
        )

    # ── helpers ──────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
