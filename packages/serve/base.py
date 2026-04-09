"""Abstract base for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool = True
    total_duration_ms: Optional[float] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class InferenceBackend(ABC):
    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    async def list_models(self) -> list[str]: ...
