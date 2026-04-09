"""Cyber copilot — LLM-powered security analysis assistant.

Provides programmatic access to prompt-pack-driven chat for:
- Threat analysis and triage
- Detection rule generation (Sigma, YARA, Suricata)
- Incident report drafting
- ATT&CK mapping suggestions
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


class CopilotMessage(BaseModel):
    role: str  # user | assistant | system
    content: str


class CopilotRequest(BaseModel):
    model_id: str = ""
    messages: list[CopilotMessage] = []
    task_mode: str = "cyber"
    prompt_pack: str = "analyst"
    prompt_key: str = "analyst_system"
    return_artifact_candidates: bool = False


class CopilotResponse(BaseModel):
    reply: str = ""
    artifacts: list[dict[str, str]] = []
    success: bool = True
    error: Optional[str] = None


class CyberCopilot:
    """Orchestrates prompt-pack-driven chat with an LLM backend."""

    def __init__(self, ollama_client=None):
        self._ollama = ollama_client

    def set_client(self, ollama_client):
        self._ollama = ollama_client

    async def chat(self, request: CopilotRequest) -> CopilotResponse:
        """Send a chat request through the copilot pipeline."""
        if self._ollama is None:
            return CopilotResponse(success=False, error="Ollama client not configured")

        from packages.cyber.prompt_packs import load_prompt_pack

        pack = load_prompt_pack(request.prompt_pack)
        system_prompt = pack.get(request.prompt_key, pack.get("analyst_system", ""))

        from packages.serve.base import ChatMessage
        from packages.serve.base import ChatRequest as BaseChatRequest

        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        for m in request.messages:
            messages.append(ChatMessage(role=m.role, content=m.content))

        model = request.model_id
        if not model:
            try:
                available = await self._ollama.list_models()
                model = available[0] if available else ""
            except Exception:
                pass

        if not model:
            return CopilotResponse(success=False, error="No model specified and no models available")

        try:
            chat_req = BaseChatRequest(model=model, messages=messages)
            response = await self._ollama.chat(chat_req)
            reply = response.message.content

            artifacts: list[dict[str, str]] = []
            if request.return_artifact_candidates:
                artifacts = extract_artifacts(reply)

            return CopilotResponse(reply=reply, artifacts=artifacts, success=True)
        except Exception as e:
            log.exception("Copilot chat failed")
            return CopilotResponse(success=False, error=str(e))


def extract_artifacts(text: str) -> list[dict[str, str]]:
    """Extract potential Sigma/YARA/Suricata rule artifacts from LLM output."""
    artifacts: list[dict[str, str]] = []

    if "title:" in text and "detection:" in text:
        artifacts.append({"type": "sigma", "content": text})

    if "rule " in text and "condition:" in text:
        artifacts.append({"type": "yara", "content": text})

    if ("alert " in text or "drop " in text) and "sid:" in text:
        artifacts.append({"type": "suricata", "content": text})

    # Extract ATT&CK technique IDs
    attack_ids = re.findall(r"T\d{4}(?:\.\d{3})?", text)
    if attack_ids:
        artifacts.append({"type": "attack_techniques", "content": ", ".join(set(attack_ids))})

    return artifacts
