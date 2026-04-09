"""Cyber router — log ingestion, copilot chat, validation, ATT&CK mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, Request
from pydantic import BaseModel

from packages.cyber.ingestion import NormalisedEvent, ingest

router = APIRouter()

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "packages" / "cyber" / "prompts"


# ── Log ingestion ────────────────────────────────────────────────


class IngestRequest(BaseModel):
    source_type: str  # sysmon_jsonl | suricata_eve | zeek_conn | windows_security
    path: str


class IngestResponse(BaseModel):
    events_count: int = 0
    events: list[dict[str, Any]] = []
    success: bool = True
    error: Optional[str] = None


@router.post("/logs/ingest", response_model=IngestResponse)
async def ingest_logs(req: IngestRequest):
    """Ingest security logs and return normalised events."""
    try:
        events = ingest(req.source_type, req.path)
        return IngestResponse(
            events_count=len(events),
            events=[e.model_dump() for e in events[:500]],  # Cap response to 500 events
            success=True,
        )
    except (FileNotFoundError, ValueError) as e:
        return IngestResponse(success=False, error=str(e))


# ── Copilot chat ─────────────────────────────────────────────────


class CopilotMessage(BaseModel):
    role: str  # user | assistant | system
    content: str


class CopilotRequest(BaseModel):
    model_id: str = ""  # Ollama model name
    messages: list[CopilotMessage]
    task_mode: str = "cyber"
    prompt_pack: str = "analyst"  # analyst | detection | report
    prompt_key: str = "analyst_system"  # key within the pack
    return_artifact_candidates: bool = False


class CopilotResponse(BaseModel):
    reply: str = ""
    artifacts: list[dict[str, str]] = []  # [{type: sigma|yara, content: ...}]
    success: bool = True
    error: Optional[str] = None


def _load_prompt_pack(pack_name: str) -> dict[str, str]:
    """Load a YAML prompt pack by name."""
    pack_file = _PROMPT_DIR / f"{pack_name}_pack.yaml"
    if not pack_file.exists():
        return {}
    return yaml.safe_load(pack_file.read_text(encoding="utf-8")) or {}


@router.post("/copilot/chat", response_model=CopilotResponse)
async def copilot_chat(req: CopilotRequest, request: Request):
    """Cyber copilot chat — uses prompt packs for system instructions."""
    ollama = getattr(request.app.state, "ollama", None)
    if ollama is None:
        return CopilotResponse(success=False, error="Ollama client not configured")

    # Load system prompt from pack
    pack = _load_prompt_pack(req.prompt_pack)
    system_prompt = pack.get(req.prompt_key, pack.get("analyst_system", ""))

    # Build messages
    from packages.serve.base import ChatMessage, ChatRequest

    messages = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    for m in req.messages:
        messages.append(ChatMessage(role=m.role, content=m.content))

    model = req.model_id
    if not model:
        # Try to get a default from installed models
        try:
            available = await ollama.list_models()
            model = available[0] if available else ""
        except Exception:
            pass

    if not model:
        return CopilotResponse(success=False, error="No model specified and no models available")

    try:
        chat_req = ChatRequest(model=model, messages=messages)
        response = await ollama.chat(chat_req)
        reply = response.message.content

        artifacts = []
        if req.return_artifact_candidates:
            artifacts = _extract_artifacts(reply)

        return CopilotResponse(reply=reply, artifacts=artifacts, success=True)
    except Exception as e:
        return CopilotResponse(success=False, error=str(e))


def _extract_artifacts(text: str) -> list[dict[str, str]]:
    """Extract potential Sigma/YARA/Suricata artifacts from LLM output."""
    artifacts = []

    # Check for Sigma rules (YAML with title: and detection:)
    if "title:" in text and "detection:" in text:
        artifacts.append({"type": "sigma", "content": text})

    # Check for YARA rules
    if "rule " in text and "condition:" in text:
        artifacts.append({"type": "yara", "content": text})

    # Check for Suricata rules
    if ("alert " in text or "drop " in text) and "sid:" in text:
        artifacts.append({"type": "suricata", "content": text})

    return artifacts


# ── Validation endpoints ─────────────────────────────────────────


class ValidateSigmaRequest(BaseModel):
    text: str
    target_backend: str = "splunk"


class ValidateYaraRequest(BaseModel):
    text: str


class ValidateResponse(BaseModel):
    valid: bool = False
    errors: list[str] = []
    warnings: list[str] = []


@router.post("/validate/sigma", response_model=ValidateResponse)
async def validate_sigma(req: ValidateSigmaRequest):
    """Validate a Sigma rule."""
    from packages.cyber.verifiers import SigmaVerifier
    v = SigmaVerifier()
    result = v.verify(req.text)
    return ValidateResponse(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
    )


@router.post("/validate/yara", response_model=ValidateResponse)
async def validate_yara(req: ValidateYaraRequest):
    """Validate a YARA rule by compiling it."""
    from packages.cyber.verifiers import YaraVerifier
    v = YaraVerifier()
    result = v.verify(req.text)
    return ValidateResponse(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
    )


# ── ATT&CK mapping ──────────────────────────────────────────────


class AttackMapRequest(BaseModel):
    artifact_text: str
    candidate_ids: list[str] = []


class AttackMapResponse(BaseModel):
    mapped_techniques: list[dict[str, str]] = []
    success: bool = True


@router.post("/map/attack", response_model=AttackMapResponse)
async def map_attack(req: AttackMapRequest):
    """Map artifact text to MITRE ATT&CK technique IDs (plan.md §5.7)."""
    import re

    from packages.cyber.verifiers import AttackVerifier

    verifier = AttackVerifier()

    # Extract technique IDs from the text
    found_ids = re.findall(r"T\d{4}(?:\.\d{3})?", req.artifact_text)

    # If candidate IDs are provided, validate those too
    all_ids = list(set(found_ids + req.candidate_ids))

    if not all_ids:
        return AttackMapResponse(
            mapped_techniques=[],
            success=True,
        )

    techniques = []
    for tid in all_ids:
        result = verifier.verify(tid)
        techniques.append({
            "technique_id": tid,
            "valid": str(result.valid),
            "source": "extracted" if tid in found_ids else "candidate",
        })

    return AttackMapResponse(mapped_techniques=techniques, success=True)


# ── Dataset info ─────────────────────────────────────────────────


@router.get("/datasets")
async def list_datasets():
    """List available cyber dataset loaders."""
    return {
        "datasets": [
            {"name": "NSL-KDD", "loader": "load_nsl_kdd", "status": "available"},
            {"name": "CICIDS2017", "loader": "load_cicids2017", "status": "available"},
            {"name": "CSE-CIC-IDS2018", "loader": "load_cse_cic_ids2018", "status": "available"},
            {"name": "UNSW-NB15", "loader": "load_unsw_nb15", "status": "available"},
        ]
    }
