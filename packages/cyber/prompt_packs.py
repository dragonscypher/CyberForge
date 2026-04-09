"""Prompt packs — YAML-based system prompt templates for the cyber copilot.

Each pack is a YAML file in packages/cyber/prompts/ with string keys mapping
to system prompt templates.  Three built-in packs ship with CyberForge:

- analyst_pack.yaml — threat analysis and triage
- detection_pack.yaml — rule generation (Sigma/YARA/Suricata)
- report_pack.yaml — incident report drafting
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# In-memory cache: pack_name → {key: prompt_string}
_pack_cache: dict[str, dict[str, str]] = {}

# ── Built-in defaults (used when YAML files are absent) ─────────

_DEFAULT_PACKS: dict[str, dict[str, str]] = {
    "analyst": {
        "analyst_system": (
            "You are a senior cyber-security analyst. Summarise events, "
            "highlight IOCs, classify severity (critical/high/medium/low/info), "
            "and recommend next steps. Be precise and terse."
        ),
        "triage_system": (
            "You are a SOC triage analyst. Given raw log events, identify "
            "potential threats, rank by severity, and suggest escalation paths."
        ),
    },
    "detection": {
        "sigma_system": (
            "You are a detection engineer. Generate valid Sigma rules from "
            "the provided threat description. Include title, logsource, "
            "detection (selection + condition), level, and tags."
        ),
        "yara_system": (
            "You are a malware analyst. Generate valid YARA rules that detect "
            "the described indicators. Include descriptive meta, accurate strings, "
            "and efficient conditions."
        ),
        "suricata_system": (
            "You are a network security engineer. Generate Suricata IDS rules "
            "for the described traffic pattern. Use proper protocol, content "
            "matches, and assign an appropriate SID and severity."
        ),
    },
    "report": {
        "incident_system": (
            "You are a cyber incident response lead. Draft a structured incident "
            "report covering: executive summary, timeline, affected systems, "
            "indicators of compromise, root cause, remediation actions, and lessons learned."
        ),
        "executive_system": (
            "You are a CISO advisor. Summarise the security findings for an "
            "executive audience. Use business impact language, avoid jargon, "
            "and include risk ratings and recommended budget priorities."
        ),
    },
}


def load_prompt_pack(pack_name: str) -> dict[str, str]:
    """Load a prompt pack by name, with caching.

    Resolution order:
    1. In-memory cache
    2. YAML file in packages/cyber/prompts/{pack_name}_pack.yaml
    3. Built-in default dict
    4. Empty dict
    """
    if pack_name in _pack_cache:
        return _pack_cache[pack_name]

    pack_file = _PROMPT_DIR / f"{pack_name}_pack.yaml"
    if pack_file.exists():
        try:
            data = yaml.safe_load(pack_file.read_text(encoding="utf-8")) or {}
            _pack_cache[pack_name] = data
            return data
        except Exception as e:
            log.warning("Failed to load prompt pack %s: %s", pack_file, e)

    # Fall back to built-in
    defaults = _DEFAULT_PACKS.get(pack_name, {})
    _pack_cache[pack_name] = defaults
    return defaults


def list_packs() -> list[dict[str, Any]]:
    """List all available prompt packs."""
    packs: list[dict[str, Any]] = []

    # Scan YAML files
    seen: set[str] = set()
    if _PROMPT_DIR.exists():
        for f in _PROMPT_DIR.glob("*_pack.yaml"):
            name = f.stem.replace("_pack", "")
            seen.add(name)
            data = load_prompt_pack(name)
            packs.append({"name": name, "keys": list(data.keys()), "source": "yaml"})

    # Add built-in defaults not already found on disk
    for name, data in _DEFAULT_PACKS.items():
        if name not in seen:
            packs.append({"name": name, "keys": list(data.keys()), "source": "builtin"})

    return packs


def get_prompt(pack_name: str, key: str) -> str:
    """Get a specific prompt string from a pack."""
    pack = load_prompt_pack(pack_name)
    return pack.get(key, "")


def clear_cache():
    """Clear the prompt pack cache (useful for reloading from disk)."""
    _pack_cache.clear()
