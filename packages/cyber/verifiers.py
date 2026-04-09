"""Cyber verifiers — validate generated Sigma, YARA, Suricata, ATT&CK outputs.

Milestone 4 implementation. Skeleton defines the verifier interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class VerifyResult(BaseModel):
    valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    artifact_type: str = ""


class CyberVerifier(ABC):
    @abstractmethod
    def verify(self, content: str) -> VerifyResult: ...


class SigmaVerifier(CyberVerifier):
    """Validate Sigma rules against the Sigma specification/schema."""

    def verify(self, content: str) -> VerifyResult:
        import yaml as _yaml

        errors: list[str] = []
        try:
            doc = _yaml.safe_load(content)
        except Exception as e:
            return VerifyResult(valid=False, errors=[f"YAML parse error: {e}"], artifact_type="sigma")

        if not isinstance(doc, dict):
            errors.append("Sigma rule must be a YAML mapping")
        else:
            for field in ("title", "logsource", "detection"):
                if field not in doc:
                    errors.append(f"Missing required field: {field}")

        return VerifyResult(valid=len(errors) == 0, errors=errors, artifact_type="sigma")


class YaraVerifier(CyberVerifier):
    """Validate YARA rules — attempt compilation via yara-python if available."""

    def verify(self, content: str) -> VerifyResult:
        try:
            import yara
            yara.compile(source=content)
            return VerifyResult(valid=True, artifact_type="yara")
        except ImportError:
            # Fallback: basic syntax check
            errors = []
            if "rule " not in content:
                errors.append("Missing 'rule' keyword")
            if "condition:" not in content:
                errors.append("Missing 'condition:' section")
            return VerifyResult(valid=len(errors) == 0, errors=errors, artifact_type="yara")
        except Exception as e:
            return VerifyResult(valid=False, errors=[str(e)], artifact_type="yara")


class SuricataVerifier(CyberVerifier):
    """Basic Suricata rule syntax validation."""

    def verify(self, content: str) -> VerifyResult:
        errors: list[str] = []
        for i, line in enumerate(content.strip().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # A Suricata rule must start with an action keyword
            actions = ("alert", "pass", "drop", "reject", "rejectsrc", "rejectdst", "rejectboth")
            if not any(line.startswith(a) for a in actions):
                errors.append(f"Line {i}: must start with an action keyword")
            if "(" not in line or ")" not in line:
                errors.append(f"Line {i}: missing rule options in parentheses")
        return VerifyResult(valid=len(errors) == 0, errors=errors, artifact_type="suricata")


class AttackVerifier(CyberVerifier):
    """Validate MITRE ATT&CK technique IDs with optional cached STIX bundle lookup.

    If a STIX bundle JSON is available at ``data/cache/attack_stix_bundle.json``
    (or downloaded on first use), technique IDs are validated against the real
    ATT&CK knowledge base.  Otherwise falls back to format-only validation.
    """

    _STIX_CACHE_PATH = "data/cache/attack_stix_bundle.json"
    _STIX_URL = (
        "https://raw.githubusercontent.com/mitre/cti/master/"
        "enterprise-attack/enterprise-attack.json"
    )
    _technique_ids: set[str] | None = None

    def __init__(self) -> None:
        if AttackVerifier._technique_ids is None:
            AttackVerifier._technique_ids = self._load_stix_ids()

    # ── public ───────────────────────────────────────────────────

    def verify(self, content: str) -> VerifyResult:
        import re

        found = re.findall(r"T\d{4}(?:\.\d{3})?", content)
        if not found:
            return VerifyResult(
                valid=False,
                errors=["No ATT&CK technique IDs found"],
                artifact_type="attack",
            )

        known = self._technique_ids or set()
        errors: list[str] = []
        warnings: list[str] = []

        if not known:
            warnings.append("STIX bundle not available — format-only validation")
        for tid in set(found):
            if known and tid not in known:
                errors.append(f"{tid} not found in ATT&CK STIX bundle")

        valid = len(errors) == 0
        if not errors and known:
            warnings.append(f"All {len(set(found))} ID(s) validated against STIX bundle")

        return VerifyResult(valid=valid, errors=errors, warnings=warnings, artifact_type="attack")

    # ── STIX bundle helpers ──────────────────────────────────────

    @classmethod
    def _load_stix_ids(cls) -> set[str]:
        """Load technique IDs from cached STIX bundle, downloading if needed."""
        import json
        import re
        from pathlib import Path

        cache = Path(cls._STIX_CACHE_PATH)
        if cache.exists():
            try:
                return cls._parse_stix_bundle(json.loads(cache.read_text(encoding="utf-8")))
            except Exception:
                pass

        # Attempt download
        try:
            return cls._download_stix_bundle(cache)
        except Exception:
            return set()

    @classmethod
    def _download_stix_bundle(cls, cache_path) -> set[str]:
        """Download ATT&CK enterprise STIX bundle and cache it."""
        import json
        from pathlib import Path
        from urllib.request import Request, urlopen

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        req = Request(cls._STIX_URL, headers={"User-Agent": "CyberForge/0.1"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return cls._parse_stix_bundle(data)

    @staticmethod
    def _parse_stix_bundle(bundle: dict) -> set[str]:
        """Extract technique IDs from STIX 2.1 bundle objects."""
        ids: set[str] = set()
        for obj in bundle.get("objects", []):
            if obj.get("type") == "attack-pattern":
                for ref in obj.get("external_references", []):
                    eid = ref.get("external_id", "")
                    if eid.startswith("T"):
                        ids.add(eid)
        return ids

    @classmethod
    def reload_stix_bundle(cls) -> int:
        """Force re-download of the STIX bundle. Returns count of technique IDs loaded."""
        from pathlib import Path
        cache = Path(cls._STIX_CACHE_PATH)
        cls._technique_ids = cls._download_stix_bundle(cache)
        return len(cls._technique_ids)
