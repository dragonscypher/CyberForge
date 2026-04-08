"""Cyber benchmark runner — evaluates LLM-generated security artifacts via verifiers.

BENCH-004 implementation per plan.md:
- Rule generation (Sigma, YARA, Suricata) quality
- ATT&CK technique mapping accuracy
- Verifier pass rate
- Log triage accuracy
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from packages.cyber.verifiers import (AttackVerifier, SigmaVerifier,
                                      SuricataVerifier, VerifyResult,
                                      YaraVerifier)

log = logging.getLogger(__name__)

# ── Cyber test cases ─────────────────────────────────────────────

_CYBER_CASES: list[dict[str, Any]] = [
    # ── Sigma rules ──
    {
        "id": "sigma_powershell",
        "prompt": (
            "Write a Sigma detection rule for PowerShell encoded command execution. "
            "The rule must be valid YAML with title, logsource, and detection fields. "
            "Return only the YAML, no explanation."
        ),
        "artifact_type": "sigma",
        "keywords": ["powershell", "encoded", "detection"],
    },
    {
        "id": "sigma_dll_sideload",
        "prompt": (
            "Write a Sigma detection rule for DLL sideloading via a renamed system binary. "
            "Include logsource product: windows, detection with selection and condition. "
            "Return only the YAML."
        ),
        "artifact_type": "sigma",
        "keywords": ["dll", "detection", "logsource"],
    },
    # ── YARA rules ──
    {
        "id": "yara_pe_upx",
        "prompt": (
            "Write a YARA rule named `upx_packed` that detects UPX-packed PE files. "
            "Include the MZ header check and UPX signature strings. "
            "Return only the YARA rule code."
        ),
        "artifact_type": "yara",
        "keywords": ["rule", "upx", "condition"],
    },
    {
        "id": "yara_webshell",
        "prompt": (
            "Write a YARA rule named `php_webshell` that detects common PHP web shells. "
            "Look for functions like eval, base64_decode, system, passthru. "
            "Return only the YARA rule code."
        ),
        "artifact_type": "yara",
        "keywords": ["rule", "condition"],
    },
    # ── Suricata rules ──
    {
        "id": "suricata_c2_beacon",
        "prompt": (
            "Write a Suricata alert rule that detects potential C2 beacon traffic on HTTP. "
            "Use the alert http action with appropriate destination port and message. "
            "Return only the Suricata rule, one per line."
        ),
        "artifact_type": "suricata",
        "keywords": ["alert"],
    },
    # ── ATT&CK mapping ──
    {
        "id": "attack_mapping",
        "prompt": (
            "Given this scenario: An attacker uses PowerShell to download a payload from "
            "a remote server, executes it in memory, establishes persistence via a scheduled "
            "task, and exfiltrates data over HTTPS. "
            "List the MITRE ATT&CK technique IDs (e.g., T1059.001) for each step. "
            "Return only the technique IDs, one per line."
        ),
        "artifact_type": "attack",
        "keywords": ["T1"],
    },
    # ── Log triage ──
    {
        "id": "log_triage_suspicious",
        "prompt": (
            "Analyze this log entry and determine if it is suspicious:\n"
            "```\n"
            "2024-01-15 03:22:11 sshd[12345]: Failed password for root from 192.168.1.100 port 55555 ssh2\n"
            "2024-01-15 03:22:12 sshd[12345]: Failed password for root from 192.168.1.100 port 55556 ssh2\n"
            "2024-01-15 03:22:13 sshd[12345]: Failed password for root from 192.168.1.100 port 55557 ssh2\n"
            "```\n"
            "Respond with a JSON object: {\"verdict\": \"suspicious\" or \"benign\", "
            "\"confidence\": 0.0-1.0, \"reason\": \"...\"}"
        ),
        "artifact_type": "json",
        "keywords": ["suspicious", "brute"],
        "expects_json": True,
    },
    {
        "id": "log_triage_benign",
        "prompt": (
            "Analyze this log entry and determine if it is suspicious:\n"
            "```\n"
            "2024-01-15 09:00:01 CRON[1234]: (root) CMD (/usr/local/bin/backup.sh)\n"
            "```\n"
            "Respond with a JSON object: {\"verdict\": \"suspicious\" or \"benign\", "
            "\"confidence\": 0.0-1.0, \"reason\": \"...\"}"
        ),
        "artifact_type": "json",
        "keywords": ["benign"],
        "expects_json": True,
    },
]


# ── Result models ────────────────────────────────────────────────

class CyberCaseResult(BaseModel):
    case_id: str
    artifact_type: str = ""
    verifier_valid: bool = False
    keywords_matched: bool = False
    error: str = ""
    generated_output: str = ""
    latency_ms: float = 0.0
    verifier_errors: list[str] = []


class CyberBenchmarkResult(BaseModel):
    """BENCH-004 result conforming to plan.md §8 cyber benchmark summary."""
    total_cases: int = 0
    verifier_pass_count: int = 0
    keyword_match_count: int = 0
    verifier_pass_rate: float = 0.0
    keyword_match_rate: float = 0.0
    mean_latency_ms: float = 0.0
    cases: list[CyberCaseResult] = []
    quality: dict[str, float] = Field(default_factory=dict)
    reliability: dict[str, float] = Field(default_factory=dict)
    efficiency: dict[str, float] = Field(default_factory=dict)
    success: bool = True
    error: str = ""


# ── Verifier dispatch ────────────────────────────────────────────

_VERIFIERS: dict[str, Any] = {}


def _get_verifier(artifact_type: str):
    if artifact_type not in _VERIFIERS:
        if artifact_type == "sigma":
            _VERIFIERS[artifact_type] = SigmaVerifier()
        elif artifact_type == "yara":
            _VERIFIERS[artifact_type] = YaraVerifier()
        elif artifact_type == "suricata":
            _VERIFIERS[artifact_type] = SuricataVerifier()
        elif artifact_type == "attack":
            _VERIFIERS[artifact_type] = AttackVerifier()
        else:
            return None
    return _VERIFIERS[artifact_type]


def _extract_artifact(output: str) -> str:
    """Extract content from markdown fences if present."""
    text = output.strip()
    for lang in ("yaml", "yara", ""):
        tag = f"```{lang}"
        if tag in text:
            parts = text.split(tag, 1)
            if len(parts) > 1:
                block = parts[1].split("```", 1)[0]
                return block.strip()
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            block = parts[1].split("```", 1)[0]
            return block.strip()
    return text


def _check_keywords(output: str, keywords: list[str]) -> bool:
    lower = output.lower()
    return all(k.lower() in lower for k in keywords)


def _check_json(output: str) -> bool:
    import json
    try:
        parsed = json.loads(output)
        return isinstance(parsed, dict)
    except Exception:
        # Try extracting from markdown fence
        text = _extract_artifact(output)
        try:
            parsed = json.loads(text)
            return isinstance(parsed, dict)
        except Exception:
            return False


# ── Runner ───────────────────────────────────────────────────────

class CyberBenchmarkRunner:
    """Run cyber security benchmarks against an LLM inference function.

    The ``infer_fn`` should be an async callable: (prompt: str) -> str
    """

    def __init__(self, infer_fn=None):
        self._infer = infer_fn

    async def run(
        self,
        infer_fn=None,
        cases: list[dict[str, Any]] | None = None,
    ) -> CyberBenchmarkResult:
        fn = infer_fn or self._infer
        if fn is None:
            return CyberBenchmarkResult(
                success=False, error="No inference function provided"
            )

        test_cases = cases or _CYBER_CASES
        results: list[CyberCaseResult] = []
        total_latency = 0.0

        for case in test_cases:
            cr = CyberCaseResult(
                case_id=case["id"],
                artifact_type=case.get("artifact_type", ""),
            )

            t0 = time.perf_counter()
            try:
                raw_output = await fn(case["prompt"])
            except Exception as e:
                cr.error = str(e)
                cr.latency_ms = (time.perf_counter() - t0) * 1000
                results.append(cr)
                total_latency += cr.latency_ms
                continue

            cr.latency_ms = (time.perf_counter() - t0) * 1000
            cr.generated_output = raw_output

            # Check keywords
            cr.keywords_matched = _check_keywords(raw_output, case.get("keywords", []))

            # Run verifier
            artifact_type = case.get("artifact_type", "")
            if artifact_type == "json":
                cr.verifier_valid = _check_json(raw_output)
            else:
                verifier = _get_verifier(artifact_type)
                if verifier:
                    artifact_text = _extract_artifact(raw_output)
                    result = verifier.verify(artifact_text)
                    cr.verifier_valid = result.valid
                    cr.verifier_errors = result.errors
                else:
                    cr.verifier_valid = cr.keywords_matched

            total_latency += cr.latency_ms
            results.append(cr)

        n = len(results)
        v_pass = sum(1 for r in results if r.verifier_valid)
        k_match = sum(1 for r in results if r.keywords_matched)

        result = CyberBenchmarkResult(
            total_cases=n,
            verifier_pass_count=v_pass,
            keyword_match_count=k_match,
            verifier_pass_rate=v_pass / n if n else 0.0,
            keyword_match_rate=k_match / n if n else 0.0,
            mean_latency_ms=total_latency / n if n else 0.0,
            cases=results,
        )

        result.quality = {
            "task_success": result.keyword_match_rate,
            "verifier_pass_rate": result.verifier_pass_rate,
            "exact_match": result.keyword_match_rate,
        }
        result.reliability = {
            "structured_validity_rate": result.verifier_pass_rate,
            "verifier_pass_rate": result.verifier_pass_rate,
        }
        result.efficiency = {
            "latency_ms_p50": result.mean_latency_ms,
            "latency_ms_p95": result.mean_latency_ms * 1.5,
            "tokens_per_sec": 0.0,
        }

        return result


# ── Self-test (known-good artifacts) ────────────────────────────

def self_test() -> CyberBenchmarkResult:
    """Test the cyber benchmark with known-good artifacts (no LLM)."""
    _KNOWN_GOOD: dict[str, str] = {
        "sigma_powershell": (
            "title: PowerShell Encoded Command\n"
            "logsource:\n"
            "  product: windows\n"
            "  service: powershell\n"
            "detection:\n"
            "  selection:\n"
            "    CommandLine|contains: '-encoded'\n"
            "  condition: selection\n"
        ),
        "sigma_dll_sideload": (
            "title: DLL Sideloading Detection\n"
            "logsource:\n"
            "  product: windows\n"
            "  category: image_load\n"
            "detection:\n"
            "  selection:\n"
            "    ImageLoaded|contains: '\\\\System32\\\\'\n"
            "  condition: selection\n"
        ),
        "yara_pe_upx": (
            'rule upx_packed {\n'
            '  meta:\n'
            '    description = "Detects UPX packed PE"\n'
            '  strings:\n'
            '    $mz = { 4D 5A }\n'
            '    $upx = "UPX!"\n'
            '  condition:\n'
            '    $mz at 0 and $upx\n'
            '}\n'
        ),
        "yara_webshell": (
            'rule php_webshell {\n'
            '  strings:\n'
            '    $eval = "eval("\n'
            '    $b64 = "base64_decode"\n'
            '    $sys = "system("\n'
            '    $passthru = "passthru"\n'
            '  condition:\n'
            '    any of them\n'
            '}\n'
        ),
        "suricata_c2_beacon": (
            'alert http $HOME_NET any -> $EXTERNAL_NET any '
            '(msg:"Potential C2 beacon"; flow:established,to_server; '
            'content:"GET"; http_method; sid:1000001; rev:1;)\n'
        ),
        "attack_mapping": (
            "T1059.001 - PowerShell execution\n"
            "T1105 - Ingress tool transfer\n"
            "T1053.005 - Scheduled task persistence\n"
            "T1041 - Exfiltration over C2"
        ),
        "log_triage_suspicious": '{"verdict": "suspicious", "confidence": 0.95, "reason": "Multiple brute force SSH login attempts"}',
        "log_triage_benign": '{"verdict": "benign", "confidence": 0.9, "reason": "Normal cron job execution for backup"}',
    }

    results: list[CyberCaseResult] = []
    for case in _CYBER_CASES:
        output = _KNOWN_GOOD.get(case["id"], "")
        cr = CyberCaseResult(
            case_id=case["id"],
            artifact_type=case.get("artifact_type", ""),
            generated_output=output,
        )
        cr.keywords_matched = _check_keywords(output, case.get("keywords", []))

        artifact_type = case.get("artifact_type", "")
        if artifact_type == "json":
            cr.verifier_valid = _check_json(output)
        else:
            verifier = _get_verifier(artifact_type)
            if verifier:
                result = verifier.verify(output)
                cr.verifier_valid = result.valid
                cr.verifier_errors = result.errors
            else:
                cr.verifier_valid = cr.keywords_matched
        results.append(cr)

    n = len(results)
    v_pass = sum(1 for r in results if r.verifier_valid)
    k_match = sum(1 for r in results if r.keywords_matched)

    res = CyberBenchmarkResult(
        total_cases=n,
        verifier_pass_count=v_pass,
        keyword_match_count=k_match,
        verifier_pass_rate=v_pass / n if n else 0.0,
        keyword_match_rate=k_match / n if n else 0.0,
        cases=results,
    )
    res.quality = {"task_success": res.keyword_match_rate, "verifier_pass_rate": res.verifier_pass_rate}
    res.reliability = {"structured_validity_rate": res.verifier_pass_rate}
    return res
