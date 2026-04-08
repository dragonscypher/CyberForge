"""Log ingestion — parsers for Sysmon, Suricata, Zeek, and Windows Security logs.

Each parser normalises events into a common schema for downstream analysis.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

log = logging.getLogger(__name__)


class NormalisedEvent(BaseModel):
    """Common event schema across all log sources."""
    timestamp: str = ""
    source_type: str = ""  # sysmon | suricata | zeek | windows_security
    event_id: str = ""
    severity: str = "info"  # info | low | medium | high | critical
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: str = ""
    action: str = ""  # allow | block | alert | create | terminate | ...
    summary: str = ""
    raw: dict[str, Any] = {}


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ── Sysmon (JSONL / EVTX-exported JSON) ─────────────────────────


def parse_sysmon_jsonl(path: str | Path) -> list[NormalisedEvent]:
    """Parse Sysmon logs in JSONL format (one JSON object per line)."""
    events: list[NormalisedEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Sysmon JSONL parse error at line %d", line_num)
                continue

            # Sysmon events can be nested under EventData or System
            event_data = raw.get("EventData", raw)
            system = raw.get("System", {})

            events.append(NormalisedEvent(
                timestamp=event_data.get("UtcTime", system.get("TimeCreated", {}).get("SystemTime", "")),
                source_type="sysmon",
                event_id=str(system.get("EventID", event_data.get("EventID", ""))),
                severity=_sysmon_severity(system.get("EventID", 0)),
                src_ip=event_data.get("SourceIp", ""),
                dst_ip=event_data.get("DestinationIp", ""),
                src_port=_safe_int(event_data.get("SourcePort")),
                dst_port=_safe_int(event_data.get("DestinationPort")),
                protocol=event_data.get("Protocol", ""),
                action=event_data.get("EventType", ""),
                summary=_sysmon_summary(event_data),
                raw=raw,
            ))
    return events


def _sysmon_severity(event_id: Any) -> str:
    eid = _safe_int(event_id)
    # High-interest events
    if eid in (1, 3, 7, 8, 10, 11, 12, 13, 15, 22, 23, 25):
        return "medium"
    if eid in (2, 9, 17, 18, 19, 20, 21):
        return "high"
    return "info"


def _sysmon_summary(data: dict) -> str:
    image = data.get("Image", data.get("TargetFilename", ""))
    cmd = data.get("CommandLine", "")
    if cmd:
        return f"{image}: {cmd[:200]}"
    return str(image)


# ── Suricata (EVE JSON) ─────────────────────────────────────────


def parse_suricata_eve(path: str | Path) -> list[NormalisedEvent]:
    """Parse Suricata EVE JSON log (one JSON object per line)."""
    events: list[NormalisedEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = raw.get("event_type", "")
            alert = raw.get("alert", {})

            severity_map = {1: "high", 2: "medium", 3: "low"}
            sev = severity_map.get(alert.get("severity", 4), "info")

            events.append(NormalisedEvent(
                timestamp=raw.get("timestamp", ""),
                source_type="suricata",
                event_id=str(alert.get("signature_id", "")),
                severity=sev if alert else "info",
                src_ip=raw.get("src_ip", ""),
                dst_ip=raw.get("dest_ip", ""),
                src_port=_safe_int(raw.get("src_port")),
                dst_port=_safe_int(raw.get("dest_port")),
                protocol=raw.get("proto", ""),
                action=raw.get("alert", {}).get("action", event_type),
                summary=alert.get("signature", event_type),
                raw=raw,
            ))
    return events


# ── Zeek (TSV conn.log / JSON) ──────────────────────────────────


def parse_zeek_conn(path: str | Path) -> list[NormalisedEvent]:
    """Parse Zeek conn.log (TSV format with # comment headers, or JSON)."""
    path = Path(path)
    content = path.read_text(encoding="utf-8")

    # Detect JSON format
    first_line = content.split("\n", 1)[0].strip()
    if first_line.startswith("{"):
        return _parse_zeek_json(content)

    return _parse_zeek_tsv(content)


def _parse_zeek_json(content: str) -> list[NormalisedEvent]:
    events: list[NormalisedEvent] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = raw.get("ts", "")
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts).isoformat()

        events.append(NormalisedEvent(
            timestamp=str(ts),
            source_type="zeek",
            event_id=raw.get("uid", ""),
            severity="info",
            src_ip=raw.get("id.orig_h", raw.get("id_orig_h", "")),
            dst_ip=raw.get("id.resp_h", raw.get("id_resp_h", "")),
            src_port=_safe_int(raw.get("id.orig_p", raw.get("id_orig_p"))),
            dst_port=_safe_int(raw.get("id.resp_p", raw.get("id_resp_p"))),
            protocol=raw.get("proto", ""),
            action=raw.get("conn_state", ""),
            summary=f"{raw.get('service', '')} {raw.get('conn_state', '')}".strip(),
            raw=raw,
        ))
    return events


def _parse_zeek_tsv(content: str) -> list[NormalisedEvent]:
    events: list[NormalisedEvent] = []
    headers: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#fields"):
            headers = line.split("\t")[1:]
            continue
        if line.startswith("#") or not line:
            continue
        if not headers:
            continue

        fields = line.split("\t")
        raw: dict[str, str] = {}
        for i, h in enumerate(headers):
            raw[h] = fields[i] if i < len(fields) else "-"

        ts = raw.get("ts", "")
        try:
            ts = datetime.fromtimestamp(float(ts)).isoformat()
        except (ValueError, TypeError, OSError):
            pass

        events.append(NormalisedEvent(
            timestamp=ts,
            source_type="zeek",
            event_id=raw.get("uid", ""),
            severity="info",
            src_ip=raw.get("id.orig_h", ""),
            dst_ip=raw.get("id.resp_h", ""),
            src_port=_safe_int(raw.get("id.orig_p")),
            dst_port=_safe_int(raw.get("id.resp_p")),
            protocol=raw.get("proto", ""),
            action=raw.get("conn_state", ""),
            summary=f"{raw.get('service', '')} {raw.get('conn_state', '')}".strip(),
            raw=raw,
        ))
    return events


# ── Windows Security (EVTX-exported JSONL) ──────────────────────


def parse_windows_security(path: str | Path) -> list[NormalisedEvent]:
    """Parse Windows Security event log exported as JSONL."""
    events: list[NormalisedEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            system = raw.get("System", raw)
            event_data = raw.get("EventData", {})
            eid = _safe_int(system.get("EventID", system.get("event_id", 0)))

            events.append(NormalisedEvent(
                timestamp=system.get("TimeCreated", {}).get("SystemTime", raw.get("timestamp", "")),
                source_type="windows_security",
                event_id=str(eid),
                severity=_windows_severity(eid),
                src_ip=event_data.get("IpAddress", event_data.get("SourceAddress", "")),
                dst_ip=event_data.get("DestAddress", ""),
                src_port=_safe_int(event_data.get("SourcePort")),
                dst_port=_safe_int(event_data.get("DestPort")),
                protocol="",
                action=_windows_action(eid),
                summary=_windows_summary(eid, event_data),
                raw=raw,
            ))
    return events


def _windows_severity(eid: int) -> str:
    # Logon failure, explicit credential use, privilege escalation, etc.
    critical_ids = {4625, 4648, 4672, 4697, 4698, 4699, 4720, 4728, 4732, 4756}
    medium_ids = {4624, 4634, 4647, 4688, 4689}
    if eid in critical_ids:
        return "high"
    if eid in medium_ids:
        return "medium"
    return "info"


def _windows_action(eid: int) -> str:
    actions = {
        4624: "logon", 4625: "logon_failed", 4634: "logoff",
        4647: "logoff", 4648: "explicit_logon", 4672: "special_privilege",
        4688: "process_create", 4689: "process_terminate",
        4697: "service_install", 4698: "task_create", 4699: "task_delete",
        4720: "account_create", 4728: "group_member_add",
        4732: "group_member_add", 4756: "group_member_add",
    }
    return actions.get(eid, f"event_{eid}")


def _windows_summary(eid: int, data: dict) -> str:
    user = data.get("TargetUserName", data.get("SubjectUserName", ""))
    process = data.get("NewProcessName", data.get("ProcessName", ""))
    parts = [f"EID:{eid}"]
    if user:
        parts.append(f"user={user}")
    if process:
        parts.append(f"proc={process}")
    return " ".join(parts)


# ── Unified ingest function ─────────────────────────────────────


def ingest(source_type: str, path: str | Path) -> list[NormalisedEvent]:
    """Ingest logs from a given source type and path.

    Supported source_type values:
      sysmon_jsonl, suricata_eve, zeek_conn, windows_security
    """
    parsers = {
        "sysmon_jsonl": parse_sysmon_jsonl,
        "sysmon": parse_sysmon_jsonl,
        "suricata_eve": parse_suricata_eve,
        "suricata": parse_suricata_eve,
        "zeek_conn": parse_zeek_conn,
        "zeek": parse_zeek_conn,
        "windows_security": parse_windows_security,
    }

    parser = parsers.get(source_type.lower())
    if parser is None:
        raise ValueError(
            f"Unknown source_type: {source_type}. "
            f"Supported: {', '.join(sorted(parsers.keys()))}"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    events = parser(path)
    log.info("Ingested %d events from %s (%s)", len(events), path, source_type)
    return events
