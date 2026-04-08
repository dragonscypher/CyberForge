"""Cyber discovery scanner — enumerate local network services and assets.

Provides lightweight, non-intrusive scanning of reachable hosts and services
for the cyber copilot's situational awareness.  This does NOT perform
penetration testing or exploit delivery.
"""

from __future__ import annotations

import logging
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


class DiscoveredService(BaseModel):
    host: str
    port: int
    protocol: str = "tcp"
    service_name: str = ""
    banner: str = ""
    open: bool = True


class DiscoveryScanResult(BaseModel):
    target: str
    services: list[DiscoveredService] = []
    total_scanned: int = 0
    total_open: int = 0
    error: str = ""


# Well-known ports for quick service identification
_COMMON_PORTS: dict[int, str] = {
    22: "ssh", 53: "dns", 80: "http", 443: "https", 445: "smb",
    3306: "mysql", 3389: "rdp", 5432: "postgresql", 6379: "redis",
    8080: "http-alt", 8443: "https-alt", 9200: "elasticsearch",
    27017: "mongodb",
}


def _check_port(host: str, port: int, timeout: float = 1.0) -> DiscoveredService | None:
    """Check if a single TCP port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            if result == 0:
                service_name = _COMMON_PORTS.get(port, "")
                banner = ""
                try:
                    s.sendall(b"\r\n")
                    s.settimeout(0.5)
                    banner = s.recv(256).decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
                return DiscoveredService(
                    host=host, port=port, service_name=service_name, banner=banner[:200],
                )
    except Exception:
        pass
    return None


def scan_host(
    host: str,
    ports: list[int] | None = None,
    timeout: float = 1.0,
    max_workers: int = 20,
) -> DiscoveryScanResult:
    """Scan a single host for open TCP ports.

    Args:
        host: Target hostname or IP.
        ports: List of ports to check.  Defaults to common service ports.
        timeout: Socket timeout in seconds.
        max_workers: Thread pool size.
    """
    if ports is None:
        ports = list(_COMMON_PORTS.keys())

    services: list[DiscoveredService] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_check_port, host, p, timeout): p for p in ports}
            for fut in as_completed(futures):
                svc = fut.result()
                if svc is not None:
                    services.append(svc)
    except Exception as e:
        return DiscoveryScanResult(target=host, total_scanned=len(ports), error=str(e))

    services.sort(key=lambda s: s.port)
    return DiscoveryScanResult(
        target=host,
        services=services,
        total_scanned=len(ports),
        total_open=len(services),
    )


def quick_scan(host: str = "127.0.0.1") -> dict[str, Any]:
    """Quick scan of common ports on a host, returns dict summary."""
    result = scan_host(host)
    return {
        "host": result.target,
        "open_ports": [{"port": s.port, "service": s.service_name} for s in result.services],
        "total_open": result.total_open,
    }
