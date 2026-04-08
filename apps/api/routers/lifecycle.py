"""Lifecycle endpoints — cache, saved models, cleanup, save/discard with audit."""

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from packages.core.lifecycle import ArtifactInfo

router = APIRouter()


@router.get("/cache", response_model=list[ArtifactInfo])
async def list_cache(request: Request):
    return request.app.state.lifecycle.list_cache()


@router.get("/saved", response_model=list[ArtifactInfo])
async def list_saved(request: Request):
    return request.app.state.lifecycle.list_saved()


@router.get("/disk-usage")
async def disk_usage(request: Request):
    return request.app.state.lifecycle.disk_usage_mb()


class PromoteRequest(BaseModel):
    cache_name: str
    new_name: str | None = None


@router.post("/promote")
async def promote(body: PromoteRequest, request: Request):
    result = request.app.state.lifecycle.promote_to_saved(body.cache_name, body.new_name)
    if result is None:
        raise HTTPException(404, f"Cache artifact '{body.cache_name}' not found")
    return {"saved_path": result}


@router.delete("/artifact")
async def delete_artifact(path: str, request: Request):
    ok = request.app.state.lifecycle.delete_artifact(path)
    if not ok:
        raise HTTPException(404, "Artifact not found")
    return {"deleted": True}


@router.post("/cleanup/volatile")
async def cleanup_volatile(request: Request):
    count = request.app.state.lifecycle.cleanup_volatile()
    return {"deleted_count": count}


@router.post("/cleanup/stale-cache")
async def cleanup_stale_cache(request: Request, max_age_hours: float = 24):
    count = request.app.state.lifecycle.cleanup_stale_cache(max_age_hours)
    return {"deleted_count": count}


# ── Save/Discard with audit (OPT-005) ───────────────────────────


class SaveRequest(BaseModel):
    """Save a temp artifact: promotes from cache to permanent storage."""
    cache_name: str
    display_name: str | None = None


class DiscardRequest(BaseModel):
    """Discard a temp artifact: deletes it from cache."""
    cache_name: str


class AuditEvent(BaseModel):
    entity_type: str
    entity_id: str
    action: str
    details: dict | None = None
    created_at: float = 0.0


_audit_log: list[AuditEvent] = []


def _record_audit(entity_type: str, entity_id: str, action: str, details: dict | None = None):
    """Record an audit event in memory (persisted via DB in production)."""
    _audit_log.append(
        AuditEvent(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            details=details,
            created_at=time.time(),
        )
    )


@router.post("/save")
async def save_artifact(body: SaveRequest, request: Request):
    """Save (promote) a temp artifact from cache to permanent storage.

    Records an audit event per plan.md Section 9.2 save/discard flow.
    """
    lm = request.app.state.lifecycle
    result = lm.promote_to_saved(body.cache_name, body.display_name)
    if result is None:
        raise HTTPException(404, f"Cache artifact '{body.cache_name}' not found")

    _record_audit(
        entity_type="artifact",
        entity_id=body.cache_name,
        action="save",
        details={"saved_path": result, "display_name": body.display_name},
    )
    return {"saved_path": result, "audit_recorded": True}


@router.post("/discard")
async def discard_artifact(body: DiscardRequest, request: Request):
    """Discard a temp artifact — deletes it and records an audit event."""
    lm = request.app.state.lifecycle

    # Find the artifact in cache
    cache_items = lm.list_cache()
    target = next((i for i in cache_items if i.name == body.cache_name), None)
    if target is None:
        raise HTTPException(404, f"Cache artifact '{body.cache_name}' not found")

    ok = lm.delete_artifact(target.path)
    if not ok:
        raise HTTPException(500, "Failed to delete artifact")

    _record_audit(
        entity_type="artifact",
        entity_id=body.cache_name,
        action="discard",
        details={"deleted_path": target.path},
    )
    return {"deleted": True, "audit_recorded": True}


@router.get("/audit", response_model=list[AuditEvent])
async def list_audit_events(limit: int = 100):
    """List recent audit events (most recent first)."""
    return list(reversed(_audit_log[-limit:]))
