"""Jobs endpoints — status, cancel, list, SSE stream."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class JobStatusResponse(BaseModel):
    job_id: str
    type: str
    status: str
    progress: int
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@router.get("/", response_model=list[JobStatusResponse])
async def list_jobs(request: Request):
    """List all jobs ordered by creation time (newest first)."""
    runner = request.app.state.job_runner
    jobs = await runner.list_all()
    return [JobStatusResponse(**j) for j in jobs]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, request: Request):
    runner = request.app.state.job_runner
    status = await runner.get_status(job_id)
    if status is None:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return JobStatusResponse(**status)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request):
    runner = request.app.state.job_runner
    cancelled = await runner.cancel(job_id)
    if not cancelled:
        raise HTTPException(404, "Job not found or not cancellable")
    return {"cancelled": True, "job_id": job_id}


@router.get("/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    """SSE endpoint for live job progress (plan.md §5.5)."""
    runner = request.app.state.job_runner

    async def event_generator():
        terminal_states = {"completed", "failed", "cancelled"}
        while True:
            status = await runner.get_status(job_id)
            if status is None:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                return
            yield f"data: {json.dumps(status)}\n\n"
            if status.get("status") in terminal_states:
                return
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
