"""Job runner — local worker process with state machine.

Supports: sequential execution, progress persistence, cancel, crash recovery.
No Redis/Celery — v1 uses a local asyncio task loop.

Job states: queued → preparing → running → completed | failed | cancelled
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from apps.api.db import AuditEvent, Job, _new_id, _utcnow

logger = logging.getLogger(__name__)

# Type for step functions: async (job_id, payload_dict, progress_cb) -> result_dict
StepFn = Callable[
    [str, dict[str, Any], Callable[[int], Coroutine[Any, Any, None]]],
    Coroutine[Any, Any, dict[str, Any]],
]


class JobRunner:
    """Sequential local job runner backed by SQLite."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self._handlers: dict[str, StepFn] = {}
        self._task: asyncio.Task | None = None
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._running = False

    def register(self, job_type: str, handler: StepFn) -> None:
        """Register a handler for a job type."""
        self._handlers[job_type] = handler

    async def start(self) -> None:
        """Start the background worker loop and recover stale jobs."""
        self._running = True
        await self._recover_stale_jobs()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the worker loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue(
        self,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 5,
    ) -> str:
        """Create a new job and persist it to the DB. Returns job_id."""
        import json

        job_id = _new_id()
        async with self._session_factory() as session:
            job = Job(
                id=job_id,
                job_type=job_type,
                status="queued",
                priority=priority,
                payload_json=json.dumps(payload),
                progress=0,
                created_at=_utcnow(),
            )
            session.add(job)
            await session.commit()
        return job_id

    async def cancel(self, job_id: str) -> bool:
        """Request cancellation. Returns True if the job was running/queued."""
        evt = self._cancel_events.get(job_id)
        if evt:
            evt.set()

        async with self._session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job is None:
                return False
            if job.status in ("queued", "preparing", "running"):
                job.status = "cancelled"
                job.completed_at = _utcnow()
                await session.commit()
                await self._audit(session, "job", job_id, "cancelled")
                return True
        return False

    async def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Return current job state from DB."""
        import json

        async with self._session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job is None:
                return None
            return {
                "job_id": job.id,
                "type": job.job_type,
                "status": job.status,
                "progress": job.progress,
                "result": json.loads(job.result_json) if job.result_json else None,
                "error": job.error_text,
                "started_at": str(job.started_at) if job.started_at else None,
                "completed_at": str(job.completed_at) if job.completed_at else None,
            }

    async def list_all(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return all jobs ordered by creation time (newest first)."""
        import json

        async with self._session_factory() as session:
            result = await session.execute(
                select(Job).order_by(Job.created_at.desc()).limit(limit)
            )
            jobs = result.scalars().all()
            return [
                {
                    "job_id": j.id,
                    "type": j.job_type,
                    "status": j.status,
                    "progress": j.progress,
                    "result": json.loads(j.result_json) if j.result_json else None,
                    "error": j.error_text,
                    "started_at": str(j.started_at) if j.started_at else None,
                    "completed_at": str(j.completed_at) if j.completed_at else None,
                }
                for j in jobs
            ]

    # ── Internal ─────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main worker loop — pick next queued job, execute, repeat."""
        while self._running:
            job_id = await self._pick_next()
            if job_id:
                await self._execute(job_id)
            else:
                await asyncio.sleep(1)

    async def _pick_next(self) -> str | None:
        """Find the highest-priority queued job."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(Job)
                .where(Job.status == "queued")
                .order_by(Job.priority.asc(), Job.created_at.asc())
                .limit(1)
            )
            job = result.scalar_one_or_none()
            return job.id if job else None

    async def _execute(self, job_id: str) -> None:
        """Execute a single job through its state machine."""
        import json

        cancel_evt = asyncio.Event()
        self._cancel_events[job_id] = cancel_evt

        try:
            async with self._session_factory() as session:
                result = await session.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job is None or job.status != "queued":
                    return

                handler = self._handlers.get(job.job_type)
                if handler is None:
                    job.status = "failed"
                    job.error_text = f"No handler for job type: {job.job_type}"
                    job.completed_at = _utcnow()
                    await session.commit()
                    return

                # preparing
                job.status = "preparing"
                job.started_at = _utcnow()
                await session.commit()

                payload = json.loads(job.payload_json)

            # running
            await self._update_status(job_id, "running")

            async def progress_cb(pct: int) -> None:
                if cancel_evt.is_set():
                    raise asyncio.CancelledError("Job cancelled by user")
                await self._update_progress(job_id, pct)

            result_data = await handler(job_id, payload, progress_cb)

            # completed
            async with self._session_factory() as session:
                res = await session.execute(select(Job).where(Job.id == job_id))
                job = res.scalar_one_or_none()
                if job and job.status == "running":
                    job.status = "completed"
                    job.progress = 100
                    job.result_json = json.dumps(result_data) if result_data else None
                    job.completed_at = _utcnow()
                    await session.commit()
                    await self._audit(session, "job", job_id, "completed")

        except asyncio.CancelledError:
            await self._update_status(job_id, "cancelled")
            logger.info("Job %s cancelled", job_id)
        except Exception:
            tb = traceback.format_exc()
            logger.error("Job %s failed: %s", job_id, tb)
            async with self._session_factory() as session:
                res = await session.execute(select(Job).where(Job.id == job_id))
                job = res.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.error_text = tb[-2000:]  # truncate
                    job.completed_at = _utcnow()
                    await session.commit()
                    await self._audit(session, "job", job_id, "failed")
        finally:
            self._cancel_events.pop(job_id, None)

    async def _update_status(self, job_id: str, status: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(Job).where(Job.id == job_id).values(status=status)
            )
            await session.commit()

    async def _update_progress(self, job_id: str, progress: int) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(Job).where(Job.id == job_id).values(progress=min(progress, 100))
            )
            await session.commit()

    async def _recover_stale_jobs(self) -> None:
        """Mark jobs stuck in running/preparing as failed (crash recovery)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(Job).where(Job.status.in_(["running", "preparing"]))
            )
            stale = result.scalars().all()
            for job in stale:
                logger.warning("Recovering stale job %s (was %s)", job.id, job.status)
                job.status = "failed"
                job.error_text = "Recovered after crash — job was interrupted"
                job.completed_at = _utcnow()
                await self._audit(session, "job", job.id, "crash_recovered")
            if stale:
                await session.commit()
            logger.info("Recovered %d stale jobs", len(stale))

    @staticmethod
    async def _audit(
        session: AsyncSession, entity_type: str, entity_id: str, action: str
    ) -> None:
        import json

        event = AuditEvent(
            id=_new_id(),
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            details_json=json.dumps({"timestamp": _utcnow().isoformat()}),
            created_at=_utcnow(),
        )
        session.add(event)
        await session.commit()
