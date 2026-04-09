"""Workflow state machine — governs optimization and training job flows.

Implements the state transitions specified in plan.md §9:
  - Optimize flow: queued → preparing → benchmark_before → optimizing →
    benchmark_after → validating → completed | failed | cancelled → cleanup_pending
  - Save/Discard flow: pending_review → saved | discarded
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel

log = logging.getLogger(__name__)


class OptimizeState(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    BENCHMARK_BEFORE = "benchmark_before"
    OPTIMIZING = "optimizing"
    BENCHMARK_AFTER = "benchmark_after"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLEANUP_PENDING = "cleanup_pending"


class SaveDiscardState(str, Enum):
    PENDING_REVIEW = "pending_review"
    SAVED = "saved"
    DISCARDED = "discarded"


# Valid transitions for the optimize flow
_OPTIMIZE_TRANSITIONS: dict[OptimizeState, list[OptimizeState]] = {
    OptimizeState.QUEUED: [OptimizeState.PREPARING, OptimizeState.CANCELLED],
    OptimizeState.PREPARING: [OptimizeState.BENCHMARK_BEFORE, OptimizeState.FAILED, OptimizeState.CANCELLED],
    OptimizeState.BENCHMARK_BEFORE: [OptimizeState.OPTIMIZING, OptimizeState.FAILED, OptimizeState.CANCELLED],
    OptimizeState.OPTIMIZING: [OptimizeState.BENCHMARK_AFTER, OptimizeState.FAILED, OptimizeState.CANCELLED],
    OptimizeState.BENCHMARK_AFTER: [OptimizeState.VALIDATING, OptimizeState.FAILED, OptimizeState.CANCELLED],
    OptimizeState.VALIDATING: [OptimizeState.COMPLETED, OptimizeState.FAILED, OptimizeState.CANCELLED],
    OptimizeState.COMPLETED: [OptimizeState.CLEANUP_PENDING],
    OptimizeState.FAILED: [OptimizeState.CLEANUP_PENDING],
    OptimizeState.CANCELLED: [OptimizeState.CLEANUP_PENDING],
    OptimizeState.CLEANUP_PENDING: [],
}

_SAVE_DISCARD_TRANSITIONS: dict[SaveDiscardState, list[SaveDiscardState]] = {
    SaveDiscardState.PENDING_REVIEW: [SaveDiscardState.SAVED, SaveDiscardState.DISCARDED],
    SaveDiscardState.SAVED: [],
    SaveDiscardState.DISCARDED: [],
}


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""


class OptimizeStateMachine:
    """Tracks state for an optimization job."""

    def __init__(self, job_id: str, initial: OptimizeState = OptimizeState.QUEUED):
        self.job_id = job_id
        self.state = initial
        self._history: list[tuple[OptimizeState, OptimizeState]] = []

    def transition(self, target: OptimizeState) -> None:
        allowed = _OPTIMIZE_TRANSITIONS.get(self.state, [])
        if target not in allowed:
            raise StateTransitionError(
                f"Job {self.job_id}: cannot transition from {self.state.value} "
                f"to {target.value}. Allowed: {[s.value for s in allowed]}"
            )
        old = self.state
        self.state = target
        self._history.append((old, target))
        log.info("Job %s: %s → %s", self.job_id, old.value, target.value)

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            OptimizeState.COMPLETED,
            OptimizeState.FAILED,
            OptimizeState.CANCELLED,
            OptimizeState.CLEANUP_PENDING,
        )

    @property
    def history(self) -> list[tuple[str, str]]:
        return [(a.value, b.value) for a, b in self._history]


class SaveDiscardStateMachine:
    """Tracks state for the save/discard decision after optimization."""

    def __init__(self, artifact_id: str):
        self.artifact_id = artifact_id
        self.state = SaveDiscardState.PENDING_REVIEW

    def save(self) -> None:
        self._transition(SaveDiscardState.SAVED)

    def discard(self) -> None:
        self._transition(SaveDiscardState.DISCARDED)

    def _transition(self, target: SaveDiscardState) -> None:
        allowed = _SAVE_DISCARD_TRANSITIONS.get(self.state, [])
        if target not in allowed:
            raise StateTransitionError(
                f"Artifact {self.artifact_id}: cannot transition from "
                f"{self.state.value} to {target.value}"
            )
        old = self.state
        self.state = target
        log.info("Artifact %s: %s → %s", self.artifact_id, old.value, target.value)
