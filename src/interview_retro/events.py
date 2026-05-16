"""
Event types and event bus for the interview-retro pipeline.

The backend publishes events to the EventBus; crew listeners subscribe via
EventBus.start_workers(), which lazily imports listener.py so that no crew
code is loaded when this module is imported.
"""
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

ANALYSIS_QUEUE_MAX = 1


@dataclass
class AnalysisRequested:
    """Published by the backend when a transcript is ready for AI analysis."""
    interview_id: str
    transcript: str


@dataclass
class RegradeRequested:
    """
    Published by the backend when a user submits an improved answer.

    ``future`` is resolved by the listener with the grading result so the
    HTTP handler can await it directly.
    """
    question: str
    new_answer: str
    category: str
    future: "asyncio.Future[dict[str, Any]]"


class EventBus:
    """
    Mediates between the backend and crew listeners.

    The backend publishes AnalysisRequested / RegradeRequested events to the
    queues on this bus.  Crew workers are wired up via start_workers(), which
    lazily imports listener.py — the only module that may import crew code —
    so that importing events.py never pulls in any crew dependencies.
    """

    def __init__(self, analysis_queue_max: int = ANALYSIS_QUEUE_MAX) -> None:
        self.analysis_queue: asyncio.Queue[AnalysisRequested] = asyncio.Queue(
            maxsize=analysis_queue_max
        )
        self.regrade_queue: asyncio.Queue[RegradeRequested] = asyncio.Queue()

    def start_workers(
        self,
        on_start: Callable[[str, int], None],
        on_complete: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> tuple["asyncio.Task[None]", "asyncio.Task[None]"]:
        """
        Lazily import crew listener code and start async worker tasks.

        Crew imports are deferred to this call so that importing events.py
        does not pull in any crew dependencies.  Returns the two asyncio Tasks
        so the caller can cancel them on shutdown.
        """
        from interview_retro.listener import (  # noqa: PLC0415
            run_analysis_listener,
            run_regrade_listener,
        )

        analysis_task: asyncio.Task[None] = asyncio.create_task(
            run_analysis_listener(
                self.analysis_queue,
                on_start=on_start,
                on_complete=on_complete,
            )
        )
        regrade_task: asyncio.Task[None] = asyncio.create_task(
            run_regrade_listener(self.regrade_queue)
        )
        return analysis_task, regrade_task
