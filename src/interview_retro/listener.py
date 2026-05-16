"""
Crew event listeners.

This module is the ONLY place in the codebase that imports crew code.
The backend publishes AnalysisRequested / RegradeRequested events; the
listeners here subscribe (drain queues), run the crew in a thread, and
resolve results via callbacks or asyncio Futures.
"""
import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, Callable

from interview_retro.analysis import InterviewAnalysisCrew
from interview_retro.events import AnalysisRequested, RegradeRequested

logger = logging.getLogger(__name__)


async def run_analysis_listener(
    queue: asyncio.Queue[AnalysisRequested],
    on_start: Callable[[str, int], None],
    on_complete: Callable[[str, dict[str, Any]], Awaitable[None]],
) -> None:
    """
    Drain the analysis queue one job at a time.

    on_start(interview_id, remaining_in_queue) — called before crew runs.
    on_complete(interview_id, result_dict)      — called after crew finishes
                                                  (or on error/skip).
    """
    logger.info("Crew analysis listener started")
    while True:
        event = await queue.get()
        try:
            on_start(event.interview_id, queue.qsize())

            if not event.transcript or len(event.transcript) < 100:
                reason = (
                    "Transcript too short to analyze"
                    if event.transcript
                    else "No transcript provided"
                )
                logger.warning(f"Skipping analysis for {event.interview_id}: {reason}")
                await on_complete(
                    event.interview_id,
                    {
                        "analysis_status": "skipped",
                        "analysis_error": reason,
                        "rated_qa": [],
                        "overall_score": None,
                        "strengths": [],
                        "weaknesses": [],
                        "summary": None,
                    },
                )
                continue

            result = await asyncio.to_thread(
                InterviewAnalysisCrew().run,
                event.transcript,
                event.company_name,
                event.role,
                event.stage,
            )
            await on_complete(event.interview_id, result)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"Analysis listener error for {event.interview_id}: {exc}",
                exc_info=True,
            )
            await on_complete(
                event.interview_id,
                {
                    "analysis_status": "failed",
                    "analysis_error": str(exc),
                    "rated_qa": [],
                    "overall_score": None,
                    "strengths": [],
                    "weaknesses": [],
                    "summary": None,
                },
            )
        finally:
            queue.task_done()


async def run_regrade_listener(queue: asyncio.Queue[RegradeRequested]) -> None:
    """
    Drain the regrade queue one job at a time.

    Each RegradeRequested carries an asyncio.Future; the listener resolves
    it so the HTTP handler can await the result directly.
    """
    logger.info("Crew regrade listener started")
    while True:
        event = await queue.get()
        try:
            result = await asyncio.to_thread(
                InterviewAnalysisCrew().regrade_answer,
                event.question,
                event.new_answer,
                event.category,
            )
            if not event.future.done():
                event.future.set_result(result)
        except asyncio.CancelledError:
            if not event.future.done():
                event.future.cancel()
            raise
        except Exception as exc:
            if not event.future.done():
                event.future.set_exception(exc)
        finally:
            queue.task_done()
