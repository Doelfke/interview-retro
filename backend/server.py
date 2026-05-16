"""
Interview Retro Backend
FastAPI server. AI analysis is orchestrated by CrewAI.
"""
import asyncio
import json
import logging
import os
import sys
import uuid
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast

# Add src/ so interview_retro package is importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from storage.models import Interview, QAPair, engine, init_db
from interview_retro.events import AnalysisRequested, EventBus, RegradeRequested

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---- App state -------------------------------------------------------------


class AppState:
    event_bus: EventBus | None = None
    analysis_worker_task: asyncio.Task[None] | None = None
    regrade_worker_task: asyncio.Task[None] | None = None
    meetily_watcher_task: asyncio.Task[None] | None = None
    currently_analyzing: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()

    logger.info("=" * 60)
    logger.info("  Interview Retro — CrewAI")
    if os.getenv("OPENAI_API_KEY"):
        logger.info(f"  Mode:  local (OpenAI-compatible)")
        logger.info(f"  Model: {os.getenv('OPENAI_MODEL', '(OPENAI_MODEL not set)')}")
        logger.info(f"  URL:   {os.getenv('OPENAI_BASE_URL', '(OPENAI_BASE_URL not set)')}")
    else:
        logger.info(f"  Mode:  hosted (OpenRouter)")
        logger.info(f"  Model: {os.getenv('OPENROUTER_MODEL', 'gpt-oss:20b')}")
    logger.info("=" * 60)

    state.event_bus = EventBus()
    state.analysis_worker_task, state.regrade_worker_task = state.event_bus.start_workers(
        on_start=_on_analysis_start,
        on_complete=_on_analysis_complete,
    )
    state.meetily_watcher_task = asyncio.create_task(_meetily_watcher())

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    asyncio.create_task(_open_dashboard(host, port))

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    if state.meetily_watcher_task:
        state.meetily_watcher_task.cancel()
        try:
            await state.meetily_watcher_task
        except asyncio.CancelledError:
            pass
    if state.regrade_worker_task:
        state.regrade_worker_task.cancel()
        try:
            await state.regrade_worker_task
        except asyncio.CancelledError:
            pass
    if state.analysis_worker_task:
        state.analysis_worker_task.cancel()
        try:
            await state.analysis_worker_task
        except asyncio.CancelledError:
            pass


async def _open_dashboard(host: str, port: int) -> None:
    """Open the dashboard once on initial startup; skip on hot-reload."""
    import tempfile
    sentinel = Path(tempfile.gettempdir()) / ".interview_retro_sentinel"
    ppid = str(os.getppid())
    if sentinel.exists() and sentinel.read_text().strip() == ppid:
        logger.info("Reload detected — skipping browser open")
        return
    sentinel.write_text(ppid)
    await asyncio.sleep(1)
    url = f"http://{host}:{port}/dashboard"
    logger.info(f"Opening dashboard at {url}")
    webbrowser.open(url)


app = FastAPI(title="Interview Retro (local)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Analysis event callbacks (no crew code here) -------------------------

def _on_analysis_start(interview_id: str, remaining: int) -> None:
    state.currently_analyzing = True
    logger.info(f"Starting analysis {interview_id} ({remaining} more in queue)")


async def _on_analysis_complete(interview_id: str, result: dict[str, Any]) -> None:
    state.currently_analyzing = False
    status = result.get("analysis_status", "complete")
    if status == "failed":
        logger.error(f"Analysis failed for {interview_id}: {result.get('analysis_error')}")
        await asyncio.to_thread(
            _mark_analysis_result, interview_id, status="failed", error=result.get("analysis_error")
        )
    elif status == "skipped":
        logger.warning(f"Analysis skipped for {interview_id}: {result.get('analysis_error')}")
        await asyncio.to_thread(
            _mark_analysis_result, interview_id, status="skipped", error=result.get("analysis_error")
        )
    else:
        logger.info(f"Analysis complete for {interview_id} — {result.get('overall_score')}/10")
        await asyncio.to_thread(_update_analysis, interview_id, result)


def _mark_analysis_result(
    interview_id: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    with Session(engine) as db:
        interview = db.get(Interview, interview_id)
        if not interview:
            logger.error(f"_mark_analysis_result: interview {interview_id} not found")
            return
        interview.analysis_status = status
        interview.analysis_error = error
        db.commit()


def _update_analysis(interview_id: str, analysis: dict[str, Any]) -> None:
    with Session(engine) as db:
        interview = db.get(Interview, interview_id)
        if not interview:
            logger.error(f"_update_analysis: interview {interview_id} not found")
            return

        interview.analysis_status = analysis.get("analysis_status", "complete")
        interview.analysis_error  = analysis.get("analysis_error")
        interview.overall_score   = analysis.get("overall_score")
        interview.summary         = analysis.get("summary")
        interview.strengths       = analysis.get("strengths", [])
        interview.weaknesses      = analysis.get("weaknesses", [])

        raw_rated = analysis.get("rated_qa")
        rated_qa: list[dict[str, Any]] = []
        if isinstance(raw_rated, list):
            raw_rated_list = cast(list[Any], raw_rated)
            for row in raw_rated_list:
                if isinstance(row, dict):
                    rated_qa.append(cast(dict[str, Any], row))

        checkpoints = analysis.get("_checkpoints")
        raw_extracted: Any = cast(dict[str, Any], checkpoints).get("qa_pairs") if isinstance(checkpoints, dict) else None
        extracted_qa: list[dict[str, Any]] = []
        if isinstance(raw_extracted, list):
            raw_extracted_list = cast(list[Any], raw_extracted)
            for row in raw_extracted_list:
                if isinstance(row, dict):
                    extracted_qa.append(cast(dict[str, Any], row))

        # Fallback: if judge output omits answer/category fields (common),
        # reuse the extracted Q&A payload so dashboard/API never shows blanks.
        qa_rows = rated_qa if rated_qa else extracted_qa
        for idx, qa in enumerate(qa_rows):
            if not isinstance(qa, dict):
                continue

            extracted: dict[str, Any] = extracted_qa[idx] if idx < len(extracted_qa) else {}
            question = str(qa.get("question") or extracted.get("question") or "").strip()
            answer = str(qa.get("answer") or extracted.get("answer") or "").strip()
            category = str(qa.get("category") or extracted.get("category") or "general")
            timestamp_raw = (
                qa.get("timestamp_seconds")
                or extracted.get("timestamp_seconds")
                or extracted.get("timestamp_in_meeting")
                or extracted.get("timestamp")
            )
            timestamp: int | None = None
            if isinstance(timestamp_raw, (int, float)):
                timestamp = int(timestamp_raw)

            if not question:
                continue

            db.add(QAPair(
                id=str(uuid.uuid4()),
                interview_id=interview_id,
                question=question,
                answer=answer,
                score=qa.get("score"),
                category=category,
                feedback=qa.get("feedback"),
                suggested_answer=qa.get("suggested_answer"),
                timestamp_in_meeting=timestamp,
            ))

        db.commit()


# ---- Meetily folder watcher ------------------------------------------------

MEETILY_WATCH_DIR = Path.home() / "Movies" / "meetily-recordings"


async def _meetily_watcher() -> None:
    """
    On startup, scan ~/Movies/meetily-recordings/ for transcripts.json files
    that have not yet been ingested and store them as pending.
    No continuous folder watching is performed.
    """
    MEETILY_WATCH_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Meetily startup scan — checking {MEETILY_WATCH_DIR}")

    # Build the processed set from the database so we don't re-ingest meetings
    # that were already handled in a previous run.
    with Session(engine) as db:
        processed: set[Path] = {
            Path(r.source_path)
            for r in db.query(Interview).filter(Interview.source_path.isnot(None)).all()
            if r.source_path is not None
        }
    if processed:
        logger.info(f"Loaded {len(processed)} already-processed transcript path(s) from DB")

    # Discover any existing transcript files that were never stored in the DB
    # and ingest them immediately.
    existing = set(MEETILY_WATCH_DIR.rglob("transcripts.json"))
    unprocessed = existing - processed
    if unprocessed:
        logger.info(f"Found {len(unprocessed)} unprocessed transcript(s) — queuing for ingestion")
        for transcript_path in sorted(unprocessed):
            processed.add(transcript_path)
            try:
                await _ingest_meetily_transcript(transcript_path)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to ingest existing transcript {transcript_path}: {e}",
                    exc_info=True,
                )
    else:
        logger.info("No unprocessed transcripts found at startup")

    logger.info("Meetily startup scan complete — folder watching disabled")


def _estimate_meeting_start(segments: list[dict[str, Any]], fallback_path: Path | None = None) -> datetime:
    """
    Estimate when the meeting started by finding the earliest wall-clock
    ``timestamp`` value in the transcript segments.

    Falls back to the transcript file's mtime, then to utcnow().
    """
    _FORMATS = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ]

    earliest: datetime | None = None
    for seg in segments:
        raw = (seg.get("timestamp") or "").strip()
        if not raw:
            continue
        dt: datetime | None = None
        try:
            dt = datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            for fmt in _FORMATS:
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            continue
        if earliest is None or dt < earliest:
            earliest = dt

    if earliest is not None:
        return earliest

    # Fallback: file mtime
    if fallback_path is not None:
        try:
            return datetime.fromtimestamp(fallback_path.stat().st_mtime)
        except OSError:
            pass

    return datetime.utcnow()


def _meetily_transcript_to_text(segments: list[dict[str, Any]]) -> str:
    """
    Convert a list of Meetily transcript segment objects into a plain-text
    transcript string suitable for the analysis crew.

    Each segment is expected to have at least a ``text`` field.  The optional
    ``timestamp`` (wall-clock string) and ``audio_start_time`` (seconds) are
    used to annotate speaker turns when present.
    """
    lines: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        ts = seg.get("audio_start_time")
        if ts is not None:
            minutes, seconds = divmod(int(ts), 60)
            prefix = f"[{minutes:02d}:{seconds:02d}] "
        else:
            wall = seg.get("timestamp", "")
            prefix = f"[{wall}] " if wall else ""
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)


async def _ingest_meetily_transcript(transcript_path: Path) -> None:
    """
    Parse a finished transcripts.json and persist it as a pending interview.
    """
    try:
        raw = transcript_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"Cannot parse {transcript_path}: {e}")
        return

    # transcripts.json may be a bare list or wrapped in an object
    raw_segments: list[dict[str, object]]
    if isinstance(data, list):
        data_list = cast(list[object], data)
        raw_segments = [cast(dict[str, object], s) for s in data_list if isinstance(s, dict)]
    elif isinstance(data, dict):
        # Try common wrapper keys
        wrapped: list[object] = list(
            data.get("transcripts")
            or data.get("segments")
            or data.get("transcript")
            or []
        )
        raw_segments = [cast(dict[str, object], s) for s in wrapped if isinstance(s, dict)]
    else:
        logger.error(f"Unexpected transcripts.json shape in {transcript_path}")
        return

    # Filter out partial / low-confidence segments if flagged
    segments: list[dict[str, object]] = [s for s in raw_segments if not s.get("is_partial", False)]

    transcript_text = _meetily_transcript_to_text(segments)
    if not transcript_text:
        logger.warning(f"No usable text in {transcript_path} — skipping")
        return

    # Estimate meeting start time from transcript timestamps
    meeting_start = _estimate_meeting_start(segments, fallback_path=transcript_path)
    date_str = meeting_start.strftime("%Y-%m-%d")
    time_str = meeting_start.strftime("%H:%M")

    title = f"Interview - {date_str} {time_str}"
    role  = "Software Engineer"
    stage = "Recording"

    with Session(engine) as db:
        interview_id = str(uuid.uuid4())
        interview = Interview(
            id=interview_id,
            title=title,
            stage=stage,
            transcript=transcript_text,
            source_path=str(transcript_path),
            analysis_status="pending",
        )
        db.add(interview)
        db.commit()

    logger.info(
        f"Ingested meetily interview {interview_id} as pending ({len(transcript_text)} chars)"
    )


# ---- REST API --------------------------------------------------------------

@app.get("/status")
async def get_status() -> dict[str, Any]:
    llm_ready = _is_llm_ready()
    queue_depth = state.event_bus.analysis_queue.qsize() if state.event_bus else 0
    if os.getenv("OPENAI_API_KEY"):
        mode = "local"
        model = os.getenv("OPENAI_MODEL", "gpt-oss:20b")
    else:
        mode = "hosted"
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    return {
        "status": "running",
        "llm_mode": mode,
        "model": model,
        "llm_configured": llm_ready,
        "analysis_queue_depth": queue_depth,
        "currently_analyzing": state.currently_analyzing,
    }


def _is_llm_ready() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )


@app.get("/api/interviews")
async def list_interviews() -> list[dict[str, Any]]:
    with Session(engine) as db:
        interviews = db.query(Interview).order_by(Interview.created_at.desc()).all()
        return [i.to_dict() for i in interviews]


@app.patch("/api/interviews/{interview_id}")
async def rename_interview(interview_id: str, body: dict[str, Any]) -> dict[str, Any]:
    title = str(body.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "title is required")
    if len(title) > 200:
        raise HTTPException(400, "title must be 200 characters or fewer")
    with Session(engine) as db:
        interview = db.get(Interview, interview_id)
        if not interview:
            raise HTTPException(404, "Not found")
        interview.title = title
        db.commit()
    return {"ok": True}


@app.delete("/api/interviews/{interview_id}", status_code=204)
async def delete_interview(interview_id: str) -> None:
    with Session(engine) as db:
        interview = db.get(Interview, interview_id)
        if not interview:
            raise HTTPException(404, "Not found")
        db.delete(interview)
        db.commit()


@app.delete("/api/interviews/{interview_id}/qa/{qa_id}", status_code=200)
async def delete_qa_pair(interview_id: str, qa_id: str) -> dict[str, Any]:
    with Session(engine) as db:
        qa = db.get(QAPair, qa_id)
        if not qa or qa.interview_id != interview_id:
            raise HTTPException(404, "Not found")
        db.delete(qa)
        db.flush()

        # Recalculate overall_score from remaining QA pairs
        remaining = db.query(QAPair).filter(QAPair.interview_id == interview_id).all()
        scored = [p.score for p in remaining if p.score is not None]
        new_score = (sum(scored) / len(scored)) if scored else None

        # Recalculate potential_overall_score
        potential_scored: list[float] = [
            float(p.potential_score if p.potential_score is not None else p.score)  # type: ignore[arg-type]
            for p in remaining
            if (p.potential_score is not None or p.score is not None)
        ]
        potential_overall = (sum(potential_scored) / len(potential_scored)) if potential_scored else None
        has_potential = any(p.potential_score is not None for p in remaining)

        interview = db.get(Interview, interview_id)
        if interview:
            interview.overall_score = new_score

        db.commit()
    return {"ok": True, "overall_score": new_score, "potential_overall_score": potential_overall, "has_potential_scores": has_potential}


@app.post("/api/interviews/{interview_id}/qa/{qa_id}/regrade")
async def regrade_qa_pair(interview_id: str, qa_id: str, body: dict[str, Any]) -> dict[str, Any]:
    new_answer = str(body.get("new_answer") or "").strip()
    if not new_answer:
        raise HTTPException(400, "new_answer is required")
    if len(new_answer) > 10000:
        raise HTTPException(400, "new_answer must be 10,000 characters or fewer")

    with Session(engine) as db:
        qa = db.get(QAPair, qa_id)
        if not qa or qa.interview_id != interview_id:
            raise HTTPException(404, "Not found")
        question = qa.question
        category = qa.category or "general"

    if not _is_llm_ready():
        raise HTTPException(503, "No LLM configured — set OPENAI_API_KEY (local) or OPENROUTER_API_KEY (hosted)")

    regrade_queue = state.event_bus.regrade_queue if state.event_bus else None
    if regrade_queue is None:
        raise HTTPException(503, "Regrade queue not initialized")

    future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
    await regrade_queue.put(RegradeRequested(question=question, new_answer=new_answer, category=category, future=future))

    try:
        result = await future
    except Exception as e:
        logger.error(f"Regrade failed for qa {qa_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Grading failed: {e}")

    potential_score = result.get("score")
    feedback = result.get("feedback")
    suggested_answer = result.get("suggested_answer")

    with Session(engine) as db:
        qa = db.get(QAPair, qa_id)
        if not qa or qa.interview_id != interview_id:
            raise HTTPException(404, "Not found")
        qa.potential_answer = new_answer
        qa.potential_score = potential_score
        qa.potential_feedback = feedback
        qa.potential_suggested_answer = suggested_answer
        db.flush()

        all_qa = db.query(QAPair).filter(QAPair.interview_id == interview_id).all()
        potential_scored: list[float] = [
            float(p.potential_score if p.potential_score is not None else p.score)  # type: ignore[arg-type]
            for p in all_qa
            if (p.potential_score is not None or p.score is not None)
        ]
        potential_overall = (sum(potential_scored) / len(potential_scored)) if potential_scored else None

        db.commit()

    return {
        "potential_score": potential_score,
        "feedback": feedback,
        "suggested_answer": suggested_answer,
        "potential_overall_score": potential_overall,
    }


@app.get("/api/interviews/{interview_id}")
async def get_interview(interview_id: str) -> dict[str, Any]:
    with Session(engine) as db:
        i = db.get(Interview, interview_id)
        if not i:
            raise HTTPException(404, "Not found")
        data = i.to_dict()
        data["qa_pairs"] = [q.to_dict() for q in i.qa_pairs]
        return data


@app.post("/api/interviews")
async def create_interview(body: dict[str, Any]) -> dict[str, Any]:
    """
    Create an interview record.

    Optional fields: role, stage, title, transcript, start_time, end_time, duration_seconds
    """
    transcript = str(body.get("transcript") or "")
    title = str(body.get("title") or "").strip()
    role  = str(body.get("role") or "Software Engineer")
    stage = str(body.get("stage") or "Phone Screen")

    interview_id = str(uuid.uuid4())
    with Session(engine) as db:
        interview = Interview(
            id=interview_id,
            title=title or f"Interview — {stage}",
            stage=stage,
            transcript=transcript,
            start_time=datetime.fromisoformat(str(body["start_time"])) if body.get("start_time") else None,
            end_time=datetime.fromisoformat(str(body["end_time"])) if body.get("end_time") else None,
            duration_seconds=int(body["duration_seconds"]) if body.get("duration_seconds") is not None else None,
            analysis_status="pending",
        )
        db.add(interview)
        db.commit()

    return {"interview_id": interview_id, "analysis_status": "pending"}


@app.post("/api/interviews/{interview_id}/analyze")
async def trigger_interview_analysis(interview_id: str) -> dict[str, Any]:
    if not _is_llm_ready():
        raise HTTPException(503, "No LLM configured — set OPENAI_API_KEY (local) or HF_TOKEN (hosted)")
    if state.event_bus is None:
        raise HTTPException(503, "Event bus not initialized")

    with Session(engine) as db:
        interview = db.get(Interview, interview_id)
        if not interview:
            raise HTTPException(404, "Not found")
        transcript = (interview.transcript or "").strip()
        if not transcript:
            raise HTTPException(400, "Interview has no transcript to analyze")
        if interview.analysis_status == "queued":
            return {"interview_id": interview_id, "analysis_status": "queued"}

        if interview.analysis_status == "complete":
            raise HTTPException(409, "Interview analysis is already complete")

        interview.analysis_status = "queued"
        interview.analysis_error = None
        db.commit()

        title = interview.title or interview.stage or "Interview"
        role = "Software Engineer"
        stage = interview.stage or "Interview"

    try:
        state.event_bus.analysis_queue.put_nowait(
            AnalysisRequested(
                interview_id=interview_id,
                transcript=transcript,
                company_name=title,
                role=role,
                stage=stage,
            )
        )
    except asyncio.QueueFull:
        _mark_analysis_result(interview_id, status="failed", error="Analysis queue full")
        raise HTTPException(503, "Analysis queue full — try again later")

    logger.info(f"Queued interview {interview_id} for manual analysis trigger")
    return {"interview_id": interview_id, "analysis_status": "queued"}


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard() -> HTMLResponse:
    path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    if os.path.exists(path):
        with open(path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run("server:app", host=host, port=port, reload=True)
