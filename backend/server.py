"""
Interview Retro Backend
FastAPI server. All AI runs locally via MLX (mlx-lm).

On startup the lifespan manager launches mlx_lm.server as a subprocess,
exposes an OpenAI-compatible endpoint on localhost:8081, and shuts it down
cleanly when the backend exits.
"""
import asyncio
import json
import logging
import os
import uuid
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from storage.models import Interview, QAPair, engine, init_db
from crews.crews import InterviewAnalysisCrew

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---- App state -------------------------------------------------------------

ANALYSIS_QUEUE_MAX = 20  # warn if more than this many analyses are pending


class AppState:
    analysis_queue: Optional[asyncio.Queue] = None
    analysis_worker_task: Optional[asyncio.Task] = None
    meetily_watcher_task: Optional[asyncio.Task] = None
    mlx_server_proc: Optional[asyncio.subprocess.Process] = None
    currently_analyzing: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    mlx_model = os.getenv("MLX_MODEL", "mlx-community/Qwen2.5-32B-Instruct-4bit")
    mlx_port  = int(os.getenv("MLX_SERVER_PORT", "8081"))
    mlx_host  = os.getenv("MLX_SERVER_HOST", "127.0.0.1")
    mlx_ctx   = int(os.getenv("MLX_CONTEXT_LENGTH", "8192"))

    logger.info("=" * 60)
    logger.info("  Interview Retro — fully local via Apple MLX")
    logger.info(f"  LLM : {mlx_model}")
    logger.info("=" * 60)

    state.mlx_server_proc = await _start_mlx_server(mlx_model, mlx_host, mlx_port, mlx_ctx)

    state.analysis_queue = asyncio.Queue(maxsize=ANALYSIS_QUEUE_MAX)
    state.analysis_worker_task = asyncio.create_task(_analysis_worker())
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
    if state.analysis_worker_task:
        state.analysis_worker_task.cancel()
        try:
            await state.analysis_worker_task
        except asyncio.CancelledError:
            pass
    if state.mlx_server_proc:
        logger.info("Stopping mlx_lm.server...")
        state.mlx_server_proc.terminate()
        try:
            await asyncio.wait_for(state.mlx_server_proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            state.mlx_server_proc.kill()


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


async def _start_mlx_server(
    model: str, host: str, port: int, ctx: int
) -> asyncio.subprocess.Process:
    """
    Launch mlx_lm.server as a subprocess and wait until it is accepting
    connections before returning.
    """
    import sys
    import httpx

    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model",      model,
        "--host",       host,
        "--port",       str(port),
        "--max-tokens", str(ctx),
    ]
    logger.info(f"Starting mlx_lm.server: {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    asyncio.create_task(_log_subprocess(proc, "mlx_lm.server"))

    # Wait up to 600 s — a cold first run can download ~18 GB
    url = f"http://{host}:{port}/v1/models"
    for attempt in range(600):
        await asyncio.sleep(1)
        if proc.returncode is not None:
            raise RuntimeError(
                f"mlx_lm.server exited unexpectedly (code {proc.returncode}). "
                "Check that the MLX_MODEL path is correct."
            )
        try:
            async with httpx.AsyncClient(timeout=1) as c:
                r = await c.get(url)
                if r.status_code == 200:
                    logger.info(f"mlx_lm.server ready on {host}:{port} after {attempt + 1}s")
                    return proc
        except Exception:
            pass

    raise TimeoutError("mlx_lm.server did not become ready within 600 s — check logs")


async def _log_subprocess(proc: asyncio.subprocess.Process, name: str) -> None:
    if proc.stdout is None:
        return
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        logger.debug(f"[{name}] {line.decode(errors='replace').rstrip()}")


app = FastAPI(title="Interview Retro (local)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Analysis worker -------------------------------------------------------

async def _analysis_worker() -> None:
    """Single worker that drains the analysis queue one job at a time."""
    logger.info("Analysis worker started")
    analysis_queue = state.analysis_queue
    if analysis_queue is None:
        raise RuntimeError("Analysis queue not initialized")

    while True:
        interview_id, transcript, company_name, role, stage = await analysis_queue.get()
        state.currently_analyzing = True
        try:
            remaining = analysis_queue.qsize()
            logger.info(f"Starting analysis {interview_id} ({remaining} more in queue)")
            await _analyze_interview(interview_id, transcript, company_name, role, stage)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Analysis worker error for {interview_id}: {e}", exc_info=True)
        finally:
            state.currently_analyzing = False
            analysis_queue.task_done()


async def _analyze_interview(
    interview_id: str, transcript: str, company_name: str, role: str, stage: str
) -> None:
    if not transcript or len(transcript) < 100:
        reason = "Transcript too short to analyze" if transcript else "No transcript provided"
        logger.warning(f"Skipping analysis for interview {interview_id}: {reason}")
        await asyncio.to_thread(_mark_analysis_result, interview_id, status="skipped", error=reason)
        return

    logger.info(f"Starting CrewAI analysis: {company_name} / {stage}")
    try:
        crew = InterviewAnalysisCrew()
        result = await asyncio.to_thread(
            crew.run,
            transcript=transcript,
            company_name=company_name,
            role=role,
            stage=stage,
        )
        await asyncio.to_thread(_update_analysis, interview_id, result)
        if result.get("analysis_status") == "failed":
            logger.error(f"Analysis failed for {company_name}: {result.get('analysis_error')}")
        else:
            logger.info(
                f"Analysis complete for {company_name} — {result.get('overall_score')}/10"
            )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        await asyncio.to_thread(_mark_analysis_result, interview_id, status="failed", error=str(e))


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


def _update_analysis(interview_id: str, analysis: dict) -> None:
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

        for qa in analysis.get("rated_qa", []):
            db.add(QAPair(
                id=str(uuid.uuid4()),
                interview_id=interview_id,
                question=qa.get("question", ""),
                answer=qa.get("answer", ""),
                score=qa.get("score"),
                category=qa.get("category", "general"),
                feedback=qa.get("feedback"),
                suggested_answer=qa.get("suggested_answer"),
                timestamp_in_meeting=qa.get("timestamp_seconds"),
            ))

        db.commit()


# ---- Meetily folder watcher ------------------------------------------------

MEETILY_WATCH_DIR = Path.home() / "Movies" / "meetily-recordings"


async def _meetily_watcher() -> None:
    """
    On startup, scan ~/Movies/meetily-recordings/ for transcripts.json files
    that have not yet been ingested and queue them for analysis.
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
    # and queue them immediately.
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


def _estimate_meeting_start(segments: list[dict], fallback_path: Path | None = None) -> datetime:
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


def _meetily_transcript_to_text(segments: list[dict]) -> str:
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
    Parse a finished transcripts.json and push the recording into the analysis
    pipeline, creating a Company + Interview record as needed.
    """
    try:
        raw = transcript_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"Cannot parse {transcript_path}: {e}")
        return

    # transcripts.json may be a bare list or wrapped in an object
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict):
        # Try common wrapper keys
        segments = (
            data.get("transcripts")
            or data.get("segments")
            or data.get("transcript")
            or []
        )
    else:
        logger.error(f"Unexpected transcripts.json shape in {transcript_path}")
        return

    # Filter out partial / low-confidence segments if flagged
    segments = [s for s in segments if not s.get("is_partial", False)]

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

    analysis_queue = state.analysis_queue
    if analysis_queue is None:
        logger.error("Analysis queue not ready — cannot ingest meetily transcript")
        return

    with Session(engine) as db:
        interview_id = str(uuid.uuid4())
        interview = Interview(
            id=interview_id,
            title=title,
            stage=stage,
            transcript=transcript_text,
            source_path=str(transcript_path),
            analysis_status="queued",
        )
        db.add(interview)
        db.commit()

    try:
        analysis_queue.put_nowait(
            (interview_id, transcript_text, title, role, stage)
        )
        logger.info(f"Queued meetily interview {interview_id} for analysis ({len(transcript_text)} chars)")
    except asyncio.QueueFull:
        _mark_analysis_result(interview_id, status="failed", error="Analysis queue full")
        logger.error("Analysis queue full — meetily transcript was saved but not analyzed")


# ---- REST API --------------------------------------------------------------

@app.get("/status")
async def get_status():
    mlx_ok = await _check_mlx_server()
    queue_depth = state.analysis_queue.qsize() if state.analysis_queue else 0
    return {
        "status": "running",
        "mlx_model": os.getenv("MLX_MODEL", "mlx-community/Qwen2.5-32B-Instruct-4bit"),
        "mlx_server_ok": mlx_ok,
        "analysis_queue_depth": queue_depth,
        "currently_analyzing": state.currently_analyzing,
    }


async def _check_mlx_server() -> bool:
    import httpx
    port = os.getenv("MLX_SERVER_PORT", "8081")
    host = os.getenv("MLX_SERVER_HOST", "127.0.0.1")
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"http://{host}:{port}/v1/models")
            return r.status_code == 200
    except Exception:
        return False


@app.get("/api/interviews")
async def list_interviews():
    with Session(engine) as db:
        interviews = db.query(Interview).order_by(Interview.created_at.desc()).all()
        return [i.to_dict() for i in interviews]


@app.patch("/api/interviews/{interview_id}")
async def rename_interview(interview_id: str, body: dict):
    title = (body.get("title") or "").strip()
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
async def delete_interview(interview_id: str):
    with Session(engine) as db:
        interview = db.get(Interview, interview_id)
        if not interview:
            raise HTTPException(404, "Not found")
        db.delete(interview)
        db.commit()


@app.delete("/api/interviews/{interview_id}/qa/{qa_id}", status_code=200)
async def delete_qa_pair(interview_id: str, qa_id: str):
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


def _regrade_answer_sync(question: str, new_answer: str, category: str) -> dict:
    """
    Grade a single interview answer using the same advocate → critic → judge
    crew pipeline used for full interview analysis.
    Returns score, feedback, and optional suggested_answer.
    """
    crew = InterviewAnalysisCrew()
    return crew.regrade_answer(question=question, new_answer=new_answer, category=category)


@app.post("/api/interviews/{interview_id}/qa/{qa_id}/regrade")
async def regrade_qa_pair(interview_id: str, qa_id: str, body: dict):
    new_answer = (body.get("new_answer") or "").strip()
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

    mlx_ok = await _check_mlx_server()
    if not mlx_ok:
        raise HTTPException(503, "AI server is not available — please wait for it to finish loading")

    try:
        result = await asyncio.to_thread(_regrade_answer_sync, question, new_answer, category)
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
async def get_interview(interview_id: str):
    with Session(engine) as db:
        i = db.get(Interview, interview_id)
        if not i:
            raise HTTPException(404, "Not found")
        data = i.to_dict()
        data["qa_pairs"] = [q.to_dict() for q in i.qa_pairs]
        return data


@app.post("/api/interviews")
async def create_interview(body: dict):
    """
    Create an interview record and queue it for AI analysis.

    Optional fields: role, stage, title, transcript, start_time, end_time, duration_seconds
    """
    transcript = body.get("transcript", "")
    title = (body.get("title") or "").strip()
    role  = body.get("role", "Software Engineer")
    stage = body.get("stage", "Phone Screen")

    interview_id = str(uuid.uuid4())
    with Session(engine) as db:
        interview = Interview(
            id=interview_id,
            title=title or f"Interview — {stage}",
            stage=stage,
            transcript=transcript,
            start_time=datetime.fromisoformat(body["start_time"]) if body.get("start_time") else None,
            end_time=datetime.fromisoformat(body["end_time"]) if body.get("end_time") else None,
            duration_seconds=body.get("duration_seconds"),
            analysis_status="queued" if transcript else "pending",
        )
        db.add(interview)
        db.commit()

    if transcript:
        analysis_queue = state.analysis_queue
        if analysis_queue is None:
            raise HTTPException(503, "Analysis queue not initialized")
        try:
            analysis_queue.put_nowait((interview_id, transcript, title or stage, role, stage))
        except asyncio.QueueFull:
            _mark_analysis_result(interview_id, status="failed", error="Analysis queue full")
            raise HTTPException(503, "Analysis queue full — try again later")

    return {"interview_id": interview_id, "analysis_status": "queued" if transcript else "pending"}


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run("server:app", host=host, port=port, reload=True)
