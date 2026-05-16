"""
Database Models
SQLite stored in ~/Library/Mobile Documents/com~apple~CloudDocs/JoBound/interview-retro/interviews.db
"""
import os
from datetime import datetime
from typing import Any, Optional
from sqlalchemy import (
    create_engine, String, Integer, Float,
    DateTime, Text, ForeignKey, JSON, text
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column


# ─── Local database path ─────────────────────────────────────────────────────
DB_DIR = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/JoBound/interview-retro")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "interviews.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class Base(DeclarativeBase):
    pass


# ─── Models ─────────────────────────────────────────────────────────────────

class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)                       # e.g. "Phone Screen", "Technical Round 2"
    stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)                       # Interview stage
    transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                    # Full interview transcript
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)                 # absolute path to originating transcript file
    analysis_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)             # pending, queued, complete, skipped, failed
    analysis_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                # Why analysis was skipped or failed
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # 0-10 rating
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                       # AI-generated summary
    strengths: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)                    # List of strengths
    weaknesses: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)                   # List of weaknesses
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)

    qa_pairs: Mapped[list["QAPair"]] = relationship("QAPair", back_populates="interview", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        # Compute potential overall score (uses potential_score if set, else original score)
        potential_scored: list[float] = [
            float(p.potential_score if p.potential_score is not None else p.score)  # type: ignore[arg-type]
            for p in self.qa_pairs
            if (p.potential_score is not None or p.score is not None)
        ]
        potential_overall = (sum(potential_scored) / len(potential_scored)) if potential_scored else None
        has_potential = any(p.potential_score is not None for p in self.qa_pairs)
        return {
            "id": self.id,
            "title": self.title,
            "stage": self.stage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "analysis_status": self.analysis_status,
            "analysis_error": self.analysis_error,
            "transcript_char_count": len(self.transcript or ""),
            "overall_score": self.overall_score,
            "summary": self.summary,
            "strengths": self.strengths or [],
            "weaknesses": self.weaknesses or [],
            "qa_count": len(self.qa_pairs),
            "potential_overall_score": potential_overall,
            "has_potential_scores": has_potential,
        }


class QAPair(Base):
    __tablename__ = "qa_pairs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    interview_id: Mapped[str] = mapped_column(String, ForeignKey("interviews.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                      # 0-10 rating for this specific answer
    category: Mapped[Optional[str]] = mapped_column(String, nullable=True)                    # behavioral, technical, situational, etc.
    feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                      # What was good/bad about the answer
    suggested_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)              # Better answer suggestion (only if score < 6)
    timestamp_in_meeting: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)       # Seconds from start
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    potential_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)              # New answer the user tried
    potential_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # Score for the potential answer
    potential_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)            # Feedback for the potential answer
    potential_suggested_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)    # Better answer for potential (if score < 6)

    interview: Mapped["Interview"] = relationship("Interview", back_populates="qa_pairs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "interview_id": self.interview_id,
            "question": self.question,
            "answer": self.answer,
            "score": self.score,
            "category": self.category,
            "feedback": self.feedback,
            "suggested_answer": self.suggested_answer if (self.score or 10) < 6 else None,
            "timestamp_in_meeting": self.timestamp_in_meeting,
            "potential_answer": self.potential_answer,
            "potential_score": self.potential_score,
            "potential_feedback": self.potential_feedback,
            "potential_suggested_answer": self.potential_suggested_answer if (self.potential_score or 10) < 6 else None,
        }


class Settings(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)


# ─── Init ───────────────────────────────────────────────────────────────────

def _pre_migrate() -> None:
    """
    One-time migration: drop the companies table and rebuild interviews
    without the company_id column (which had a NOT NULL FK constraint).
    Safe to call on every startup — exits immediately if already migrated.
    """
    with engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))}
        if "companies" not in tables:
            return  # already migrated or fresh install

        conn.execute(text("PRAGMA foreign_keys = OFF"))

        # Rebuild interviews without company_id
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(interviews)"))}
        if "company_id" in cols:
            conn.execute(text("""
                CREATE TABLE interviews_new (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    stage TEXT,
                    transcript TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration_seconds INTEGER,
                    analysis_status TEXT,
                    analysis_error TEXT,
                    overall_score FLOAT,
                    summary TEXT,
                    strengths JSON,
                    weaknesses JSON,
                    created_at DATETIME
                )
            """))
            conn.execute(text("""
                INSERT INTO interviews_new
                SELECT id, title, stage, transcript, start_time, end_time,
                       duration_seconds, analysis_status, analysis_error,
                       overall_score, summary, strengths, weaknesses, created_at
                FROM interviews
            """))
            conn.execute(text("DROP TABLE interviews"))
            conn.execute(text("ALTER TABLE interviews_new RENAME TO interviews"))

        conn.execute(text("DROP TABLE IF EXISTS companies"))
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()


def _add_missing_columns() -> None:
    """
    Additive migrations: add any columns that are present in the ORM models
    but missing from the live SQLite tables.  Safe to call on every startup.
    """
    additions = [
        ("interviews", "source_path", "TEXT"),
        ("qa_pairs", "potential_answer", "TEXT"),
        ("qa_pairs", "potential_score", "FLOAT"),
        ("qa_pairs", "potential_feedback", "TEXT"),
        ("qa_pairs", "potential_suggested_answer", "TEXT"),
    ]
    with engine.connect() as conn:
        for table, column, col_type in additions:
            existing = {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})"))}
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                conn.commit()


def init_db() -> None:
    _pre_migrate()
    Base.metadata.create_all(engine)
    _add_missing_columns()
    print(f"[DB] Database initialized at: {DB_PATH}")
