# Re-export shim — the canonical definition lives in src/crew.py so that
# crewai's deploy validator (which scans that file) finds @CrewBase.
from crew import InterviewRetroCrew

__all__ = ["InterviewRetroCrew"]
