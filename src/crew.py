"""
InterviewRetroCrew — top-level src/crew.py shim.

Imports the canonical class from interview_retro.crew so any code that
does `from crew import InterviewRetroCrew` (with src/ on sys.path) still works.
"""
from interview_retro.crew import InterviewRetroCrew

__all__ = ["InterviewRetroCrew"]
