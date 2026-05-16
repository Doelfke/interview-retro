# Deployment compatibility shim.
#
# CrewAI's deploy system expects the crew entry point at src/crew.py and
# the config files at src/config/.  The actual implementation lives in the
# interview_retro package (src/interview_retro/crew.py), which is where
# @CrewBase resolves its config paths.  This shim re-exports the class so
# the deployment pre-flight checks pass and the crew is importable from here.

from interview_retro.crew import InterviewRetroCrew

__all__ = ["InterviewRetroCrew"]
