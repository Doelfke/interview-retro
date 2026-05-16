"""
Interview Retro — entry point for crewai run / crewai deploy.

Required inputs (set as env vars or pass via crewai deploy API):
  TRANSCRIPT    — raw interview transcript text
  COMPANY_NAME  — company being interviewed at  (default: "Unknown Company")
  ROLE          — job role / position title     (default: "Unknown Role")
  STAGE         — interview stage               (default: "interview")
"""
import os

from interview_retro.crew import InterviewRetroCrew


def run() -> None:
    inputs = {
        "transcript":   os.environ.get("TRANSCRIPT", ""),
        "company_name": os.environ.get("COMPANY_NAME", "Unknown Company"),
        "role":         os.environ.get("ROLE", "Unknown Role"),
        "stage":        os.environ.get("STAGE", "interview"),
    }
    result = InterviewRetroCrew().crew().kickoff(inputs=inputs)
    print(result)


if __name__ == "__main__":
    run()
