from typing import Any

from crewai.flow.flow import Flow, start

from interview_retro.crew import InterviewRetroCrew


class InterviewRetroFlow(Flow[dict[str, Any]]):
    @start()
    def run_crew(self) -> object:
        return InterviewRetroCrew().crew().kickoff()


def kickoff() -> None:
    flow = InterviewRetroFlow()
    flow.kickoff()


def plot() -> None:
    flow = InterviewRetroFlow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
