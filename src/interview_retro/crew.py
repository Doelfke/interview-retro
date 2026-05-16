"""
InterviewRetroCrew — @CrewBase entry point for the interview_retro package.

@CrewBase resolves config paths relative to this file, so the YAML files are at:
  src/interview_retro/config/agents.yaml
  src/interview_retro/config/tasks.yaml
"""
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from interview_retro.llm import make_llm


@CrewBase
class InterviewRetroCrew:
    """Interview retrospective analysis crew."""

    agents_config: dict[str, Any] = "config/agents.yaml"  # type: ignore[assignment]
    tasks_config: dict[str, Any] = "config/tasks.yaml"  # type: ignore[assignment]

    @agent
    def transcription_agent(self) -> Agent:
        return Agent(config=self.agents_config["transcription_agent"], verbose=True, llm=make_llm())

    @agent
    def qa_extractor_agent(self) -> Agent:
        return Agent(config=self.agents_config["qa_extractor_agent"], verbose=True, llm=make_llm())

    @agent
    def advocate_agent(self) -> Agent:
        return Agent(config=self.agents_config["advocate_agent"], verbose=True, llm=make_llm())

    @agent
    def critic_agent(self) -> Agent:
        return Agent(config=self.agents_config["critic_agent"], verbose=True, llm=make_llm())

    @agent
    def judge_agent(self) -> Agent:
        return Agent(config=self.agents_config["judge_agent"], verbose=True, llm=make_llm())

    @task
    def transcription_task(self) -> Task:
        return Task(config=self.tasks_config["transcription_task"])  # type: ignore[call-arg]

    @task
    def qa_extraction_task(self) -> Task:
        return Task(config=self.tasks_config["qa_extraction_task"])  # type: ignore[call-arg]

    @task
    def advocate_task(self) -> Task:
        return Task(config=self.tasks_config["advocate_task"])  # type: ignore[call-arg]

    @task
    def critic_task(self) -> Task:
        return Task(config=self.tasks_config["critic_task"])  # type: ignore[call-arg]

    @task
    def judge_task(self) -> Task:
        return Task(config=self.tasks_config["judge_task"])  # type: ignore[call-arg]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # type: ignore[attr-defined]
            tasks=self.tasks,  # type: ignore[attr-defined]
            process=Process.sequential,
            verbose=True,
        )


__all__ = ["InterviewRetroCrew"]
