"""
InterviewRetroCrew — canonical @CrewBase entry point.

CrewAI's deploy system statically scans src/crew.py for a @CrewBase-decorated
class.  This file IS the definition; src/interview_retro/crew.py re-exports it.

@CrewBase resolves config paths relative to the file that defines the class
(this file lives at src/crew.py) so the YAML files must be at:
  src/interview_retro/config/agents.yaml
  src/interview_retro/config/tasks.yaml
"""
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class InterviewRetroCrew:
    """Interview retrospective analysis crew."""

    agents_config: dict[str, Any] = "config/agents.yaml"  # type: ignore[assignment]
    tasks_config: dict[str, Any] = "config/tasks.yaml"  # type: ignore[assignment]

    @agent
    def transcription_agent(self) -> Agent:
        return Agent(config=self.agents_config["transcription_agent"], verbose=True)

    @agent
    def qa_extractor_agent(self) -> Agent:
        return Agent(config=self.agents_config["qa_extractor_agent"], verbose=True)

    @agent
    def advocate_agent(self) -> Agent:
        return Agent(config=self.agents_config["advocate_agent"], verbose=True)

    @agent
    def critic_agent(self) -> Agent:
        return Agent(config=self.agents_config["critic_agent"], verbose=True)

    @agent
    def judge_agent(self) -> Agent:
        return Agent(config=self.agents_config["judge_agent"], verbose=True)

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
