"""
InterviewAnalysisCrew — imperative crew runner for the FastAPI backend.

Processes an interview transcript through a 5-agent debate pipeline:
  1. Transcription  — cleans and speaker-labels the raw transcript
  2. Q&A extraction — pulls every question + answer pair
  3. Advocate       — argues the strongest case FOR each answer
  4. Critic         — reads the advocate's case and argues AGAINST it
  5. Judge          — weighs both arguments, scores, and gives a verdict

This class accepts the transcript and interview metadata at call time,
interpolates them into the task descriptions, and returns structured JSON.
"""
import json
import re
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, LLM, Process, Task

from interview_retro.llm import make_llm

_CONFIG_DIR = Path(__file__).parent / "config"
_AGENTS_YAML = _CONFIG_DIR / "agents.yaml"
_TASKS_YAML = _CONFIG_DIR / "tasks.yaml"


def _load_task_descs() -> dict[str, str]:
    with open(_TASKS_YAML) as f:
        cfg = yaml.safe_load(f)
    return {key: cfg[key]["description"] for key in cfg}


_TASK_DESCS = _load_task_descs()


def _strip_fence(text: str) -> str:
    """Remove ```json ... ``` fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _parse_list(raw: str, key: str) -> list[dict[str, object]]:
    """Safely parse a JSON array from a raw task output string."""
    try:
        return json.loads(_strip_fence(raw)).get(key, [])  # type: ignore[no-any-return]
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def _make_agents(llm: LLM) -> dict[str, Agent]:
    with open(_AGENTS_YAML) as f:
        config = yaml.safe_load(f)

    def _agent(key: str) -> Agent:
        cfg = config[key]
        return Agent(
            role=cfg["role"].strip(),
            goal=cfg["goal"].strip(),
            backstory=cfg["backstory"].strip(),
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    return {
        "transcription": _agent("transcription_agent"),
        "qa_extractor":  _agent("qa_extractor_agent"),
        "advocate":      _agent("advocate_agent"),
        "critic":        _agent("critic_agent"),
        "judge":         _agent("judge_agent"),
    }


class InterviewAnalysisCrew:
    """Runs a full interview analysis or single-answer regrade."""

    def __init__(self, llm: LLM | None = None) -> None:
        self.agents = _make_agents(llm or make_llm())

    def run(
        self,
        transcript: str,
        company_name: str,
        role: str,
        stage: str,
        enrichment: str | None = None,
    ) -> dict[str, Any]:
        agents = self.agents

        transcription_desc = _TASK_DESCS["transcription_task"].format(company_name=company_name)
        if enrichment:
            transcription_desc = enrichment + transcription_desc

        task_structure = Task(
            description=transcription_desc + f"\n\nTranscript:\n{transcript}",
            agent=agents["transcription"],
            expected_output="JSON with structured_transcript array",
        )

        task_extract_qa = Task(
            description=_TASK_DESCS["qa_extraction_task"].format(
                company_name=company_name, role=role, stage=stage
            ),
            agent=agents["qa_extractor"],
            expected_output="JSON with qa_pairs array",
            context=[task_structure],
        )

        task_advocate = Task(
            description=_TASK_DESCS["advocate_task"].format(
                company_name=company_name, stage=stage, role=role
            ),
            agent=agents["advocate"],
            expected_output="JSON with advocacy array — one entry per Q&A pair",
            context=[task_extract_qa],
        )

        task_critic = Task(
            description=_TASK_DESCS["critic_task"].format(
                company_name=company_name, stage=stage
            ),
            agent=agents["critic"],
            expected_output="JSON with criticism array — one entry per Q&A pair",
            context=[task_extract_qa, task_advocate],
        )

        task_judge = Task(
            description=_TASK_DESCS["judge_task"].format(
                company_name=company_name, stage=stage, role=role
            ),
            agent=agents["judge"],
            expected_output="JSON with rated_qa, overall_score, strengths, weaknesses, summary",
            context=[task_extract_qa, task_advocate, task_critic],
        )

        crew = Crew(
            agents=[
                agents["transcription"],
                agents["qa_extractor"],
                agents["advocate"],
                agents["critic"],
                agents["judge"],
            ],
            tasks=[task_structure, task_extract_qa, task_advocate, task_critic, task_judge],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        qa_pairs = _parse_list(
            task_extract_qa.output.raw if task_extract_qa.output else "", "qa_pairs"
        )
        advocacy = _parse_list(
            task_advocate.output.raw if task_advocate.output else "", "advocacy"
        )
        criticism = _parse_list(
            task_critic.output.raw if task_critic.output else "", "criticism"
        )

        try:
            parsed: dict[str, Any] = json.loads(_strip_fence(str(result)))
            parsed.setdefault("analysis_status", "complete")
            parsed.setdefault("analysis_error", None)
        except json.JSONDecodeError:
            parsed = {
                "rated_qa": [],
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "summary": str(result),
                "raw_output": str(result),
                "analysis_status": "failed",
                "analysis_error": "Crew output was not valid JSON",
            }

        parsed["qa_pairs_extracted"] = len(qa_pairs)
        parsed["_checkpoints"] = {
            "qa_pairs": qa_pairs,
            "advocacy": advocacy,
            "criticism": criticism,
        }
        return parsed

    def run_from_judge(
        self,
        qa_pairs: list[dict[str, Any]],
        advocacy: list[dict[str, Any]],
        criticism: list[dict[str, Any]],
        company_name: str,
        role: str,
        stage: str,
        enrichment: str | None = None,
    ) -> dict[str, Any]:
        """Run only the judge task with all context embedded in the description."""
        agents = self.agents

        judge_desc = (
            (enrichment or "")
            + _TASK_DESCS["judge_task"].format(
                company_name=company_name, stage=stage, role=role
            )
            + f"\n\nQ&A Pairs:\n{json.dumps(qa_pairs, indent=2)}"
            + f"\n\nAdvocacy:\n{json.dumps(advocacy, indent=2)}"
            + f"\n\nCriticism:\n{json.dumps(criticism, indent=2)}"
        )

        task_judge = Task(
            description=judge_desc,
            agent=agents["judge"],
            expected_output="JSON with rated_qa, overall_score, strengths, weaknesses, summary",
        )

        crew = Crew(
            agents=[agents["judge"]],
            tasks=[task_judge],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        try:
            parsed: dict[str, Any] = json.loads(_strip_fence(str(result)))
            parsed.setdefault("analysis_status", "complete")
            parsed.setdefault("analysis_error", None)
        except json.JSONDecodeError:
            parsed = {
                "rated_qa": [],
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "summary": str(result),
                "analysis_status": "failed",
                "analysis_error": "Judge output was not valid JSON",
            }

        parsed["qa_pairs_extracted"] = len(qa_pairs)
        parsed["_checkpoints"] = {
            "qa_pairs": qa_pairs,
            "advocacy": advocacy,
            "criticism": criticism,
        }
        return parsed

    def run_from_debate(
        self,
        qa_pairs: list[dict[str, Any]],
        company_name: str,
        role: str,
        stage: str,
        enrichment: str | None = None,
    ) -> dict[str, Any]:
        """Run advocate + critic + judge with qa_pairs embedded in the advocate description."""
        agents = self.agents

        advocate_desc = (
            (enrichment or "")
            + _TASK_DESCS["advocate_task"].format(
                company_name=company_name, stage=stage, role=role
            )
            + f"\n\nQ&A Pairs to evaluate:\n{json.dumps(qa_pairs, indent=2)}"
        )

        task_advocate = Task(
            description=advocate_desc,
            agent=agents["advocate"],
            expected_output="JSON with advocacy array — one entry per Q&A pair",
        )

        task_critic = Task(
            description=_TASK_DESCS["critic_task"].format(
                company_name=company_name, stage=stage
            ),
            agent=agents["critic"],
            expected_output="JSON with criticism array — one entry per Q&A pair",
            context=[task_advocate],
        )

        task_judge = Task(
            description=_TASK_DESCS["judge_task"].format(
                company_name=company_name, stage=stage, role=role
            ),
            agent=agents["judge"],
            expected_output="JSON with rated_qa, overall_score, strengths, weaknesses, summary",
            context=[task_advocate, task_critic],
        )

        crew = Crew(
            agents=[agents["advocate"], agents["critic"], agents["judge"]],
            tasks=[task_advocate, task_critic, task_judge],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        new_advocacy = _parse_list(
            task_advocate.output.raw if task_advocate.output else "", "advocacy"
        )
        new_criticism = _parse_list(
            task_critic.output.raw if task_critic.output else "", "criticism"
        )

        try:
            parsed: dict[str, Any] = json.loads(_strip_fence(str(result)))
            parsed.setdefault("analysis_status", "complete")
            parsed.setdefault("analysis_error", None)
        except json.JSONDecodeError:
            parsed = {
                "rated_qa": [],
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "summary": str(result),
                "analysis_status": "failed",
                "analysis_error": "Debate output was not valid JSON",
            }

        parsed["qa_pairs_extracted"] = len(qa_pairs)
        parsed["_checkpoints"] = {
            "qa_pairs": qa_pairs,
            "advocacy": new_advocacy,
            "criticism": new_criticism,
        }
        return parsed

    def regrade_answer(
        self,
        question: str,
        new_answer: str,
        category: str,
        company_name: str = "this company",
        stage: str = "the interview",
        role: str = "this role",
    ) -> dict[str, Any]:
        """Run a single Q&A pair through the advocate → critic → judge pipeline."""
        agents = self.agents

        task_present_qa = Task(
            description=f"""
            Present the following interview Q&A pair for debate evaluation.

            Return ONLY this JSON:
            {{
              "qa_pairs": [
                {{
                  "question": {json.dumps(question)},
                  "answer": {json.dumps(new_answer)},
                  "category": {json.dumps(category)},
                  "timestamp_seconds": 0
                }}
              ]
            }}
            """,
            agent=agents["qa_extractor"],
            expected_output="JSON with qa_pairs array containing one entry",
        )

        task_advocate = Task(
            description=_TASK_DESCS["advocate_task"].format(
                company_name=company_name, stage=stage, role=role
            ),
            agent=agents["advocate"],
            expected_output="JSON with advocacy array — one entry per Q&A pair",
            context=[task_present_qa],
        )

        task_critic = Task(
            description=_TASK_DESCS["critic_task"].format(
                company_name=company_name, stage=stage
            ),
            agent=agents["critic"],
            expected_output="JSON with criticism array — one entry per Q&A pair",
            context=[task_present_qa, task_advocate],
        )

        task_judge = Task(
            description=_TASK_DESCS["judge_task"].format(
                company_name=company_name, stage=stage, role=role
            ),
            agent=agents["judge"],
            expected_output="JSON with rated_qa, overall_score, strengths, weaknesses, summary",
            context=[task_present_qa, task_advocate, task_critic],
        )

        crew = Crew(
            agents=[agents["qa_extractor"], agents["advocate"], agents["critic"], agents["judge"]],
            tasks=[task_present_qa, task_advocate, task_critic, task_judge],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        output_text = str(result).strip()
        if output_text.startswith("```"):
            output_text = output_text.split("```")[1]
            if output_text.startswith("json"):
                output_text = output_text[4:]
        output_text = output_text.strip()

        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", output_text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in crew output: {output_text[:200]}")
            parsed = json.loads(match.group())

        rated = parsed.get("rated_qa", [{}])[0] if parsed.get("rated_qa") else parsed
        score = float(rated.get("score", 0))
        score = max(0.0, min(10.0, score))
        suggested = rated.get("suggested_answer") if score < 7 else None

        return {
            "score": score,
            "feedback": rated.get("feedback", ""),
            "suggested_answer": suggested,
        }


