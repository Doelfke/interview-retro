"""
CrewAI Crews
InterviewAnalysisCrew — processes a transcript through a 5-agent debate pipeline.
"""
import json
from pathlib import Path
from typing import Any
import yaml
from crewai import Crew, Task, Process, LLM
from agents.agents import make_agents

_TASKS_YAML = Path(__file__).parent.parent.parent / "src" / "config" / "tasks.yaml"

def _load_task_descs() -> dict[str, str]:
    with open(_TASKS_YAML) as f:
        cfg = yaml.safe_load(f)
    return {key: cfg[key]["description"] for key in cfg}

_TASK_DESCS = _load_task_descs()


# ─── Interview Analysis Crew ─────────────────────────────────────────────────

class InterviewAnalysisCrew:
    """
    Processes a completed interview via a 5-task sequential pipeline:

    1. Transcription agent  — cleans and speaker-labels the raw transcript
    2. Q&A extractor agent  — pulls every question + answer pair
    3. Advocate agent       — argues the strongest case FOR each answer
    4. Critic agent         — reads the advocate's case and argues AGAINST it
    5. Judge agent          — weighs both arguments, scores, and gives a verdict

    The debate in steps 3-4 produces more calibrated scores than a single rater
    because the critic is forced to engage with the advocate's best points, and
    the judge must rule on concrete arguments rather than first impressions.
    """

    def __init__(self, llm: LLM | None = None) -> None:
        self.agents = make_agents(llm)

    def run(self, transcript: str, company_name: str, role: str, stage: str) -> dict[str, Any]:
        agents = self.agents

        # ── Task 1: Structure the transcript ──────────────────────────────────
        task_structure = Task(
            description=(
                _TASK_DESCS["transcription_task"].format(company_name=company_name)
                + f"\n\nTranscript:\n{transcript}"
            ),
            agent=agents["transcription"],
            expected_output="JSON with structured_transcript array",
        )

        # ── Task 2: Extract Q&A pairs ─────────────────────────────────────────
        task_extract_qa = Task(
            description=_TASK_DESCS["qa_extraction_task"].format(
                company_name=company_name, role=role, stage=stage
            ),
            agent=agents["qa_extractor"],
            expected_output="JSON with qa_pairs array",
            context=[task_structure],
        )

        # ── Task 3: Advocate — argue FOR each answer ──────────────────────────
        task_advocate = Task(
            description=_TASK_DESCS["advocate_task"].format(company_name=company_name, stage=stage, role=role),
            agent=agents["advocate"],
            expected_output="JSON with advocacy array — one entry per Q&A pair",
            context=[task_extract_qa],
        )

        # ── Task 4: Critic — argue AGAINST each answer ────────────────────────
        task_critic = Task(
            description=_TASK_DESCS["critic_task"].format(company_name=company_name, stage=stage),
            agent=agents["critic"],
            expected_output="JSON with criticism array — one entry per Q&A pair",
            context=[task_extract_qa, task_advocate],
        )

        # ── Task 5: Judge — final verdict ─────────────────────────────────────
        task_judge = Task(
            description=_TASK_DESCS["judge_task"].format(company_name=company_name, stage=stage, role=role),
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

        try:
          output_text = str(result).strip()
          if output_text.startswith("```"):
            output_text = output_text.split("```")[1]
            if output_text.startswith("json"):
              output_text = output_text[4:]

          parsed = json.loads(output_text.strip())
          parsed.setdefault("analysis_status", "complete")
          parsed.setdefault("analysis_error", None)
          return parsed
        except json.JSONDecodeError:
          return {
            "rated_qa": [],
            "overall_score": 0,
            "strengths": [],
            "weaknesses": [],
            "summary": str(result),
            "raw_output": str(result),
            "analysis_status": "failed",
            "analysis_error": "Crew output was not valid JSON",
          }

    def regrade_answer(
        self,
        question: str,
        new_answer: str,
        category: str,
        company_name: str = "this company",
        stage: str = "the interview",
        role: str = "this role",
    ) -> dict[str, Any]:
        """
        Run a single Q&A pair through the same advocate → critic → judge pipeline
        used by the full interview analysis.  Returns score, feedback, and
        (if score < 6) suggested_answer.
        """
        agents = self.agents

        # Seed the pipeline with the single QA so the advocate/critic/judge
        # receive it in the same format as task_extract_qa produces.
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
            description=_TASK_DESCS["advocate_task"].format(company_name=company_name, stage=stage, role=role),
            agent=agents["advocate"],
            expected_output="JSON with advocacy array — one entry per Q&A pair",
            context=[task_present_qa],
        )

        task_critic = Task(
            description=_TASK_DESCS["critic_task"].format(company_name=company_name, stage=stage),
            agent=agents["critic"],
            expected_output="JSON with criticism array — one entry per Q&A pair",
            context=[task_present_qa, task_advocate],
        )

        task_judge = Task(
            description=_TASK_DESCS["judge_task"].format(company_name=company_name, stage=stage, role=role),
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
            import re
            match = re.search(r"\{.*\}", output_text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in crew output: {output_text[:200]}")
            parsed = json.loads(match.group())

        # Judge output follows the same schema as run() — grab the first rated_qa entry
        rated = parsed.get("rated_qa", [{}])[0] if parsed.get("rated_qa") else parsed
        score = float(rated.get("score", 0))
        score = max(0.0, min(10.0, score))
        suggested = rated.get("suggested_answer") if score < 7 else None

        return {
            "score": score,
            "feedback": rated.get("feedback", ""),
            "suggested_answer": suggested,
        }
