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


def _parse_qa_pairs(raw: str) -> list[dict[str, Any]]:
    """Parse Q&A pairs from extraction output, handling common LLM wrapping variations.

    Handles:
      - {"qa_pairs": [pair1, pair2]}          ← normal case
      - [pair1, pair2]                         ← bare array
      - [{"qa_pairs": [pair1, pair2]}]         ← doubly-wrapped (LLM nests the object)
      - {"qa_pairs": [{"qa_pairs": [...]}]}    ← same, but dict outer
    """
    try:
        raw_parsed: Any = json.loads(_strip_fence(raw))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []

    def _to_list(val: Any) -> list[Any]:  # noqa: ANN401
        return list(val) if isinstance(val, list) else []

    # Unwrap top-level object into a list
    if isinstance(raw_parsed, dict):
        d: dict[str, Any] = raw_parsed  # type: ignore[assignment]
        items: list[Any] = _to_list(d.get("qa_pairs") or d.get("pairs"))
    elif isinstance(raw_parsed, list):
        items = raw_parsed  # type: ignore[assignment]
    else:
        return []

    # Each item should be a flat pair dict; if an item still carries a
    # "qa_pairs" wrapper the LLM nested things one level too deep — flatten it.
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_d: dict[str, Any] = item  # type: ignore[assignment]
        if "qa_pairs" in item_d or "pairs" in item_d:
            inner = _to_list(item_d.get("qa_pairs") or item_d.get("pairs"))
            result.extend(d for d in inner if isinstance(d, dict))
        elif "question" in item_d or "answer" in item_d:
            result.append(item_d)

    return result

    return result


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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_pair_debate(
        self,
        qa_pair: dict[str, Any],
        company_name: str,
        role: str,
        stage: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run advocate → critic → judge for a single Q&A pair.

        Returns (advocacy_entry, criticism_entry, judge_entry).
        """
        agents = self.agents
        pair_json = json.dumps(qa_pair, indent=2)

        task_advocate = Task(
            description=(
                _TASK_DESCS["advocate_single_task"].format(
                    company_name=company_name, stage=stage, role=role
                )
                + f"\n\nQ&A Pair:\n{pair_json}"
            ),
            agent=agents["advocate"],
            expected_output="JSON with strengths (array) and advocate_summary",
        )

        task_critic = Task(
            description=_TASK_DESCS["critic_single_task"].format(
                company_name=company_name, stage=stage
            ),
            agent=agents["critic"],
            expected_output="JSON with weaknesses (array), rebuttal_of_advocate, critic_summary",
            context=[task_advocate],
        )

        task_judge = Task(
            description=_TASK_DESCS["judge_single_task"].format(
                company_name=company_name, stage=stage, role=role
            ),
            agent=agents["judge"],
            expected_output="JSON with score (float), feedback (string), suggested_answer (string or null)",
            context=[task_advocate, task_critic],
        )

        crew = Crew(
            agents=[agents["advocate"], agents["critic"], agents["judge"]],
            tasks=[task_advocate, task_critic, task_judge],
            process=Process.sequential,
            verbose=True,
        )

        crew.kickoff()

        def _parse_obj(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
            try:
                return json.loads(_strip_fence(raw))  # type: ignore[no-any-return]
            except (json.JSONDecodeError, AttributeError, TypeError):
                return fallback

        advocacy_entry = _parse_obj(
            task_advocate.output.raw if task_advocate.output else "",
            {"strengths": [], "advocate_summary": ""},
        )
        criticism_entry = _parse_obj(
            task_critic.output.raw if task_critic.output else "",
            {"weaknesses": [], "rebuttal_of_advocate": "", "critic_summary": ""},
        )
        judge_entry = _parse_obj(
            task_judge.output.raw if task_judge.output else "",
            {"score": 0.0, "feedback": "", "suggested_answer": None},
        )

        return advocacy_entry, criticism_entry, judge_entry

    @staticmethod
    def _assemble_result(
        qa_pairs: list[dict[str, Any]],
        all_advocacy: list[dict[str, Any]],
        all_criticism: list[dict[str, Any]],
        per_pair_judges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Combine per-pair debate results into the standard output structure."""
        rated_qa: list[dict[str, Any]] = []
        for qa_pair, judge_entry in zip(qa_pairs, per_pair_judges):
            rated_qa.append({
                "question": qa_pair.get("question", ""),
                "answer": qa_pair.get("answer", ""),
                "category": qa_pair.get("category", ""),
                "timestamp_seconds": qa_pair.get("timestamp_seconds"),
                "score": judge_entry.get("score", 0.0),
                "feedback": judge_entry.get("feedback", ""),
                "suggested_answer": judge_entry.get("suggested_answer"),
            })

        overall_score = (
            sum(r["score"] for r in rated_qa) / len(rated_qa) if rated_qa else 0.0
        )

        return {
            "rated_qa": rated_qa,
            "overall_score": overall_score,
            "strengths": [],
            "weaknesses": [],
            "summary": (
                f"{len(rated_qa)} Q&A pairs evaluated. "
                f"Overall score: {overall_score:.1f}/10."
            ),
            "analysis_status": "complete",
            "analysis_error": None,
            "qa_pairs_extracted": len(qa_pairs),
            "_checkpoints": {
                "qa_pairs": qa_pairs,
                "advocacy": all_advocacy,
                "criticism": all_criticism,
            },
        }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

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

        Crew(
            agents=[agents["transcription"], agents["qa_extractor"]],
            tasks=[task_structure, task_extract_qa],
            process=Process.sequential,
            verbose=True,
        ).kickoff()

        qa_pairs = _parse_qa_pairs(
            task_extract_qa.output.raw if task_extract_qa.output else ""
        )

        # Run advocate → critic → judge for each Q&A pair in series
        all_advocacy: list[dict[str, Any]] = []
        all_criticism: list[dict[str, Any]] = []
        per_pair_judges: list[dict[str, Any]] = []

        for qa_pair in qa_pairs:
            advocacy_entry, criticism_entry, judge_entry = self._run_pair_debate(
                qa_pair, company_name, role, stage
            )
            all_advocacy.append(advocacy_entry)
            all_criticism.append(criticism_entry)
            per_pair_judges.append(judge_entry)

        return self._assemble_result(qa_pairs, all_advocacy, all_criticism, per_pair_judges)

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
        """Run only the judge task per Q&A pair with pre-computed advocacy and criticism."""
        agents = self.agents
        per_pair_judges: list[dict[str, Any]] = []

        for qa_pair, adv_entry, crit_entry in zip(qa_pairs, advocacy, criticism):
            judge_desc = (
                (enrichment or "")
                + _TASK_DESCS["judge_single_task"].format(
                    company_name=company_name, stage=stage, role=role
                )
                + f"\n\nQ&A Pair:\n{json.dumps(qa_pair, indent=2)}"
                + f"\n\nAdvocacy:\n{json.dumps(adv_entry, indent=2)}"
                + f"\n\nCriticism:\n{json.dumps(crit_entry, indent=2)}"
            )

            task_judge = Task(
                description=judge_desc,
                agent=agents["judge"],
                expected_output="JSON with score (float), feedback (string), suggested_answer (string or null)",
            )

            Crew(
                agents=[agents["judge"]],
                tasks=[task_judge],
                process=Process.sequential,
                verbose=True,
            ).kickoff()

            try:
                judge_entry: dict[str, Any] = json.loads(
                    _strip_fence(task_judge.output.raw if task_judge.output else "")
                )
            except (json.JSONDecodeError, AttributeError, TypeError):
                judge_entry = {"score": 0.0, "feedback": "", "suggested_answer": None}

            per_pair_judges.append(judge_entry)

        return self._assemble_result(qa_pairs, advocacy, criticism, per_pair_judges)

    def run_from_debate(
        self,
        qa_pairs: list[dict[str, Any]],
        company_name: str,
        role: str,
        stage: str,
        enrichment: str | None = None,
    ) -> dict[str, Any]:
        """Run advocate + critic + judge per Q&A pair."""
        all_advocacy: list[dict[str, Any]] = []
        all_criticism: list[dict[str, Any]] = []
        per_pair_judges: list[dict[str, Any]] = []

        for qa_pair in qa_pairs:
            advocacy_entry, criticism_entry, judge_entry = self._run_pair_debate(
                qa_pair, company_name, role, stage
            )
            all_advocacy.append(advocacy_entry)
            all_criticism.append(criticism_entry)
            per_pair_judges.append(judge_entry)

        return self._assemble_result(qa_pairs, all_advocacy, all_criticism, per_pair_judges)

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


