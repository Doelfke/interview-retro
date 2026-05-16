"""
Interview Retro — @CrewBase entry point for crewai deploy.

Five-agent debate pipeline:
  1. Transcription  — cleans and speaker-labels the raw transcript
  2. Q&A Extraction — pulls every question + answer pair
  3. Advocate       — argues the strongest case FOR each answer
  4. Critic         — reads the advocate's case and argues AGAINST it
  5. Judge          — weighs both arguments, scores, and gives a verdict

Inputs (passed via crew.kickoff(inputs={...}) or crewai deploy API):
  transcript    — raw interview transcript text
  company_name  — company being interviewed at
  role          — job role / position title
  stage         — interview stage (e.g. "phone screen", "onsite")
"""
import os
from typing import Any, Callable, cast

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task


def _make_llm() -> LLM:
    """Build the LLM that routes to the local mlx_lm.server instance.

    Uses LiteLLM's openai/ prefix with a custom base_url so no OPENAI_API_KEY
    is required.  The model name must match whatever mlx_lm.server loaded.
    """
    base_url = os.getenv("MLX_SERVER_URL", "http://localhost:8081/v1")
    mlx_model = os.getenv(
        "MLX_MODEL", "mlx-community/Qwen2.5-32B-Instruct-4bit"
    )
    return LLM(
        model=f"openai/{mlx_model}",
        base_url=base_url,
        api_key="not-required",  # mlx_lm.server ignores auth
        temperature=0.1,
        timeout=300,
    )

# ─── Task description strings ─────────────────────────────────────────────────
# Regular Python strings with {variable} placeholders for CrewAI interpolation.
# Curly braces that are literal JSON are doubled ({{ / }}) so that Python's
# str.format() — used internally by CrewAI — renders them as { / }.

_TRANSCRIPTION_DESC = """
You have a raw transcript from a job interview at {company_name}
for the role of {role} (Stage: {stage}).

RAW TRANSCRIPT:
{transcript}

Clean and structure it:
1. Label every turn as INTERVIEWER or CANDIDATE
2. Fix obvious transcription errors ("your" vs "you're", etc.)
3. Preserve all content — do not summarize or omit anything

Return ONLY this JSON:
{{
  "structured_transcript": [
    {{"speaker": "INTERVIEWER", "text": "...", "timestamp_seconds": 0}},
    {{"speaker": "CANDIDATE",   "text": "...", "timestamp_seconds": 15}},
    ...
  ]
}}
"""

_QA_EXTRACTION_DESC = """
Using the structured transcript, extract every question-answer pair
from this {company_name} interview for the role of {role} ({stage}).

For each pair:
- question: full question text (may span multiple turns)
- answer: the candidate's complete response
- category: behavioral | technical | situational | culture_fit |
            role_specific | general
- timestamp_seconds: when the question was asked

Return ONLY this JSON:
{{
  "qa_pairs": [
    {{
      "question": "Tell me about a time you disagreed with your manager.",
      "answer": "Sure, so at my last job...",
      "category": "behavioral",
      "timestamp_seconds": 120
    }},
    ...
  ]
}}
"""

_ADVOCATE_DESC = """
You are the ADVOCATE in a structured debate about how well the candidate
performed in their {stage} interview at {company_name} for {role}.

For EACH Q&A pair, make the strongest honest case that the answer was good.
Consider: relevance to the question, structure (STAR for behavioral,
correctness for technical), specificity, communication clarity, and any
domain knowledge demonstrated.

Do NOT invent strengths that aren't there — argue only what the answer
actually contains. Vague praise ("they answered confidently") is not useful.

Return ONLY this JSON:
{{
  "advocacy": [
    {{
      "question": "...",
      "answer": "...",
      "category": "...",
      "timestamp_seconds": 0,
      "strengths": [
        "Opened with a clear situation and named the specific stakeholders involved",
        "Quantified the outcome: 'reduced deploy time by 40%'"
      ],
      "advocate_summary": "The candidate directly addressed the question with a
        concrete example and demonstrated measurable impact."
    }},
    ...
  ]
}}
"""

_CRITIC_DESC = """
You are the CRITIC in a structured debate. The advocate has just argued
the case FOR each answer in the {company_name} {stage} interview.

Read the advocate's case carefully. For EACH answer, argue specifically
AGAINST it. Challenge every claimed strength with a concrete counter.
Identify: what was missing, what was vague, what was technically wrong,
what the question actually asked for vs. what was given.

Be precise. "They could have said more" is not useful. "The advocate claims
they quantified the outcome, but the 40% figure was unattributed and the
interviewer never probed it — the candidate did not volunteer the methodology"
is useful.

Return ONLY this JSON:
{{
  "criticism": [
    {{
      "question": "...",
      "weaknesses": [
        "The advocate credits specificity, but the candidate never named
         the actual conflict — only referred to 'a stakeholder'",
        "No Result step: the STAR answer ended at Action with no outcome stated"
      ],
      "rebuttal_of_advocate": "The advocate's claim of 'measurable impact'
        is undermined by the fact that no metric was given until prompted.",
      "critic_summary": "The answer had structural bones but was too vague
        on specifics and omitted the result entirely unprompted."
    }},
    ...
  ]
}}
"""

_JUDGE_DESC = """
You are the JUDGE. The advocate and critic have argued their cases for
each answer in the {company_name} {stage} interview for {role}.

For EACH answer, weigh both arguments and deliver a final ruling:

- score (0.0–10.0): calibrated based on the strength of the arguments.
  If the critic landed concrete blows that the advocate did not anticipate,
  score lower. If the advocate's points stand and the critic only quibbled,
  score higher.
- feedback: 2-4 sentences. Must reference BOTH sides.
- suggested_answer: ONLY if score < 6.0. Write a substantially better
  answer the candidate could have given.

Scoring rubric:
  9-10: Exceptional — advocate strong, critic found little
  7-8:  Good — advocate strong, critic found minor gaps
  5-6:  Adequate — both sides had valid points
  3-4:  Weak — critic's points substantially outweighed advocate's
  1-2:  Poor — advocate case almost entirely rebuttable
  0:    No answer given

Return ONLY this JSON:
{{
  "rated_qa": [
    {{
      "question": "...",
      "answer": "...",
      "category": "...",
      "timestamp_seconds": 0,
      "score": 6.5,
      "feedback": "The advocate was right that you named specific stakeholders,
        but the critic correctly noted the result was omitted until prompted.",
      "suggested_answer": null
    }},
    ...
  ],
  "overall_score": 6.8,
  "strengths": ["Consistent use of concrete examples", "Strong technical depth"],
  "weaknesses": ["Results often omitted or vague", "Missed clarifying questions"],
  "summary": "A solid candidate who demonstrated real experience but left
    points on the table by not closing the loop on outcomes."
}}
"""


@CrewBase
class InterviewRetroCrew:
    """Five-agent debate pipeline for scoring job interview performance."""

    agents_config: Any = "config/agents.yaml"
    tasks_config: Any = "config/tasks.yaml"
    agents: list[Any]
    tasks: list[Any]

    # ── Agents ────────────────────────────────────────────────────────────────

    @agent
    def transcription_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["transcription_agent"],
            llm=_make_llm(),
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def qa_extractor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["qa_extractor_agent"],
            llm=_make_llm(),
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def advocate_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["advocate_agent"],
            llm=_make_llm(),
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def critic_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["critic_agent"],
            llm=_make_llm(),
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def judge_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["judge_agent"],
            llm=_make_llm(),
            verbose=True,
            allow_delegation=False,
        )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task
    def transcription_task(self) -> Task:
        return Task(
            description=_TRANSCRIPTION_DESC,
            agent=cast(Callable[[], Agent], self.transcription_agent)(),
            expected_output="JSON with structured_transcript array",
        )

    @task
    def qa_extraction_task(self) -> Task:
        return Task(
            description=_QA_EXTRACTION_DESC,
            agent=cast(Callable[[], Agent], self.qa_extractor_agent)(),
            expected_output="JSON with qa_pairs array",
        )

    @task
    def advocate_task(self) -> Task:
        return Task(
            description=_ADVOCATE_DESC,
            agent=cast(Callable[[], Agent], self.advocate_agent)(),
            expected_output="JSON with advocacy array — one entry per Q&A pair",
        )

    @task
    def critic_task(self) -> Task:
        return Task(
            description=_CRITIC_DESC,
            agent=cast(Callable[[], Agent], self.critic_agent)(),
            expected_output="JSON with criticism array — one entry per Q&A pair",
        )

    @task
    def judge_task(self) -> Task:
        return Task(
            description=_JUDGE_DESC,
            agent=cast(Callable[[], Agent], self.judge_agent)(),
            expected_output="JSON with rated_qa, overall_score, strengths, weaknesses, summary",
        )

    # ── Crew ──────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
