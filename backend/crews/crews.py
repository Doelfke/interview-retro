"""
CrewAI Crews
InterviewAnalysisCrew — processes a transcript through a 5-agent debate pipeline.
"""
import json
from typing import Any
from crewai import Crew, Task, Process, LLM
from agents.agents import make_agents


# ─── Task description builders ───────────────────────────────────────────────
# Extracted so that both the full-interview pipeline and the single-QA regrade
# flow use the exact same prompts.

def _advocate_description(company_name: str, stage: str, role: str) -> str:
    return f"""
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


def _critic_description(company_name: str, stage: str) -> str:
    return f"""
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
            is undermined by the fact that no metric was given until prompted,
            and even then it was hedged ('roughly 40%').",
          "critic_summary": "The answer had structural bones but was too vague
            on specifics and omitted the result entirely unprompted."
        }},
        ...
      ]
    }}
    """


def _judge_description(company_name: str, stage: str, role: str) -> str:
    return f"""
    You are the JUDGE. The advocate and critic have argued their cases for
    each answer in the {company_name} {stage} interview for {role}.

    For EACH answer, weigh both arguments and deliver a final ruling:

    - score (0.0–10.0): calibrated based on the strength of the arguments,
      not a split of the two positions. If the critic landed concrete blows
      that the advocate did not anticipate, score lower. If the advocate's
      points stand and the critic only quibbled, score higher.
    - feedback: 2-4 sentences. Must reference BOTH sides — what the advocate
      identified that genuinely worked, and what the critic identified that
      genuinely fell short. Specific, not generic.
    - suggested_answer: ONLY if score < 6.0. Write a substantially better
      answer the candidate could have given. Realistic, specific, same length
      as the original. Do not write an essay.

    Also produce overall stats across all answers.

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
            but the critic correctly noted the result was omitted until prompted.
            Lead with impact next time.",
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
            description=f"""
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
            """,
            agent=agents["transcription"],
            expected_output="JSON with structured_transcript array",
        )

        # ── Task 2: Extract Q&A pairs ─────────────────────────────────────────
        task_extract_qa = Task(
            description=f"""
            Using the structured transcript, extract every question-answer pair
            from this {company_name} interview.

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
            """,
            agent=agents["qa_extractor"],
            expected_output="JSON with qa_pairs array",
            context=[task_structure],
        )

        # ── Task 3: Advocate — argue FOR each answer ──────────────────────────
        task_advocate = Task(
            description=_advocate_description(company_name, stage, role),
            agent=agents["advocate"],
            expected_output="JSON with advocacy array — one entry per Q&A pair",
            context=[task_extract_qa],
        )

        # ── Task 4: Critic — argue AGAINST each answer ────────────────────────
        task_critic = Task(
            description=_critic_description(company_name, stage),
            agent=agents["critic"],
            expected_output="JSON with criticism array — one entry per Q&A pair",
            context=[task_extract_qa, task_advocate],
        )

        # ── Task 5: Judge — final verdict ─────────────────────────────────────
        task_judge = Task(
            description=_judge_description(company_name, stage, role),
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
            description=_advocate_description(company_name, stage, role),
            agent=agents["advocate"],
            expected_output="JSON with advocacy array — one entry per Q&A pair",
            context=[task_present_qa],
        )

        task_critic = Task(
            description=_critic_description(company_name, stage),
            agent=agents["critic"],
            expected_output="JSON with criticism array — one entry per Q&A pair",
            context=[task_present_qa, task_advocate],
        )

        task_judge = Task(
            description=_judge_description(company_name, stage, role),
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
