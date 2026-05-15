"""
CrewAI Agents — backed by mlx_lm.server (OpenAI-compatible, Metal/MLX).

mlx_lm.server starts automatically when the FastAPI backend starts (see server.py).
It exposes an OpenAI-compatible API on localhost:8081.
CrewAI routes to it via LiteLLM's openai/ provider with a custom base_url.

Recommended models for M5 Pro / 48 GB:
  mlx-community/Qwen2.5-32B-Instruct-4bit   ~18 GB  (default — best quality/speed)
  mlx-community/Qwen2.5-14B-Instruct-4bit   ~8 GB   (faster, still excellent)
  mlx-community/Mistral-Nemo-Instruct-4bit  ~7 GB   (lightest option)

Models are downloaded from HuggingFace Hub on first run and cached at
~/.cache/huggingface/hub/ — subsequent starts are instant.
"""
import os
from crewai import Agent, LLM


def make_llm() -> LLM:
    """
    Build a CrewAI LLM that routes to the local mlx_lm.server instance.
    LiteLLM's openai/ provider handles the protocol.  The model name in the
    request must match the model mlx_lm.server was started with — versions
    ≥0.18 validate (and attempt to download) any unrecognised model name.
    """
    base_url = os.getenv("MLX_SERVER_URL", "http://localhost:8081/v1")
    mlx_model = os.getenv("MLX_MODEL", "mlx-community/Qwen2.5-32B-Instruct-4bit")

    return LLM(
        model=f"openai/{mlx_model}",  # LiteLLM prefix + actual model name the server loaded
        base_url=base_url,
        api_key="not-required",       # required by LiteLLM; mlx_lm.server ignores it
        temperature=0.1,
        timeout=300,
    )


def make_agents(llm: LLM | None = None) -> dict:
    if llm is None:
        llm = make_llm()

    # ── Transcription Agent ─────────────────────────────────────────────────
    transcription_agent = Agent(
        role="Interview Transcription Specialist",
        goal=(
            "Clean and structure raw interview transcripts. "
            "Label every turn as INTERVIEWER or CANDIDATE and fix transcription errors."
        ),
        backstory=(
            "You are an expert transcriptionist who specialises in job interviews across "
            "Zoom, Google Meet, and Teams. You produce clean, speaker-labelled transcripts "
            "with precise timestamps and handle domain-specific jargon with ease."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ── Q&A Extraction Agent ────────────────────────────────────────────────
    qa_extractor_agent = Agent(
        role="Interview Q&A Extraction Expert",
        goal=(
            "Extract every question asked during a job interview and match it with the "
            "candidate's full answer. Categorise and timestamp each pair."
        ),
        backstory=(
            "You specialise in analysing interview transcripts. You perfectly identify "
            "when a question has been asked — even across multiple turns — extract the "
            "complete answer, and categorise it: behavioral, technical, situational, "
            "culture_fit, role_specific, or general."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ── Advocate Agent ──────────────────────────────────────────────────────
    advocate_agent = Agent(
        role="Interview Answer Advocate",
        goal=(
            "For each interview answer, build the strongest possible case that the "
            "candidate answered well. Find every genuine strength: relevant content, "
            "good structure, appropriate examples, domain knowledge, clear communication. "
            "Be thorough and specific — vague praise is useless."
        ),
        backstory=(
            "You are a senior recruiter who believes candidates deserve a fair hearing. "
            "Nervousness, rambling, or imperfect phrasing can mask a genuinely strong "
            "answer. Your job is to extract every real strength from what was actually "
            "said — not what was implied — and argue the best honest interpretation."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ── Critic Agent ────────────────────────────────────────────────────────
    critic_agent = Agent(
        role="Interview Answer Critic",
        goal=(
            "Read the advocate's case for each answer, then argue specifically why it "
            "falls short. Challenge every claimed strength with a concrete counter. "
            "Identify what was missed, vague, incorrect, or not asked for. Be rigorous "
            "and precise — vague criticism is as useless as vague praise."
        ),
        backstory=(
            "You are a demanding technical interviewer who has seen candidates oversell "
            "mediocre answers. You know every trick: answering a different question, "
            "burying a weak core in confident delivery, using jargon to mask shallow "
            "understanding. Challenge the advocate's case point by point and expose "
            "genuine weaknesses. Accurate, not harsh."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ── Judge Agent ─────────────────────────────────────────────────────────
    judge_agent = Agent(
        role="Interview Performance Judge",
        goal=(
            "Read the advocate's and critic's arguments for each answer. Weigh both "
            "sides and produce a final calibrated score (0-10), specific actionable "
            "feedback that references both perspectives, and — only for scores below "
            "6 — a substantially better answer the candidate could have given."
        ),
        backstory=(
            "You are an impartial panel chair who has reviewed thousands of interviews. "
            "You have heard both the advocate and critic argue their cases. You do not "
            "split the difference lazily — you weigh each argument on its merits. A "
            "strong advocate case with weak rebuttals means a high score. A weak "
            "advocate case torn apart by the critic means a low score. Your feedback "
            "must reference both sides so the candidate understands exactly what landed "
            "and what did not. Scoring rubric: 9-10 exceptional, 7-8 good, 5-6 "
            "adequate, 3-4 weak, 1-2 poor, 0 no answer. Scores below 6 require a "
            "suggested better answer."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # ── Calendar Agent ──────────────────────────────────────────────────────
    # (placeholder — removed; retained as comment for future pipeline tracking)

    return {
        "transcription": transcription_agent,
        "qa_extractor":  qa_extractor_agent,
        "advocate":      advocate_agent,
        "critic":        critic_agent,
        "judge":         judge_agent,
    }
