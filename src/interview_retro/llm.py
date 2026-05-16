"""
LLM factory for the interview_retro CrewAI package.

Routes CrewAI to Hugging Face Inference via LiteLLM.
Configure via environment variables:
  HUGGINGFACE_MODEL    — defaults to meta-llama/Llama-3.1-8B-Instruct
  HUGGINGFACE_API_KEY  — required for hosted inference
  HUGGINGFACE_BASE_URL — optional, defaults to https://router.huggingface.co/v1
"""
import os

from crewai import LLM


def make_llm() -> LLM:
    """Build a CrewAI LLM that routes to Hugging Face hosted inference."""
    model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1")

    if not api_key:
        raise RuntimeError(
            "HUGGINGFACE_API_KEY is required. Set it in your environment or .env file."
        )

    return LLM(
        model=f"huggingface/{model}",
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,
        timeout=300,
    )
