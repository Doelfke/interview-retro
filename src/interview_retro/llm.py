"""
LLM factory for the interview_retro CrewAI package.

Routes CrewAI to Hugging Face Inference.
Configure via environment variables:
  HUGGINGFACE_MODEL    — defaults to Qwen/Qwen3.5-35B-A3B
  HF_TOKEN             — required for hosted inference
  HUGGINGFACE_API_KEY  — optional backward-compatible fallback
  HUGGINGFACE_BASE_URL — optional, defaults to https://router.huggingface.co/v1
"""
import os

from crewai import LLM


def make_llm() -> LLM:
    """Build a CrewAI LLM that routes to Hugging Face hosted inference."""
    model = os.getenv("HUGGINGFACE_MODEL", "huggingface/Qwen/Qwen3.5-35B-A3B")
    api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1")

    if not api_key:
        raise RuntimeError(
            "HF_TOKEN is required. Set it in your environment or .env file."
        )

    model_name = model if model.startswith("huggingface/") else f"huggingface/{model}"

    return LLM(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,
        timeout=300,
    )
