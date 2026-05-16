"""
LLM factory for the interview_retro CrewAI package.

When running locally, routes to an OpenAI-compatible endpoint (e.g. ollama-bridge):
  OPENAI_API_KEY   — triggers local mode when set
  OPENAI_BASE_URL  — required in local mode (e.g. http://127.0.0.1:4011/v1)
  OPENAI_MODEL     — required in local mode (e.g. llama3.2)

When running via CrewAI Enterprise / hosted, routes to OpenRouter:
  OPENROUTER_API_KEY — required for OpenRouter inference
  OPENROUTER_MODEL   — optional, defaults to openai/gpt-oss-20b:free
"""
import os

from crewai import LLM


def make_llm() -> LLM:
    """
    Build a CrewAI LLM.

    Local mode  (OPENAI_API_KEY set): routes to an OpenAI-compatible endpoint
                                      such as the ollama-bridge proxy.
    Online mode (OPENROUTER_API_KEY set): routes to OpenRouter,
                                          used when running inside CrewAI Enterprise.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return _make_openrouter_llm()
    return _make_openai_llm()



def _make_openai_llm() -> LLM:
    """Local mode — OpenAI-compatible endpoint (ollama-bridge)."""
    api_key = os.getenv("OPENAI_API_KEY", '')
    base_url = os.getenv("OPENAI_BASE_URL", 'http://127.0.0.1:4011/v1')
    model = os.getenv("OPENAI_MODEL", 'gpt-oss:20b')


    if not model:
        raise RuntimeError(
            "OPENAI_MODEL is required when OPENAI_API_KEY is set. "
            "Set it in your .env file (e.g. llama3.2)."
        )

    model_name = model if model.startswith("openai/") else f"openai/{model}"

    return LLM(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,
        timeout=300
        )


def _make_openrouter_llm() -> LLM:
    """Online mode — OpenRouter inference (CrewAI Enterprise)."""
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    api_key = os.getenv("OPENROUTER_API_KEY", 'test')

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required. Set it in your environment or .env file."
        )

    model_name = model if model.startswith("openrouter/") else f"openrouter/{model}"

    return LLM(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.1,
        timeout=300,
    )
