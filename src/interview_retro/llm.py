"""
LLM factory for the interview_retro CrewAI package.

Routes to the local mlx_lm.server instance (OpenAI-compatible, Metal/MLX).
Configure via environment variables:
  MLX_SERVER_URL  — defaults to http://localhost:8081/v1
  MLX_MODEL       — defaults to mlx-community/Qwen2.5-32B-Instruct-4bit
"""
import os

from crewai import LLM


def make_llm() -> LLM:
    """Build a CrewAI LLM that routes to the local mlx_lm.server instance."""
    base_url = os.getenv("MLX_SERVER_URL", "http://localhost:8081/v1")
    mlx_model = os.getenv("MLX_MODEL", "mlx-community/Qwen2.5-32B-Instruct-4bit")

    return LLM(
        model=f"openai/{mlx_model}",  # LiteLLM openai/ prefix + model name the server loaded
        base_url=base_url,
        api_key="not-required",       # required by LiteLLM; mlx_lm.server ignores it
        temperature=0.1,
        timeout=300,
    )
