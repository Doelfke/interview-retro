"""
CrewAI Agents — backed by mlx_lm.server (OpenAI-compatible, Metal/MLX).

mlx_lm.server starts automatically when the FastAPI backend starts (see server.py).
It exposes an OpenAI-compatible API on localhost:8081.
CrewAI routes to it via LiteLLM's openai/ provider with a custom base_url.

Agent definitions (role, goal, backstory) are loaded from:
  src/interview_retro/config/agents.yaml

Recommended models for M5 Pro / 48 GB:
  mlx-community/Qwen2.5-32B-Instruct-4bit   ~18 GB  (default — best quality/speed)
  mlx-community/Qwen2.5-14B-Instruct-4bit   ~8 GB   (faster, still excellent)
  mlx-community/Mistral-Nemo-Instruct-4bit  ~7 GB   (lightest option)

Models are downloaded from HuggingFace Hub on first run and cached at
~/.cache/huggingface/hub/ — subsequent starts are instant.
"""
import os
from pathlib import Path

import yaml
from crewai import Agent, LLM

# Canonical agent definitions shared with the crewai deploy crew/ package
_AGENTS_YAML = Path(__file__).parent.parent.parent / "crew" / "config" / "agents.yaml"


def make_llm() -> LLM:
    """
    Build a CrewAI LLM that routes to the local mlx_lm.server instance.
    LiteLLM's openai/ provider handles the protocol.  The model name in the
    request must match the model mlx_lm.server was started with — versions
    ≥0.18 validate (and attempt to download) any unrecognized model name.
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


def make_agents(llm: LLM | None = None) -> dict[str, Agent]:
    if llm is None:
        llm = make_llm()

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
