#!/usr/bin/env bash
# Interview Retro — one-shot setup for Apple Silicon
set -euo pipefail

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${BOLD}Interview Retro by Jobound.io — local MLX setup${NC}\n"

# ── 1. uv ──────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo -e "${YELLOW}Installing uv...${NC}"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi
echo -e "${GREEN}✓ uv $(uv --version)${NC}"

# ── 2. Python + dependencies ───────────────────────────────────────────────
echo -e "\n${BOLD}Installing Python dependencies (mlx, mlx-lm, crewai...)${NC}"
uv sync
echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── 3. Pre-download MLX LLM model ─────────────────────────────────────────
MLX_MODEL="mlx-community/Qwen2.5-32B-Instruct-4bit"
if [ -f backend/.env ]; then
  ENV_MODEL=$(grep -E '^MLX_MODEL=' backend/.env | cut -d= -f2 | tr -d '"' | tr -d "'")
  [ -n "$ENV_MODEL" ] && MLX_MODEL="$ENV_MODEL"
fi

echo -e "\n${BOLD}Pre-downloading LLM model: ${MLX_MODEL}${NC}"
echo -e "${YELLOW}(Downloads ~18 GB on first run — grab a coffee)${NC}"
uv run python - << PYEOF
from mlx_lm import load
print(f"Downloading {r'$MLX_MODEL'}...")
load("$MLX_MODEL")
print("LLM model ready")
PYEOF
echo -e "${GREEN}✓ LLM model cached${NC}"

# ── 4. .env setup ──────────────────────────────────────────────────────────
if [ ! -f backend/.env ]; then
  cp backend/.env.example backend/.env
  echo -e "\n${YELLOW}Created backend/.env — review settings before starting.${NC}"
else
  echo -e "${GREEN}✓ backend/.env exists${NC}"
fi

# ── 5. Local database directory ────────────────────────────────────────────
mkdir -p "$HOME/.interview-retro"
echo -e "${GREEN}✓ Local database folder ready at ~/.interview-retro${NC}"

echo -e "\n${BOLD}${GREEN}Setup complete!${NC}\n"
echo -e "Start the backend:  ${BOLD}uv run python backend/server.py${NC}"
echo -e "  (mlx_lm.server starts automatically as a subprocess)"
echo -e "\nDashboard:          ${BOLD}http://localhost:8765/dashboard${NC}\n"
