#!/usr/bin/env bash
# Interview Retro — one-shot setup for Apple Silicon
set -euo pipefail

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${BOLD}Interview Retro by Jobound.io — CrewAI + Hugging Face setup${NC}\n"

# ── 1. uv ──────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo -e "${YELLOW}Installing uv...${NC}"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi
echo -e "${GREEN}✓ uv $(uv --version)${NC}"

# ── 2. Python + dependencies ───────────────────────────────────────────────
echo -e "\n${BOLD}Installing Python dependencies (crewai, fastapi, db...)${NC}"
uv sync
echo -e "${GREEN}✓ Dependencies installed${NC}"


# ── 4. Local database directory ────────────────────────────────────────────
mkdir -p "$HOME/.interview-retro"
echo -e "${GREEN}✓ Local database folder ready at ~/.interview-retro${NC}"

echo -e "\n${BOLD}${GREEN}Setup complete!${NC}\n"
echo -e "Start the backend:  ${BOLD}uv run python backend/server.py${NC}"
echo -e ""
echo -e "${BOLD}Local mode (ollama):${NC}"
echo -e "  1. Start ollama-bridge:  ${BOLD}cd ollama-bridge && uv run python server.py${NC}"
echo -e ""
echo -e "${BOLD}Hosted mode (Hugging Face):${NC}"
echo -e "  Set in .env:             ${BOLD}HF_TOKEN${NC}"
echo -e "\nDashboard:          ${BOLD}http://localhost:8765/dashboard${NC}\n"
