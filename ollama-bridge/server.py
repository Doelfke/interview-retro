"""
ollama-bridge — OpenAI-compatible proxy for a local ollama server.

Translates OpenAI-format chat-completion requests from CrewAI into ollama's
OpenAI-compatible API endpoint, with optional streaming passthrough.

Configuration (.env):
  BRIDGE_API_KEY   — the API key callers must present (set as OPENAI_API_KEY
                     in interview-retro's .env)
  OLLAMA_BASE_URL  — ollama server base URL  (default: http://localhost:11434)
  BRIDGE_HOST      — bind address            (default: 127.0.0.1)
  BRIDGE_PORT      — listen port             (default: 4011)
"""
import logging
import os
from collections.abc import AsyncIterator

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Keep noisy libraries at INFO
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "4011"))

app = FastAPI(title="ollama-bridge")


def _check_auth(request: Request) -> None:
    """Validate the Bearer token if BRIDGE_API_KEY is configured."""
    if not BRIDGE_API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    token = auth.removeprefix("Bearer ").strip()
    if token != BRIDGE_API_KEY:
        raise HTTPException(403, "Invalid API key")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    _check_auth(request)

    body = await request.body()
    headers = {
        "Content-Type": "application/json",
        "Accept": request.headers.get("Accept", "application/json"),
    }

    upstream_url = f"{OLLAMA_BASE_URL}/v1/chat/completions"

    # Detect streaming request and log request details
    try:
        import json
        payload = json.loads(body)
        is_stream = bool(payload.get("stream", False))
        # Strip the "openai/" prefix CrewAI adds — ollama doesn't expect it
        raw_model = payload.get("model", "")
        if raw_model.startswith("openai/"):
            payload["model"] = raw_model[len("openai/"):]
            body = json.dumps(payload).encode()
            logger.info("Stripped 'openai/' prefix from model: %s → %s", raw_model, payload["model"])
        logger.info(
            "Proxying chat/completions → %s | model=%s messages=%d stream=%s",
            upstream_url,
            payload.get("model", "<none>"),
            len(payload.get("messages", [])),
            is_stream,
        )
        logger.debug("Request payload: %s", json.dumps(payload, indent=2))
        # Always log the messages on DEBUG so failures are easy to trace
        for i, msg in enumerate(payload.get("messages", [])):
            logger.debug("  [%d] role=%s content=%.300s", i, msg.get("role"), str(msg.get("content", "")))
    except Exception:
        is_stream = False
        logger.info("Proxying chat/completions → %s (could not parse body)", upstream_url)

    if is_stream:
        async def _stream() -> "AsyncIterator[bytes]":
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream(
                        "POST", upstream_url, content=body, headers=headers
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except Exception:
                logger.exception("Error streaming from ollama upstream")
                raise

        return StreamingResponse(_stream(), media_type="text/event-stream")

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(upstream_url, content=body, headers=headers)
    except Exception:
        logger.exception("Error proxying request to ollama upstream %s", upstream_url)
        raise

    if resp.status_code >= 400:
        logger.error(
            "Upstream %s returned %s: %s",
            upstream_url,
            resp.status_code,
            resp.text,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "ollama_base_url": OLLAMA_BASE_URL}


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  ollama-bridge")
    logger.info(f"  Listening on http://{BRIDGE_HOST}:{BRIDGE_PORT}/v1")
    logger.info(f"  Forwarding to  {OLLAMA_BASE_URL}")
    logger.info(f"  Auth:          {'enabled' if BRIDGE_API_KEY else 'DISABLED (set BRIDGE_API_KEY)'}")
    logger.info("=" * 60)
    uvicorn.run("server:app", host=BRIDGE_HOST, port=BRIDGE_PORT, reload=True)
