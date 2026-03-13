# SPDX-License-Identifier: Apache-2.0
"""
FastAPI Chat Completions server wrapping the ChromaDB News Analyst Agent.
All tool calls go to the real ChromaDB index — no hallucination from training data.

Run from examples/news-analyst/:
    python -m agent_server.chat_completions.server
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# ── Ensure the news-analyst root is on sys.path ───────────────────────────────
_ROOT = Path(__file__).parent.parent.parent   # examples/news-analyst/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: fastapi/uvicorn not installed.")
    print("  Fix: python -m pip install fastapi uvicorn")
    sys.exit(1)

from agent_server.core.agent import NewsAnalystAgent  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")
app = FastAPI(title="News Analyst — ChromaDB Chat Completions Wrapper")


# ── Request / Response schemas ────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None          # ignored — agent controls its own model


class ChatResponse(BaseModel):
    choices: list[dict]


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": "chromadb"}


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat/completions")
async def chat_completions(
    request: ChatRequest,
) -> ChatResponse:
    # No auth required — local dev server only
    # Find the last user message
    try:
        last_user = next(m for m in reversed(request.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(status_code=400, detail="No user message in request.") from None

    # Build conversation history (exclude last user msg + system msgs — agent adds its own)
    history = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m is not last_user and m.role != "system"
    ]

    logger.info("Query: %s", last_user.content[:80])

    # Run the ChromaDB agent in a thread (sync OpenAI client inside async context)
    agent = NewsAnalystAgent(history=history, model="gpt-4o-mini")
    answer = await asyncio.to_thread(agent.invoke_sync, last_user.content)

    logger.info("Answer preview: %s", answer[:80])

    return ChatResponse(
        choices=[{"message": {"role": "assistant", "content": answer}}]
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))
    print(f"\nStarting News Analyst server on http://localhost:{port}")
    print("ChromaDB backend — all answers grounded in BBC 2004-2005 articles\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
