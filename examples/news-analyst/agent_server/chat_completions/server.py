# SPDX-License-Identifier: Apache-2.0
"""
FastAPI Chat Completions server wrapping the ChromaDB News Analyst Agent.

Key fixes vs previous version:
- asyncio.Semaphore limits concurrent ChromaDB requests (not thread-safe)
- Full exception logging so 500 errors show the real cause in the server terminal
- Traceback printed on every unhandled exception
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent   # examples/news-analyst/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: fastapi/uvicorn not installed. Run: python -m pip install fastapi uvicorn")
    sys.exit(1)

from agent_server.core.agent import NewsAnalystAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="News Analyst -- ChromaDB Chat Completions")

# Semaphore: limits to 1 concurrent ChromaDB request at a time
# ChromaDB PersistentClient is NOT thread-safe for concurrent writes/reads
# arksim will queue requests instead of crashing with 500
_semaphore = asyncio.Semaphore(1)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None


class ChatResponse(BaseModel):
    choices: list[dict]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": "chromadb"}


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    # Find last user message
    try:
        last_user = next(m for m in reversed(request.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(status_code=400, detail="No user message in request.") from None

    history = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m is not last_user and m.role != "system"
    ]

    query_preview = last_user.content[:80]
    logger.info("Received query: %s", query_preview)

    # Serialize ChromaDB access -- only 1 concurrent request allowed
    async with _semaphore:
        try:
            agent = NewsAnalystAgent(history=history, model="gpt-4o-mini")
            answer = await asyncio.to_thread(agent.invoke_sync, last_user.content)
            logger.info("Answer: %s", answer[:80])
            return ChatResponse(
                choices=[{"message": {"role": "assistant", "content": answer}}]
            )
        except Exception as exc:
            # Log the FULL traceback so you can see what actually crashed
            logger.error("Agent raised an exception for query: %s", query_preview)
            logger.error("Exception: %s", exc)
            logger.error("Traceback:\n%s", traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Agent error: {type(exc).__name__}: {exc}"
            ) from exc


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))
    print(f"\nStarting News Analyst server on http://localhost:{port}")
    print("ChromaDB backend -- serialized access (1 request at a time)\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
