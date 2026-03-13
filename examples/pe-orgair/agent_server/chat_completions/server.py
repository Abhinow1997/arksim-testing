# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from ..core.agent import PEOrgAIRAgent

logger = logging.getLogger(__name__)

AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")
app = FastAPI(title="PE-OrgAIR Chat Completions Wrapper")


class ChatCompletionRequestMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionRequestMessage]


class ChatCompletionResponse(BaseModel):
    choices: list[dict]


@app.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(None),
) -> ChatCompletionResponse:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.split(" ")
    if len(token) != 2 or token[1] != AGENT_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API token")

    try:
        last_user_msg = next(m for m in reversed(request.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(status_code=400, detail="No user message found.") from None

    chat_history = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m is not last_user_msg and m.role != "system"
    ]
    agent = PEOrgAIRAgent(history=chat_history)
    answer_text = await agent.invoke(last_user_msg.content)

    return ChatCompletionResponse(
        choices=[{"message": {"role": "assistant", "content": answer_text}}]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
