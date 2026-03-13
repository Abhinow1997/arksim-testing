# SPDX-License-Identifier: Apache-2.0
"""
Custom agent connector for arksim.
Wraps NewsAnalystAgent (pure openai function calling + ChromaDB)
so arksim can drive it without an HTTP server.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make agent_server importable when run from the examples/news-analyst directory
sys.path.insert(0, str(Path(__file__).parent))

from agent_server.core.agent import NewsAnalystAgent

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class NewsAnalystCustomAgent(BaseAgent):
    """
    BaseAgent subclass that arksim calls for each conversation turn.

    arksim calls:
      - get_chat_id()  once per conversation
      - execute(user_query) once per turn
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        # Each conversation gets a fresh agent instance with empty history
        self._agent = NewsAnalystAgent(model="gpt-4o-mini")

    async def get_chat_id(self) -> str:
        return self._agent.context_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        return await self._agent.invoke(user_query)
