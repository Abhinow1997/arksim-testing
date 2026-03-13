# SPDX-License-Identifier: Apache-2.0
"""Custom agent connector for PE-OrgAIR — no HTTP server required."""

from __future__ import annotations

from agent_server.core.agent import PEOrgAIRAgent

from arksim.config import AgentConfig
from arksim.simulation_engine.agent.base import BaseAgent


class PEOrgAIRCustomAgent(BaseAgent):
    """BaseAgent wrapper around the PE-OrgAIR multi-step assessment agent."""

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self._agent = PEOrgAIRAgent()

    async def get_chat_id(self) -> str:
        return self._agent.context_id

    async def execute(self, user_query: str, **kwargs: object) -> str:
        return await self._agent.invoke(user_query)
