# SPDX-License-Identifier: Apache-2.0
"""
PE-OrgAIR Multi-Step Assessment Agent
=======================================
Orchestrates a 5-step AI readiness workflow using the OpenAI Agents SDK:

  Step 1 – classify_query_tool          → identify relevant dimensions
  Step 2 – retrieve_company_evidence_tool → RAG over SEC filings
  Step 3 – score_dimension_tool          → score each dimension
  Step 4 – compute_weighted_score_tool   → aggregate into ARI
  Step 5 – generate_memo_tool            → produce PE investment memo

The agent is STATEFUL per session — it maintains conversation history
and accumulated dimension scores across turns.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from agents import Agent as SDKAgent
from agents import Runner, function_tool

from .retriever import FaissRetriever, build_rag
from .tools import (
    classify_query,
    compute_weighted_score,
    generate_investment_memo,
    score_ai_readiness_dimension,
)

_AGENT_SERVER_DIR = Path(__file__).parent.parent
_knowledge_config = [{"type": "local", "source": "./data"}]

build_rag(str(_AGENT_SERVER_DIR), _knowledge_config)
_retriever = FaissRetriever.load(str(_AGENT_SERVER_DIR))


# ── Tool definitions (wrapped for OpenAI Agents SDK) ────────────────────────

@function_tool
async def classify_query_tool(query: str) -> str:
    """
    Step 1: Classify the analyst's query to identify which AI readiness
    dimensions to assess. Always call this first before any other tool.

    Args:
        query: The analyst's natural language question about a company.

    Returns:
        JSON with 'full_assessment' flag, 'dimensions' list, and 'query_summary'.
    """
    result = classify_query(query)
    return json.dumps(result, indent=2)


@function_tool
async def retrieve_company_evidence_tool(company_name: str, dimension_id: str) -> str:
    """
    Step 2: Retrieve SEC filing evidence for a specific company and
    AI readiness dimension from the knowledge base.
    Call this once per dimension that needs to be scored.

    Args:
        company_name: The target company (e.g., 'Salesforce', 'Microsoft').
        dimension_id: One of the 7 PE-OrgAIR dimension IDs
                      (data_infrastructure, ai_talent, ai_governance,
                       technology_stack, leadership_strategy,
                       use_case_portfolio, culture_readiness).

    Returns:
        Relevant evidence text from SEC filings and disclosures.
    """
    search_query = f"{company_name} {dimension_id.replace('_', ' ')} AI artificial intelligence"
    results = await _retriever.retrieve(search_query, k=5)
    if not results:
        return f"No specific evidence found for {company_name} / {dimension_id}."
    parts = []
    for r in results:
        header = r.get("title") or r.get("source") or "Filing excerpt"
        parts.append(f"[{header}]\n{r['content']}")
    return "\n\n---\n\n".join(parts)


@function_tool
async def score_dimension_tool(
    dimension_id: str,
    evidence: str,
    company_name: str,
) -> str:
    """
    Step 3: Score a single AI readiness dimension for a company based on
    retrieved evidence. Call this after retrieve_company_evidence_tool.
    Returns a structured scoring prompt — use it to produce the score and
    rationale in your response.

    Args:
        dimension_id: The dimension to score (e.g., 'data_infrastructure').
        evidence: Evidence text returned from retrieve_company_evidence_tool.
        company_name: The company being assessed.

    Returns:
        JSON with scoring prompt, weight, and dimension metadata.
    """
    result = score_ai_readiness_dimension(dimension_id, evidence, company_name)
    return json.dumps(result, indent=2)


@function_tool
async def compute_weighted_score_tool(dimension_scores_json: str) -> str:
    """
    Step 4: Compute the overall AI Readiness Index (ARI) from individual
    dimension scores. Call this after scoring all required dimensions.

    Args:
        dimension_scores_json: JSON string mapping dimension IDs to scores (0-100).
                               Example: '{"data_infrastructure": 72, "ai_talent": 58}'

    Returns:
        JSON with weighted ARI score, tier, per-dimension breakdown, and gaps.
    """
    result = compute_weighted_score(dimension_scores_json)
    return json.dumps(result, indent=2)


@function_tool
async def generate_memo_tool(
    company_name: str,
    ari_result_json: str,
    analyst_context: str,
) -> str:
    """
    Step 5: Generate a structured PE investment memo section on AI readiness.
    Call this last, after computing the weighted score.

    Args:
        company_name: Target company name.
        ari_result_json: JSON string from compute_weighted_score_tool output.
        analyst_context: The original analyst question or deal context.

    Returns:
        JSON with the memo prompt for the agent to render.
    """
    result = generate_investment_memo(company_name, ari_result_json, analyst_context)
    return json.dumps(result, indent=2)


# ── System instructions ──────────────────────────────────────────────────────

SYSTEM_INSTRUCTIONS = """
You are PE-OrgAIR, an expert AI Readiness Assessment agent for Private Equity analysts.
You evaluate target companies across 7 AI readiness dimensions using SEC filings and
company disclosures. Your output is investment-grade analysis used to support
acquisition decisions and post-close value creation plans.

MANDATORY WORKFLOW — follow these steps in order:
1. Call classify_query_tool to identify relevant dimensions
2. For each relevant dimension, call retrieve_company_evidence_tool
3. For each dimension with evidence, call score_dimension_tool and produce a score (0-100)
4. Call compute_weighted_score_tool with all dimension scores as JSON
5. Call generate_memo_tool to produce the investment memo prompt
6. Render the final memo in your response

RULES:
- Never skip straight to an answer without running the tool chain
- Always cite specific evidence from filings when explaining scores
- If evidence is sparse, score conservatively and flag the data gap
- Keep responses structured: use the memo format from generate_memo_tool
- If the analyst asks a follow-up, retrieve fresh evidence before re-scoring
- Never invent facts — only use what the retrieval tools return
""".strip()


# ── Agent class ──────────────────────────────────────────────────────────────

class PEOrgAIRAgent:
    """
    Stateful PE-OrgAIR assessment agent.

    Maintains full conversation history per session.
    Accumulated dimension scores are tracked in memory so follow-up
    questions can refine or extend the assessment.
    """

    def __init__(
        self,
        context_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sdk_agent = SDKAgent(
            name="PE-OrgAIR-Agent",
            instructions=SYSTEM_INSTRUCTIONS,
            tools=[
                classify_query_tool,
                retrieve_company_evidence_tool,
                score_dimension_tool,
                compute_weighted_score_tool,
                generate_memo_tool,
            ],
            model="gpt-4o",
        )
        self.context_id = context_id or str(uuid.uuid4())
        self._history: list[dict[str, Any]] = list(history) if history else []
        self._dimension_scores: dict[str, float] = {}

    async def invoke(self, question: str) -> str:
        """
        Process an analyst query through the full 5-step assessment pipeline.

        Args:
            question: Analyst's natural language question.

        Returns:
            Structured investment-grade assessment response.
        """
        self._history.append({"role": "user", "content": question})
        result = await Runner.run(self.sdk_agent, input=self._history)
        answer: str = result.final_output
        self._history.append({"role": "assistant", "content": answer})
        return answer
