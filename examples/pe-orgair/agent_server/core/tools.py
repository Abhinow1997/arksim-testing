# SPDX-License-Identifier: Apache-2.0
"""
PE-OrgAIR Multi-Step Tool Chain
================================
Implements the 5-step AI Readiness assessment pipeline as callable tools:

  Step 1 – classify_query          : Identify which AI readiness dimension(s) to assess
  Step 2 – retrieve_company_evidence: RAG retrieval from SEC filings / knowledge base
  Step 3 – score_ai_readiness      : LLM-score a single dimension (0–100)
  Step 4 – compute_weighted_score  : Aggregate dimension scores using spec weights
  Step 5 – generate_investment_memo: Synthesize a PE-grade assessment memo

Dimension weights (PE-OrgAIR spec):
  data_infrastructure   0.20
  ai_talent             0.20
  ai_governance         0.15
  technology_stack      0.15
  leadership_strategy   0.15
  use_case_portfolio    0.10
  culture_readiness     0.05
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Dimension registry ──────────────────────────────────────────────────────

DIMENSION_WEIGHTS: dict[str, float] = {
    "data_infrastructure": 0.20,
    "ai_talent": 0.20,
    "ai_governance": 0.15,
    "technology_stack": 0.15,
    "leadership_strategy": 0.15,
    "use_case_portfolio": 0.10,
    "culture_readiness": 0.05,
}

DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "data_infrastructure": (
        "Quality and maturity of data pipelines, data lakes, warehouses, "
        "data governance policies, and real-time data availability."
    ),
    "ai_talent": (
        "Number and quality of AI/ML engineers, data scientists, ML Ops "
        "practitioners, and AI leadership roles."
    ),
    "ai_governance": (
        "Existence of AI ethics policies, bias auditing, model explainability "
        "practices, regulatory compliance frameworks, and responsible AI programs."
    ),
    "technology_stack": (
        "Cloud infrastructure maturity, MLOps tooling, model deployment "
        "capabilities, API infrastructure, and vendor partnerships."
    ),
    "leadership_strategy": (
        "Board-level AI commitment, Chief AI Officer presence, AI roadmap "
        "clarity, budget allocation for AI initiatives, and strategic vision."
    ),
    "use_case_portfolio": (
        "Breadth and revenue impact of deployed AI use cases across the "
        "business, including automation, personalisation, and predictive analytics."
    ),
    "culture_readiness": (
        "Employee AI literacy, change management programmes, cross-functional "
        "AI adoption, and experimentation culture."
    ),
}

KEYWORD_TO_DIMENSIONS: dict[str, list[str]] = {
    "data": ["data_infrastructure"],
    "pipeline": ["data_infrastructure"],
    "warehouse": ["data_infrastructure"],
    "lake": ["data_infrastructure"],
    "talent": ["ai_talent"],
    "engineer": ["ai_talent"],
    "scientist": ["ai_talent"],
    "hire": ["ai_talent"],
    "headcount": ["ai_talent"],
    "governance": ["ai_governance"],
    "ethics": ["ai_governance"],
    "bias": ["ai_governance"],
    "compliance": ["ai_governance"],
    "responsible": ["ai_governance"],
    "cloud": ["technology_stack"],
    "infrastructure": ["technology_stack", "data_infrastructure"],
    "mlops": ["technology_stack"],
    "platform": ["technology_stack"],
    "api": ["technology_stack"],
    "ceo": ["leadership_strategy"],
    "board": ["leadership_strategy"],
    "strategy": ["leadership_strategy"],
    "roadmap": ["leadership_strategy"],
    "budget": ["leadership_strategy"],
    "use case": ["use_case_portfolio"],
    "product": ["use_case_portfolio"],
    "revenue": ["use_case_portfolio"],
    "automation": ["use_case_portfolio"],
    "culture": ["culture_readiness"],
    "adoption": ["culture_readiness"],
    "training": ["culture_readiness"],
    "literacy": ["culture_readiness"],
}


def classify_query(query: str) -> dict[str, Any]:
    """
    Step 1 — Classify which AI readiness dimension(s) a query is about.

    Performs keyword-based classification. Returns a sorted list of
    relevant dimensions with their weights, plus a flag indicating
    whether a full assessment (all dimensions) was requested.

    Args:
        query: The analyst's or user's natural language question.

    Returns:
        {
            "full_assessment": bool,
            "dimensions": [{"id": str, "weight": float, "description": str}],
            "query_summary": str,
        }
    """
    q_lower = query.lower()
    full_keywords = {
        "overall", "full", "complete", "all", "total",
        "score", "readiness", "assessment", "evaluate", "rate"
    }
    is_full = any(kw in q_lower for kw in full_keywords)

    matched: set[str] = set()
    for keyword, dims in KEYWORD_TO_DIMENSIONS.items():
        if keyword in q_lower:
            matched.update(dims)

    if is_full or not matched:
        dims_out = [
            {"id": d, "weight": w, "description": DIMENSION_DESCRIPTIONS[d]}
            for d, w in DIMENSION_WEIGHTS.items()
        ]
        return {
            "full_assessment": True,
            "dimensions": dims_out,
            "query_summary": f"Full AI readiness assessment requested. Query: {query}",
        }

    dims_out = [
        {"id": d, "weight": DIMENSION_WEIGHTS[d], "description": DIMENSION_DESCRIPTIONS[d]}
        for d in matched
    ]
    dims_out.sort(key=lambda x: x["weight"], reverse=True)
    return {
        "full_assessment": False,
        "dimensions": dims_out,
        "query_summary": f"Partial assessment for {[d['id'] for d in dims_out]}. Query: {query}",
    }


def score_ai_readiness_dimension(
    dimension_id: str,
    evidence: str,
    company_name: str,
) -> dict[str, Any]:
    """
    Step 3 — Score a single AI readiness dimension based on retrieved evidence.

    This is a structured scoring function that produces a 0–100 score
    with tier classification and detailed rationale. The actual LLM call
    is handled by the orchestrating agent; this function prepares the
    scoring prompt structure and validates inputs.

    Args:
        dimension_id: One of the 7 PE-OrgAIR dimension IDs.
        evidence: Raw text evidence retrieved from SEC filings / knowledge base.
        company_name: Name of the company being assessed.

    Returns:
        {
            "dimension_id": str,
            "company": str,
            "score_prompt": str,   # Structured prompt for the agent to evaluate
            "weight": float,
            "description": str,
        }
    """
    if dimension_id not in DIMENSION_WEIGHTS:
        available = list(DIMENSION_WEIGHTS.keys())
        return {
            "error": f"Unknown dimension '{dimension_id}'. Available: {available}",
            "dimension_id": dimension_id,
        }

    evidence_preview = evidence[:2000] if len(evidence) > 2000 else evidence
    description = DIMENSION_DESCRIPTIONS[dimension_id]
    weight = DIMENSION_WEIGHTS[dimension_id]

    score_prompt = f"""
Score {company_name}'s AI readiness for the dimension: **{dimension_id}**

Dimension definition:
{description}

Evidence from SEC filings and company disclosures:
{evidence_preview}

Scoring rubric (0–100):
  0–20  : No evidence of this capability; significant gaps identified
  21–40 : Early-stage / ad-hoc efforts with no systematic approach
  41–60 : Developing capability with some formal processes in place
  61–80 : Advanced capability, systematic approach, measurable outcomes
  81–100: Industry-leading capability, competitive differentiator

Provide:
1. A numeric score (integer, 0–100)
2. Tier label: Nascent / Developing / Advancing / Leading / Pioneering
3. Three key evidence signals supporting the score
4. One critical gap or risk
5. One specific recommendation for the PE firm
""".strip()

    return {
        "dimension_id": dimension_id,
        "company": company_name,
        "score_prompt": score_prompt,
        "weight": weight,
        "description": description,
    }


def compute_weighted_score(dimension_scores: str) -> dict[str, Any]:
    """
    Step 4 — Compute the overall weighted AI Readiness Index (ARI) score.

    Args:
        dimension_scores: JSON string of {dimension_id: score (0-100)} pairs.
                          Example: '{"data_infrastructure": 72, "ai_talent": 58}'

    Returns:
        {
            "weighted_score": float,       # 0–100 overall ARI
            "tier": str,                   # Overall tier label
            "breakdown": list[dict],       # Per-dimension weighted contribution
            "coverage": float,             # % of dimensions scored (0.0–1.0)
            "missing_dimensions": list[str],
        }
    """
    try:
        scores: dict[str, float] = json.loads(dimension_scores)
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON for dimension_scores: {exc}"}

    breakdown = []
    total_weight_used = 0.0
    weighted_sum = 0.0
    missing = []

    for dim_id, weight in DIMENSION_WEIGHTS.items():
        if dim_id in scores:
            raw = float(scores[dim_id])
            contribution = raw * weight
            weighted_sum += contribution
            total_weight_used += weight
            breakdown.append({
                "dimension": dim_id,
                "raw_score": raw,
                "weight": weight,
                "weighted_contribution": round(contribution, 2),
                "tier": _score_to_tier(raw),
            })
        else:
            missing.append(dim_id)

    if total_weight_used == 0:
        return {"error": "No valid dimension scores provided."}

    # Normalise to 100 if not all dimensions were scored
    ari = (weighted_sum / total_weight_used) if total_weight_used < 1.0 else weighted_sum
    coverage = total_weight_used  # sum of weights covered

    return {
        "weighted_score": round(ari, 1),
        "tier": _score_to_tier(ari),
        "breakdown": sorted(breakdown, key=lambda x: x["weight"], reverse=True),
        "coverage": round(coverage, 2),
        "missing_dimensions": missing,
    }


def generate_investment_memo(
    company_name: str,
    ari_result: str,
    analyst_context: str,
) -> dict[str, Any]:
    """
    Step 5 — Generate a structured PE investment memo section on AI readiness.

    Args:
        company_name: Target company name.
        ari_result: JSON string of the compute_weighted_score() output.
        analyst_context: Original analyst question or deal context.

    Returns:
        {
            "memo_prompt": str,   # Structured prompt for the agent to render the memo
            "company": str,
            "ari_score": float,
            "tier": str,
        }
    """
    try:
        ari_data: dict = json.loads(ari_result)
    except json.JSONDecodeError:
        ari_data = {}

    ari_score = ari_data.get("weighted_score", "N/A")
    tier = ari_data.get("tier", "Unknown")
    breakdown = ari_data.get("breakdown", [])
    missing = ari_data.get("missing_dimensions", [])

    breakdown_text = "\n".join(
        f"  - {b['dimension']}: {b['raw_score']}/100 (weighted: {b['weighted_contribution']})"
        for b in breakdown
    ) or "  No dimension scores available."

    missing_text = (
        f"  Missing dimensions (scored as neutral): {', '.join(missing)}"
        if missing else "  All 7 dimensions scored."
    )

    memo_prompt = f"""
Write a concise PE investment memo section titled "AI Readiness Assessment" for {company_name}.

Deal Context / Analyst Question:
{analyst_context}

AI Readiness Index (ARI) Score: {ari_score}/100  |  Tier: {tier}

Dimension Breakdown:
{breakdown_text}

Coverage note:
{missing_text}

Structure your memo section as follows:
1. **Executive Summary** (2–3 sentences): Overall AI readiness verdict for the investment thesis
2. **Key Strengths** (2–3 bullets): Where the company leads
3. **Critical Risks** (2–3 bullets): Where gaps could threaten value creation
4. **100-Day AI Action Plan** (3 bullets): Immediate post-close priorities for the PE team
5. **Value Creation Potential** (1 sentence): Estimated impact of closing AI gaps on EBITDA or revenue

Keep the tone analytical and investment-grade. Be specific and evidence-based.
""".strip()

    return {
        "memo_prompt": memo_prompt,
        "company": company_name,
        "ari_score": ari_score,
        "tier": tier,
    }


def _score_to_tier(score: float) -> str:
    if score >= 81:
        return "Pioneering"
    elif score >= 61:
        return "Leading"
    elif score >= 41:
        return "Advancing"
    elif score >= 21:
        return "Developing"
    else:
        return "Nascent"
