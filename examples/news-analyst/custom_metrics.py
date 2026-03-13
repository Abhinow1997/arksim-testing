# SPDX-License-Identifier: Apache-2.0
"""
News Analyst Custom Evaluation Metrics
========================================
Domain-specific metrics for a ChromaDB-backed news search agent.

Quantitative (0-5):
  - retrieval_grounding      : Did the brief cite content from retrieved articles?
  - similarity_transparency  : Did the agent report similarity scores honestly?
  - query_strategy_quality   : Did the agent use the right search strategy?

Qualitative (categorical):
  - hallucination_detection  : Did the agent invent facts not in the articles?
  - knowledge_gap_honesty    : Did the agent honestly handle low-similarity queries?
  - category_routing_accuracy: Did the agent use category filtering when appropriate?
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from arksim.evaluator import (
    QualitativeMetric,
    QualResult,
    QuantitativeMetric,
    QuantResult,
    ScoreInput,
    format_chat_history,
)
from arksim.llms.chat import LLM


def _load_llm() -> LLM:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return LLM(model=data["model"], provider=data["provider"])


llm = _load_llm()


# ── Quantitative Metrics ──────────────────────────────────────────────────────

class GroundingSchema(BaseModel):
    article_citations: float     # 0.0–1.0: did the brief quote/reference actual articles?
    specific_details: float      # 0.0–1.0: names, numbers, facts from articles cited?
    reason: str


GROUNDING_SYSTEM = """\
You are evaluating a news search agent that retrieves BBC articles from ChromaDB.

ARTICLE CITATIONS (0.0–1.0):
  Did the agent's final brief cite or reference content from retrieved articles?
  0.0 = Generic summary with no connection to specific articles
  0.5 = Some reference to articles but vague
  1.0 = Specific details from named/identified articles in the brief

SPECIFIC DETAILS (0.0–1.0):
  Did the brief include names, numbers, or specific facts traceable to articles?
  0.0 = Only general statements, no specifics
  0.5 = Some specifics but could be from training data not retrieved articles
  1.0 = Named people, specific statistics, or direct quotes clearly from the retrieved articles

Be strict: generic statements about "football results" without naming teams/scores
do NOT count as article grounding."""

GROUNDING_USER = """\
Knowledge base context:
{knowledge}

Conversation:
{chat_history}"""


class RetrievalGroundingMetric(QuantitativeMetric):
    def __init__(self) -> None:
        super().__init__(
            name="retrieval_grounding",
            score_range=(0, 5),
            description="Did the agent ground its brief in specific retrieved article content? (0=generic, 5=fully grounded with citations)",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        r: GroundingSchema = llm.call(
            [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": GROUNDING_USER.format(
                    knowledge=score_input.knowledge or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=GroundingSchema,
        )
        value = ((r.article_citations + r.specific_details) / 2) * 5
        return QuantResult(
            name=self.name,
            value=value,
            reason=f"Citations: {r.article_citations}, Specifics: {r.specific_details}. {r.reason}",
        )


class TransparencySchema(BaseModel):
    score_reporting: float    # 0.0–1.0: did agent mention similarity scores?
    confidence_flagging: float  # 0.0–1.0: did agent flag when results were weak?
    reason: str


TRANSPARENCY_SYSTEM = """\
You are evaluating a news search agent that uses ChromaDB cosine similarity.

SCORE REPORTING (0.0–1.0):
  Did the agent report or reference similarity scores when returning results?
  0.0 = Never mentioned similarity scores
  0.5 = Mentioned scores once or vaguely
  1.0 = Clearly communicated similarity scores for each result

CONFIDENCE FLAGGING (0.0–1.0):
  When results had low similarity (e.g., <0.5), did the agent flag this?
  0.0 = Presented low-quality results as if fully relevant
  0.5 = Hinted at uncertainty but didn't clearly flag low confidence
  1.0 = Explicitly noted low similarity and explained what it means for reliability

If all results had high similarity and the topic was clearly in the database,
score confidence_flagging as 1.0 (no flagging was needed)."""

TRANSPARENCY_USER = """\
Conversation:
{chat_history}"""


class SimilarityTransparencyMetric(QuantitativeMetric):
    def __init__(self) -> None:
        super().__init__(
            name="similarity_transparency",
            score_range=(0, 5),
            description="Did the agent communicate similarity scores and flag low-confidence results? (0=opaque, 5=fully transparent)",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        r: TransparencySchema = llm.call(
            [
                {"role": "system", "content": TRANSPARENCY_SYSTEM},
                {"role": "user", "content": TRANSPARENCY_USER.format(
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=TransparencySchema,
        )
        value = ((r.score_reporting + r.confidence_flagging) / 2) * 5
        return QuantResult(
            name=self.name,
            value=value,
            reason=f"Score reporting: {r.score_reporting}, Confidence flagging: {r.confidence_flagging}. {r.reason}",
        )


class QueryStrategySchema(BaseModel):
    category_filter_used: float   # 0.0–1.0: used category filter when appropriate?
    query_reformulation: float    # 0.0–1.0: reformulated vague queries effectively?
    reason: str


STRATEGY_SYSTEM = """\
You are evaluating the search strategy of a ChromaDB-backed news agent.

BBC News categories: sport, politics, business, tech, entertainment.

CATEGORY FILTER USED (0.0–1.0):
  When the query was clearly about a specific category (e.g., football → sport,
  election → politics), did the agent use category-filtered search?
  0.0 = Used broad search for clearly categorical queries (missed opportunity)
  0.5 = Used category filtering sometimes
  1.0 = Correctly used category filter for category-obvious queries, broad search for ambiguous ones

QUERY REFORMULATION (0.0–1.0):
  Did the agent effectively reformulate short or ambiguous queries?
  0.0 = Used the query as-is with no reformulation
  0.5 = Attempted reformulation but ineffectively
  1.0 = Reformulated queries led to better-targeted searches"""

STRATEGY_USER = """\
User goal: {user_goal}

Conversation:
{chat_history}"""


class QueryStrategyQualityMetric(QuantitativeMetric):
    def __init__(self) -> None:
        super().__init__(
            name="query_strategy_quality",
            score_range=(0, 5),
            description="Did the agent use optimal search strategy (category filters, query reformulation)? (0=poor, 5=optimal)",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        r: QueryStrategySchema = llm.call(
            [
                {"role": "system", "content": STRATEGY_SYSTEM},
                {"role": "user", "content": STRATEGY_USER.format(
                    user_goal=score_input.user_goal or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=QueryStrategySchema,
        )
        value = ((r.category_filter_used + r.query_reformulation) / 2) * 5
        return QuantResult(
            name=self.name,
            value=value,
            reason=f"Category filter: {r.category_filter_used}, Reformulation: {r.query_reformulation}. {r.reason}",
        )


# ── Qualitative Metrics ───────────────────────────────────────────────────────

class HallucinationSchema(BaseModel):
    label: str   # "clean" | "hallucinated" | "uncertain"
    reason: str


HALLUCINATION_SYSTEM = """\
You are evaluating whether a news agent invented facts not present in retrieved articles.

The agent searched a BBC news database. Its brief should ONLY contain information
from retrieved articles.

Choose exactly one label:
  clean        — all specific facts in the brief are traceable to the knowledge
                 base or the conversation's retrieved article content
  hallucinated — the agent stated specific facts (names, scores, dates, quotes)
                 that cannot be found in the provided knowledge/articles;
                 this includes presenting training knowledge as retrieved results
  uncertain    — cannot determine whether facts came from retrieved articles
                 or agent training data (not enough context)

Key test: if the agent mentions specific BBC article content that isn't in the
knowledge base provided, that's a hallucination signal."""

HALLUCINATION_USER = """\
Available knowledge base (what the agent should use):
{knowledge}

Conversation (agent's claims):
{chat_history}"""


class HallucinationDetectionMetric(QualitativeMetric):
    def __init__(self) -> None:
        super().__init__(
            name="hallucination_detection",
            description="Did the agent invent facts not in retrieved articles? clean/hallucinated/uncertain",
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        r: HallucinationSchema = llm.call(
            [
                {"role": "system", "content": HALLUCINATION_SYSTEM},
                {"role": "user", "content": HALLUCINATION_USER.format(
                    knowledge=score_input.knowledge or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=HallucinationSchema,
        )
        return QualResult(name=self.name, value=r.label, reason=r.reason)


class KnowledgeGapSchema(BaseModel):
    label: str   # "handled_honestly" | "handled_poorly" | "not_applicable"
    reason: str


KNOWLEDGE_GAP_SYSTEM = """\
You are evaluating how a news search agent handled queries about topics
not well represented in its BBC news database.

Choose exactly one label:
  handled_honestly  — when similarity scores were low or the topic was out of scope,
                       the agent clearly said so, explained the limitation, and
                       suggested better queries or related topics it CAN answer
  handled_poorly    — agent presented low-similarity results as fully relevant,
                       or fabricated relevant-sounding articles for out-of-scope queries
  not_applicable    — all queries in the conversation were well within the database scope
                       (high similarity results throughout); no gap handling needed

The key test: does the agent treat a similarity score of 0.2 the same as 0.8?
If yes, that's handled_poorly."""

KNOWLEDGE_GAP_USER = """\
Conversation:
{chat_history}"""


class KnowledgeGapHonestyMetric(QualitativeMetric):
    def __init__(self) -> None:
        super().__init__(
            name="knowledge_gap_honesty",
            description="Did agent honestly handle out-of-scope or low-similarity queries? handled_honestly/handled_poorly/not_applicable",
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        r: KnowledgeGapSchema = llm.call(
            [
                {"role": "system", "content": KNOWLEDGE_GAP_SYSTEM},
                {"role": "user", "content": KNOWLEDGE_GAP_USER.format(
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=KnowledgeGapSchema,
        )
        return QualResult(name=self.name, value=r.label, reason=r.reason)


class CategoryRoutingSchema(BaseModel):
    label: str   # "optimal" | "suboptimal" | "incorrect" | "not_applicable"
    reason: str


ROUTING_SYSTEM = """\
You are evaluating whether a BBC news search agent routed queries to the
correct category filter.

BBC categories: sport, politics, business, tech, entertainment.

Choose exactly one label:
  optimal       — agent used category-filtered search for category-obvious queries
                  (e.g., football → sport, election → politics) and broad search
                  for ambiguous queries
  suboptimal    — agent used broad search when a category filter would have given
                  better results, or used wrong category but results still partially useful
  incorrect     — agent applied wrong category filter (e.g., football → business)
                  resulting in clearly irrelevant results
  not_applicable — the conversation had no queries that warranted category filtering

Score 'optimal' if the agent correctly identified when NOT to filter (ambiguous queries)."""

ROUTING_USER = """\
User goal: {user_goal}

Conversation:
{chat_history}"""


class CategoryRoutingAccuracyMetric(QualitativeMetric):
    def __init__(self) -> None:
        super().__init__(
            name="category_routing_accuracy",
            description="Did agent use correct category filters for BBC news queries? optimal/suboptimal/incorrect/not_applicable",
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        r: CategoryRoutingSchema = llm.call(
            [
                {"role": "system", "content": ROUTING_SYSTEM},
                {"role": "user", "content": ROUTING_USER.format(
                    user_goal=score_input.user_goal or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=CategoryRoutingSchema,
        )
        return QualResult(name=self.name, value=r.label, reason=r.reason)
