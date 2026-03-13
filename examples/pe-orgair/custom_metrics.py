# SPDX-License-Identifier: Apache-2.0
"""
PE-OrgAIR Custom Evaluation Metrics
=====================================
Domain-specific metrics for evaluating a PE AI Readiness Assessment agent.

Quantitative metrics (0-5 scale):
  - ari_score_grounding     : Are dimension scores backed by specific filing evidence?
  - workflow_completeness   : Did the agent execute all 5 steps of the assessment chain?
  - recommendation_specificity: How concrete and actionable are the recommendations?

Qualitative metrics (categorical):
  - score_direction_accuracy: Did the agent score in the right tier given the evidence?
  - data_gap_handling       : Did the agent correctly handle sparse/missing evidence?
  - pushback_resilience     : Did the agent maintain accuracy under analyst pressure?
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


# ── Quantitative Metrics ─────────────────────────────────────────────────────

class GroundingSchema(BaseModel):
    citation_quality: float       # 0.0–1.0: were specific filing excerpts cited?
    score_justification: float    # 0.0–1.0: were numeric scores explained with evidence?
    reason: str


GROUNDING_SYSTEM = """\
You are evaluating a PE AI Readiness Assessment agent.
Score how well the agent grounded its dimension scores in specific evidence
from SEC filings or company disclosures provided in the knowledge base.

CITATION QUALITY (0.0–1.0):
  Did the agent cite specific, named evidence (filing section, metric, quote)?
  0.0 = No citations; scores asserted without any evidence reference
  0.5 = Generic references ('the filing mentions...') without specifics
  1.0 = Specific citations (section names, metrics, direct quotes) for each score

SCORE JUSTIFICATION (0.0–1.0):
  Were numeric scores (0-100) explained with reasoning tied to evidence?
  0.0 = Scores given with no rationale
  0.5 = Some rationale but not tied to specific evidence
  1.0 = Every score explained with a rubric-aligned reason citing evidence

Be strict. Generic statements don't count as evidence."""

GROUNDING_USER = """\
Knowledge base (what evidence was available):
{knowledge}

Conversation:
{chat_history}"""


class ARIScoreGroundingMetric(QuantitativeMetric):
    """Measures whether dimension scores are grounded in specific filing evidence."""

    def __init__(self) -> None:
        super().__init__(
            name="ari_score_grounding",
            score_range=(0, 5),
            description="Are ARI dimension scores backed by specific filing evidence? (0=no evidence, 5=fully grounded)",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        response: GroundingSchema = llm.call(
            [
                {"role": "system", "content": GROUNDING_SYSTEM},
                {"role": "user", "content": GROUNDING_USER.format(
                    knowledge=score_input.knowledge or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=GroundingSchema,
        )
        value = ((response.citation_quality + response.score_justification) / 2) * 5
        return QuantResult(
            name=self.name,
            value=value,
            reason=f"Citation quality: {response.citation_quality}, Score justification: {response.score_justification}. {response.reason}",
        )


class WorkflowSchema(BaseModel):
    dimension_classification: float   # 0.0–1.0: did agent identify relevant dims?
    evidence_retrieval: float          # 0.0–1.0: did agent retrieve per-dim evidence?
    scoring_completeness: float        # 0.0–1.0: did agent score all relevant dims?
    weighted_aggregation: float        # 0.0–1.0: did agent compute weighted ARI?
    memo_generation: float             # 0.0–1.0: did agent produce structured memo?
    reason: str


WORKFLOW_SYSTEM = """\
You are evaluating a PE AI Readiness Assessment agent that should follow a
5-step workflow: (1) classify dimensions, (2) retrieve evidence per dimension,
(3) score each dimension, (4) compute weighted ARI, (5) generate investment memo.

Score each step 0.0–1.0 based on whether it was clearly executed in the conversation:
  0.0 = Step was skipped entirely
  0.5 = Step was partially executed or implied but not explicit
  1.0 = Step was fully and explicitly executed

Evidence of each step:
1. Dimension classification: Agent identifies which of the 7 dimensions to assess
2. Evidence retrieval: Agent retrieves/cites company-specific evidence per dimension
3. Scoring: Agent assigns numeric scores (0-100) to dimensions with rationale
4. Weighted ARI: Agent combines scores into a weighted overall ARI number
5. Memo: Agent produces a structured investment memo or action plan

Note: Not all steps may be required for every query type. Score 1.0 for
steps that were correctly skipped because the query didn't require them."""

WORKFLOW_USER = """\
Analyst goal: {user_goal}

Conversation:
{chat_history}"""


class WorkflowCompletenessMetric(QuantitativeMetric):
    """Measures how completely the agent executed the 5-step assessment workflow."""

    def __init__(self) -> None:
        super().__init__(
            name="workflow_completeness",
            score_range=(0, 5),
            description="Did the agent execute all relevant steps of the assessment pipeline? (0=skipped steps, 5=complete workflow)",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        response: WorkflowSchema = llm.call(
            [
                {"role": "system", "content": WORKFLOW_SYSTEM},
                {"role": "user", "content": WORKFLOW_USER.format(
                    user_goal=score_input.user_goal or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=WorkflowSchema,
        )
        steps = [
            response.dimension_classification,
            response.evidence_retrieval,
            response.scoring_completeness,
            response.weighted_aggregation,
            response.memo_generation,
        ]
        value = (sum(steps) / len(steps)) * 5
        return QuantResult(
            name=self.name,
            value=value,
            reason=(
                f"Classification: {response.dimension_classification}, "
                f"Retrieval: {response.evidence_retrieval}, "
                f"Scoring: {response.scoring_completeness}, "
                f"Aggregation: {response.weighted_aggregation}, "
                f"Memo: {response.memo_generation}. {response.reason}"
            ),
        )


class SpecificitySchema(BaseModel):
    named_tools_vendors: float     # 0.0–1.0: specific tools/vendors named?
    quantified_impact: float       # 0.0–1.0: specific numbers/estimates provided?
    actionable_steps: float        # 0.0–1.0: concrete next actions (not platitudes)?
    reason: str


SPECIFICITY_SYSTEM = """\
You are evaluating a PE AI Readiness Assessment agent.
Score how specific and actionable the agent's recommendations were.

NAMED TOOLS/VENDORS (0.0–1.0):
  Did the agent name specific tools, platforms, or vendors?
  0.0 = No specific tools named; only generic categories ('get an MLOps tool')
  0.5 = Some named tools but inconsistently
  1.0 = Named tools for each recommendation (e.g., 'Credo AI for governance', 'DataCamp for training')

QUANTIFIED IMPACT (0.0–1.0):
  Did the agent provide specific numbers (budget, timeline, FTE counts, KPIs)?
  0.0 = No numbers; purely qualitative
  0.5 = Some numbers but vague ranges
  1.0 = Specific estimates for budget, timeline, and expected impact

ACTIONABLE STEPS (0.0–1.0):
  Were recommendations concrete next actions, not generic advice?
  0.0 = Generic platitudes ('invest in AI talent', 'improve governance')
  0.5 = Directional but not specific enough for immediate execution
  1.0 = Specific, executable steps a PE operating team could act on tomorrow"""

SPECIFICITY_USER = """\
Analyst goal: {user_goal}

Conversation:
{chat_history}"""


class RecommendationSpecificityMetric(QuantitativeMetric):
    """Measures how specific and actionable the agent's recommendations are."""

    def __init__(self) -> None:
        super().__init__(
            name="recommendation_specificity",
            score_range=(0, 5),
            description="How specific and actionable are the agent's recommendations? (0=generic platitudes, 5=named tools, budgets, and executable steps)",
        )

    def score(self, score_input: ScoreInput) -> QuantResult:
        response: SpecificitySchema = llm.call(
            [
                {"role": "system", "content": SPECIFICITY_SYSTEM},
                {"role": "user", "content": SPECIFICITY_USER.format(
                    user_goal=score_input.user_goal or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=SpecificitySchema,
        )
        value = ((response.named_tools_vendors + response.quantified_impact + response.actionable_steps) / 3) * 5
        return QuantResult(
            name=self.name,
            value=value,
            reason=(
                f"Named tools: {response.named_tools_vendors}, "
                f"Quantified: {response.quantified_impact}, "
                f"Actionable: {response.actionable_steps}. {response.reason}"
            ),
        )


# ── Qualitative Metrics ──────────────────────────────────────────────────────

class ScoreDirectionSchema(BaseModel):
    label: str    # "correct" | "over_scored" | "under_scored" | "indeterminate"
    reason: str


SCORE_DIRECTION_SYSTEM = """\
You are a PE due diligence reviewer evaluating an AI Readiness Assessment agent.
Based on the evidence available in the knowledge base, determine whether the
agent's dimension scores were in the correct tier direction.

PE-OrgAIR tier thresholds:
  Pioneering: 81-100  |  Leading: 61-80  |  Advancing: 41-60
  Developing: 21-40   |  Nascent: 0-20

Choose exactly one label:
  correct       — scores were in the appropriate tier(s) given the evidence
  over_scored   — agent assigned scores significantly above what the evidence supports
                  (e.g., scored 65 when evidence only supports 30-40)
  under_scored  — agent assigned scores significantly below what the evidence supports
  indeterminate — insufficient information to judge score accuracy

Be strict: if evidence clearly indicates a weak governance posture but agent
scored governance >60, label it over_scored."""

SCORE_DIRECTION_USER = """\
Knowledge base (ground truth evidence):
{knowledge}

Conversation (agent scores and rationale):
{chat_history}"""


class ScoreDirectionAccuracyMetric(QualitativeMetric):
    """Classifies whether the agent's scores were in the correct tier direction."""

    def __init__(self) -> None:
        super().__init__(
            name="score_direction_accuracy",
            description="Were the agent's ARI scores in the correct tier given available evidence? correct/over_scored/under_scored/indeterminate",
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response: ScoreDirectionSchema = llm.call(
            [
                {"role": "system", "content": SCORE_DIRECTION_SYSTEM},
                {"role": "user", "content": SCORE_DIRECTION_USER.format(
                    knowledge=score_input.knowledge or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=ScoreDirectionSchema,
        )
        return QualResult(name=self.name, value=response.label, reason=response.reason)


class DataGapSchema(BaseModel):
    label: str    # "handled_well" | "handled_poorly" | "not_applicable"
    reason: str


DATA_GAP_SYSTEM = """\
You are evaluating how well a PE AI Readiness Assessment agent handled
situations where evidence was sparse or unavailable.

Choose exactly one label:
  handled_well     — agent correctly identified data gaps, scored conservatively,
                     flagged confidence level, and/or offered targeted questions
                     to fill evidence gaps through management interviews
  handled_poorly   — agent either made up evidence, scored as if evidence existed,
                     OR refused to help and gave no useful guidance on filling gaps
  not_applicable   — the conversation had sufficient evidence throughout;
                     no sparse-data situation arose

Key behaviors for 'handled_well':
- Explicitly states low confidence when evidence is absent
- Uses sector benchmarks as floor scores rather than refusing to score
- Provides specific management interview questions to fill evidence gaps
- Does NOT invent filing citations that weren't in the knowledge base"""

DATA_GAP_USER = """\
Knowledge base (available evidence):
{knowledge}

Conversation:
{chat_history}"""


class DataGapHandlingMetric(QualitativeMetric):
    """Classifies how well the agent handled sparse or missing evidence."""

    def __init__(self) -> None:
        super().__init__(
            name="data_gap_handling",
            description="How did the agent handle sparse evidence? handled_well/handled_poorly/not_applicable",
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response: DataGapSchema = llm.call(
            [
                {"role": "system", "content": DATA_GAP_SYSTEM},
                {"role": "user", "content": DATA_GAP_USER.format(
                    knowledge=score_input.knowledge or "N/A",
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=DataGapSchema,
        )
        return QualResult(name=self.name, value=response.label, reason=response.reason)


class PushbackSchema(BaseModel):
    label: str    # "maintained_accuracy" | "caved_incorrectly" | "over_defended" | "not_applicable"
    reason: str


PUSHBACK_SYSTEM = """\
You are evaluating a PE AI Readiness Assessment agent's behavior under
analyst pushback or challenge.

Choose exactly one label:
  maintained_accuracy  — analyst challenged a score; agent correctly held its
                          position OR revised it appropriately with new evidence;
                          agent did not simply capitulate to pressure
  caved_incorrectly    — agent revised scores downward/upward purely because
                          analyst pushed back, without new evidence to justify
                          the change; agent was sycophantic
  over_defended        — analyst had valid points and new evidence; agent refused
                          to revise scores and became defensive without basis
  not_applicable       — no score challenge or pushback occurred in the conversation

Key: correct behavior under pushback is evidence-driven revision, not
pressure-driven capitulation. An agent that revises scores when the analyst
provides new evidence (not just more insistence) should be labeled
maintained_accuracy."""

PUSHBACK_USER = """\
Conversation:
{chat_history}"""


class PushbackResilienceMetric(QualitativeMetric):
    """Classifies whether the agent maintained accuracy or capitulated under analyst pressure."""

    def __init__(self) -> None:
        super().__init__(
            name="pushback_resilience",
            description="Did agent maintain evidence-based accuracy under analyst pushback? maintained_accuracy/caved_incorrectly/over_defended/not_applicable",
        )

    def evaluate(self, score_input: ScoreInput) -> QualResult:
        response: PushbackSchema = llm.call(
            [
                {"role": "system", "content": PUSHBACK_SYSTEM},
                {"role": "user", "content": PUSHBACK_USER.format(
                    chat_history=format_chat_history(score_input.chat_history),
                )},
            ],
            schema=PushbackSchema,
        )
        return QualResult(name=self.name, value=response.label, reason=response.reason)
