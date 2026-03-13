# SPDX-License-Identifier: Apache-2.0
"""
News Analyst — full arksim pipeline runner.

PRE-REQUISITES (run once):
    python -m pip install arksim
    python build_index.py --csv agent_server/data/articles.csv

Then run from examples/news-analyst/:
    python run_pipeline.py

What this does differently from 'arksim simulate-evaluate config.yaml':
    config.yaml  → calls OpenAI API directly → LLM answers from training data (HALLUCINATION RISK)
    run_pipeline → uses NewsAnalystCustomAgent → forces ChromaDB tool calls → GROUNDED ANSWERS
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# ── Pre-flight: make sure arksim is importable ────────────────────────────────
try:
    import arksim  # noqa: F401
except ModuleNotFoundError:
    print("ERROR: arksim is not installed in this Python environment.")
    print()
    print("Fix:")
    print("  python -m pip install arksim")
    print()
    sys.exit(1)

# ── Pre-flight: make sure ChromaDB index exists ───────────────────────────────
_BASE_DIR = Path(__file__).parent
_DB_PATH = _BASE_DIR / "agent_server" / "VectorDB"
_CHROMA_DB = _DB_PATH / "chroma.sqlite3"

if not _CHROMA_DB.exists():
    print("ERROR: ChromaDB index not found.")
    print()
    print("Fix:")
    print("  python build_index.py --csv agent_server/data/articles.csv")
    print()
    sys.exit(1)

# ── Now safe to import ────────────────────────────────────────────────────────
sys.path.insert(0, str(_BASE_DIR))

from custom_agent import NewsAnalystCustomAgent
from custom_metrics import (
    CategoryRoutingAccuracyMetric,
    HallucinationDetectionMetric,
    KnowledgeGapHonestyMetric,
    QueryStrategyQualityMetric,
    RetrievalGroundingMetric,
    SimilarityTransparencyMetric,
)

from arksim.config import AgentConfig, CustomConfig
from arksim.evaluator import EvaluationParams, Evaluator
from arksim.llms.chat import LLM
from arksim.scenario import Scenarios
from arksim.simulation_engine import SimulationParams, Simulator
from arksim.utils.html_report.generate_html_report import (
    HtmlReportParams,
    generate_html_report,
)


async def main() -> None:
    base_dir = str(_BASE_DIR)
    results_dir = os.path.join(base_dir, "results")

    # Use gpt-4o-mini for both simulation and evaluation to keep costs low
    # The agent itself also uses gpt-4o-mini (set in custom_agent.py)
    model = "gpt-4o-mini"
    provider = "openai"

    print("=" * 60)
    print("News Analyst Pipeline — ChromaDB RAG Agent")
    print("=" * 60)
    print(f"  Model        : {model}")
    print(f"  Scenarios    : {os.path.join(base_dir, 'scenarios.json')}")
    print(f"  ChromaDB     : {_DB_PATH}")
    print(f"  Results dir  : {results_dir}")
    print()

    # ── Step 1: Simulation ────────────────────────────────────────────────────
    print("Step 1/2: Running simulated conversations...")
    print("  Each turn: simulated user → NewsAnalystAgent → ChromaDB tools → answer")
    print()

    scenarios = Scenarios.load(os.path.join(base_dir, "scenarios.json"))

    agent_config = AgentConfig(
        agent_type="custom",
        agent_name="NewsAnalystCustomAgent",
        custom_config=CustomConfig(agent_class=NewsAnalystCustomAgent),
    )
    llm = LLM(model=model, provider=provider)

    sim_params = SimulationParams(
        num_convos_per_scenario=1,
        max_turns=5,          # 5 turns per scenario
        num_workers=2,        # low concurrency — ChromaDB + OpenAI rate limits
        output_file_path=os.path.join(results_dir, "simulation", "simulation.json"),
    )

    simulator = Simulator(
        agent_config=agent_config,
        simulator_params=sim_params,
        llm=llm,
    )
    simulation_output = await simulator.simulate(scenarios)
    await simulator.save()

    print()
    print("Simulation complete.")
    print()

    # ── Step 2: Evaluation ────────────────────────────────────────────────────
    print("Step 2/2: Evaluating conversations...")
    print("  Built-in metrics: faithfulness, helpfulness, coherence, relevance, goal_completion")
    print("  Custom metrics:   retrieval_grounding, similarity_transparency,")
    print("                    query_strategy_quality, hallucination_detection,")
    print("                    knowledge_gap_honesty, category_routing_accuracy")
    print()

    eval_params = EvaluationParams(
        output_dir=os.path.join(results_dir, "evaluation"),
        num_workers=2,
        custom_metrics=[
            RetrievalGroundingMetric(),
            SimilarityTransparencyMetric(),
            QueryStrategyQualityMetric(),
        ],
        custom_qualitative_metrics=[
            HallucinationDetectionMetric(),
            KnowledgeGapHonestyMetric(),
            CategoryRoutingAccuracyMetric(),
        ],
    )

    evaluator = Evaluator(eval_params, llm=llm)
    eval_output = await asyncio.to_thread(evaluator.evaluate, simulation_output)
    evaluator.display_evaluation_summary()
    evaluator.save_results()

    # ── HTML report ───────────────────────────────────────────────────────────
    html_path = os.path.join(results_dir, "evaluation", "final_report.html")
    await asyncio.to_thread(
        generate_html_report,
        HtmlReportParams(
            simulation=simulation_output,
            evaluation=eval_output,
            scenarios=scenarios,
            output_path=html_path,
            chat_id_to_label=evaluator.chat_id_to_label,
        ),
    )

    print()
    print("=" * 60)
    print("Pipeline complete!")
    print(f"  HTML report: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
