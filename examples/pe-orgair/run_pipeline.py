# SPDX-License-Identifier: Apache-2.0
"""
PE-OrgAIR full pipeline runner (simulation → evaluation), no HTTP server.

Usage:
    cd examples/pe-orgair
    python run_pipeline.py
"""

from __future__ import annotations

import asyncio
import os

from custom_agent import PEOrgAIRCustomAgent
from custom_metrics import (
    ARIScoreGroundingMetric,
    DataGapHandlingMetric,
    PushbackResilienceMetric,
    RecommendationSpecificityMetric,
    ScoreDirectionAccuracyMetric,
    WorkflowCompletenessMetric,
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
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results")

    model = "gpt-4o"
    provider = "openai"
    max_turns = 6

    # ── Step 1: Simulation ───────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1/2: Simulating PE analyst conversations...")
    print("=" * 60)

    scenarios = Scenarios.load(os.path.join(base_dir, "scenarios.json"))
    agent_config = AgentConfig(
        agent_type="custom",
        agent_name=PEOrgAIRCustomAgent.__name__,
        custom_config=CustomConfig(agent_class=PEOrgAIRCustomAgent),
    )
    llm = LLM(model=model, provider=provider)

    sim_params = SimulationParams(
        num_convos_per_scenario=1,
        max_turns=max_turns,
        num_workers=10,
        output_file_path=os.path.join(results_dir, "simulation", "simulation.json"),
    )
    simulator = Simulator(agent_config=agent_config, simulator_params=sim_params, llm=llm)
    simulation_output = await simulator.simulate(scenarios)
    await simulator.save()

    # ── Step 2: Evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2/2: Evaluating with built-in + custom metrics...")
    print("=" * 60)

    eval_params = EvaluationParams(
        output_dir=os.path.join(results_dir, "evaluation"),
        num_workers=10,
        custom_metrics=[
            ARIScoreGroundingMetric(),
            WorkflowCompletenessMetric(),
            RecommendationSpecificityMetric(),
        ],
        custom_qualitative_metrics=[
            ScoreDirectionAccuracyMetric(),
            DataGapHandlingMetric(),
            PushbackResilienceMetric(),
        ],
    )

    evaluator = Evaluator(eval_params, llm=llm)
    eval_output = await asyncio.to_thread(evaluator.evaluate, simulation_output)
    evaluator.display_evaluation_summary()
    evaluator.save_results()

    html_path = os.path.join(results_dir, "evaluation", "final_report.html")
    report_params = HtmlReportParams(
        simulation=simulation_output,
        evaluation=eval_output,
        scenarios=scenarios,
        output_path=html_path,
        chat_id_to_label=evaluator.chat_id_to_label,
    )
    await asyncio.to_thread(generate_html_report, report_params)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"HTML report: {html_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
