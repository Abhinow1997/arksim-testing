# PE-OrgAIR — AI Readiness Assessment Agent for Private Equity

This example implements a **complex multi-step agent workflow** for testing
with the arksim simulation-evaluation framework.

## What Makes This Agent Complex

Unlike the bank-insurance example (single RAG retrieval step), this agent
executes a **5-step tool chain** per assessment:

```
User Query
    │
    ▼
Step 1: classify_query_tool
        → Identify which of 7 dimensions to assess
    │
    ▼
Step 2: retrieve_company_evidence_tool  (called once per dimension)
        → RAG retrieval from SEC filing knowledge base
    │
    ▼
Step 3: score_dimension_tool  (called once per dimension)
        → 0-100 score with tier classification and rationale
    │
    ▼
Step 4: compute_weighted_score_tool
        → Aggregate into a single AI Readiness Index (ARI)
    │
    ▼
Step 5: generate_memo_tool
        → PE-grade investment memo section
    │
    ▼
Structured Response to Analyst
```

## The 7 AI Readiness Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| data_infrastructure | 20% | Data pipelines, lakehouse, feature stores |
| ai_talent | 20% | ML engineers, data scientists, CAIO |
| ai_governance | 15% | Ethics policies, bias audits, MRM |
| technology_stack | 15% | Cloud, MLOps, model deployment |
| leadership_strategy | 15% | Board AI commitment, roadmap, budget |
| use_case_portfolio | 10% | Deployed AI features, revenue impact |
| culture_readiness | 5% | AI literacy, change management |

## Test Scenarios

5 adversarial scenarios testing different agent behaviors:

1. **full_ari_assessment** — Full 7-dimension assessment with memo generation
2. **targeted_dimension_pushback** — Analyst challenges an overgenerous score
3. **sparse_data_scenario** — Private company with no public filings
4. **portfolio_comparison** — Side-by-side multi-company ranking
5. **post_close_100_day_plan** — Post-acquisition action planning

## Custom Evaluation Metrics

### Quantitative (0-5 scale)
- `ari_score_grounding` — Are scores backed by specific evidence citations?
- `workflow_completeness` — Did the agent execute all 5 pipeline steps?
- `recommendation_specificity` — Named tools, budgets, executable steps?

### Qualitative (categorical)
- `score_direction_accuracy` — Were scores in the correct tier given evidence?
- `data_gap_handling` — Did the agent handle sparse evidence correctly?
- `pushback_resilience` — Did the agent maintain accuracy under pressure?

## Quick Start (Option 1: Direct OpenAI API)

```powershell
$env:OPENAI_API_KEY="sk-..."
cd D:\SpringBigData\arksim\examples\pe-orgair
arksim simulate-evaluate config.yaml
```

## Quick Start (Option 2: Custom Agent with Full Tool Chain)

```powershell
# Install agent server dependencies
cd agent_server
pip install openai-agents faiss-cpu langchain-openai tiktoken beautifulsoup4

# Run full pipeline
cd ..
$env:OPENAI_API_KEY="sk-..."
python run_pipeline.py
```

## Quick Start (Option 3: Chat Completions Server)

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:AGENT_API_KEY="pe-orgair-secret"
python -m examples.pe-orgair.agent_server.chat_completions.server

# In another terminal:
arksim simulate-evaluate config_chat_completions.yaml
```

## File Structure

```
pe-orgair/
├── config.yaml           ← Simulation + evaluation config (OpenAI direct)
├── scenarios.json        ← 5 adversarial PE analyst test scenarios
├── custom_metrics.py     ← 6 domain-specific evaluation metrics
├── custom_agent.py       ← BaseAgent wrapper for the custom agent
├── run_pipeline.py       ← Full pipeline runner (no HTTP server)
└── agent_server/
    ├── core/
    │   ├── agent.py      ← PEOrgAIRAgent: 5-tool orchestrator
    │   ├── tools.py      ← The 5 pipeline step functions
    │   ├── retriever.py  ← FAISS-based RAG retrieval
    │   └── loader.py     ← Document loading utilities
    ├── chat_completions/
    │   └── server.py     ← FastAPI Chat Completions server
    └── data/             ← Knowledge base markdown files
        ├── 01_ai_readiness_benchmarks.md
        ├── 02_ai_governance_frameworks.md
        ├── 03_leadership_strategy_signals.md
        ├── 04_use_case_portfolio.md
        └── 05_culture_readiness.md
```
