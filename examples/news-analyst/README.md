# arksim Testing — BBC News Analyst Agent

**Course:** DAMG 7245 Big Data Intelligence Analytics, Northeastern University
**Framework:** [arksim](https://pypi.org/project/arksim/) — Agent Simulation and Evaluation

---

## What is arksim?

arksim is a framework for **testing AI agents** by simulating realistic user conversations
and automatically scoring the agent's responses across multiple quality dimensions.

Instead of manually reviewing agent outputs, arksim:
1. Spins up a **simulated user** (an LLM playing a persona with a specific goal)
2. Drives a multi-turn conversation against your agent
3. Runs **built-in + custom evaluation metrics** on every turn
4. Produces a **scored HTML report** showing exactly where the agent passed or failed

```
scenarios.json          config.yaml            custom_metrics.py
(test cases)            (how to run)           (how to judge)
      |                      |                       |
      v                      v                       v
 [Simulator]  <------>  [Your Agent]  ------>  [Evaluator]
  Simulated                                    Built-in metrics
  user drives                                  + Custom metrics
  conversation                                       |
                                                     v
                                             final_report.html
```

---

## Repository Structure

```
arksim/
|-- examples/
|   |-- news-analyst/          # Main example: BBC News RAG Agent
|   |   |-- scenarios.json     # 5 adversarial test scenarios
|   |   |-- config_chromadb.yaml  # arksim config (grounded mode)
|   |   |-- config.yaml        # arksim config (baseline direct mode)
|   |   |-- custom_metrics.py  # 6 domain-specific evaluation metrics
|   |   |-- custom_agent.py    # arksim BaseAgent wrapper
|   |   |-- build_index.py     # ChromaDB index builder
|   |   |-- run_pipeline.py    # Programmatic pipeline runner
|   |   |-- agent_server/      # FastAPI server wrapping the agent
|   |   |   |-- core/
|   |   |   |   |-- agent.py           # NewsAnalystAgent (OpenAI function calling)
|   |   |   |   |-- chromadb_retriever.py  # ChromaDB wrapper
|   |   |   |-- chat_completions/
|   |   |   |   |-- server.py  # FastAPI endpoint arksim talks to
|   |   |-- result/            # Generated reports (gitignored: HTML + JSON)
|   |-- bank-insurance/        # arksim official example (reference)
|   |-- e-commerce/            # arksim official example (reference)
```

---

## The Three arksim Files

Every arksim example is controlled by exactly three files:

### 1. `scenarios.json` — What Gets Tested

Each scenario defines one complete test case with five fields:

```json
{
  "scenario_id": "semantic_sports_query",

  "goal": "Instructions for the SIMULATED USER.
            Tells it what to ask, how to push back, when to stop.
            This is the simulated user's system prompt.",

  "agent_context": "Context injected into the agent for this scenario.
                    Sets the stage for what the agent should know.",

  "user_profile": "Persona details for the simulated user.
                   Name, role, personality, expectations.",

  "knowledge": [
    { "content": "EVALUATOR GROUND TRUTH for this scenario.
                  NOT given to the agent — only to the evaluator.
                  Used to judge if the agent's answer was correct." },
    { "content": "EVALUATOR CRITERIA: what PASS and FAIL look like." }
  ]
}
```

> **Critical detail about `knowledge`:** When there is exactly 1 knowledge item,
> arksim injects it into the simulated user's prompt. When there are 2+ items,
> arksim keeps them evaluator-only. Always use 2 knowledge items to prevent
> the simulated user from reading the answer key and stopping early.

### 2. `config.yaml` — How arksim Runs

```yaml
agent_config:
  agent_type: chat_completions    # How arksim calls your agent
  api_config:
    endpoint: http://localhost:8888/chat/completions  # Your agent's endpoint
    headers:
      Content-Type: application/json
    body:
      model: gpt-4o-mini
      messages:
        - role: system
          content: "Your agent's system prompt"

scenario_file_path: ./scenarios.json
num_conversations_per_scenario: 1
max_turns: 5                      # Max back-and-forth turns per conversation

output_file_path: ./result/simulation/simulation.json
output_dir: ./result/evaluation

custom_metrics_file_paths:
  - ./custom_metrics.py           # Your domain-specific metrics

metrics_to_run:
  - faithfulness                  # Did agent stick to facts?
  - helpfulness                   # Was response useful?
  - coherence                     # Was response logically structured?
  - relevance                     # Did response address the question?
  - goal_completion               # Did user achieve their goal?

model: gpt-4o-mini                # LLM used by the EVALUATOR (not the agent)
provider: openai
num_workers: 3
```

### 3. `custom_metrics.py` — How Quality is Judged

Each metric is an LLM call that reads the conversation and scores it:

```python
# QUANTITATIVE metric (returns a number 0-5)
class RetrievalGroundingMetric(QuantitativeMetric):
    def score(self, score_input: ScoreInput) -> QuantResult:
        # score_input.chat_history  = full conversation transcript
        # score_input.knowledge     = the knowledge field from scenarios.json
        # score_input.user_goal     = the goal field from scenarios.json
        ...
        return QuantResult(name=self.name, value=3.5, reason="...")

# QUALITATIVE metric (returns a label)
class HallucinationDetectionMetric(QualitativeMetric):
    def evaluate(self, score_input: ScoreInput) -> QualResult:
        ...
        return QualResult(name=self.name, value="hallucinated", reason="...")
```

---

## How the Simulation Loop Works

```
arksim reads scenarios.json
        |
        v
For each scenario:
  Create simulated user (LLM with goal as system prompt)
  Create fresh conversation
        |
        v
  TURN LOOP (up to max_turns):
    Simulated User sends message
            |
            v
    POST to agent endpoint (your FastAPI server)
            |
            v
    Agent processes: tool calls → ChromaDB → response
            |
            v
    arksim records the exchange
    Check if simulated user generated ###STOP###
            |
            v
  End of conversation
        |
        v
Save simulation.json (raw transcripts)
        |
        v
EVALUATION:
  For each turn in each conversation:
    Run faithfulness metric (LLM call)
    Run helpfulness metric (LLM call)
    Run coherence metric (LLM call)
    Run relevance metric (LLM call)
    Run goal_completion metric (LLM call)
    Run custom metrics (LLM calls)
        |
        v
Save evaluation.json + final_report.html
```

---

## This Example: BBC News Analyst Agent

### Why this agent?

It tests arksim's ability to catch a specific and important failure mode:
**hallucination vs grounded retrieval**.

When we ran the same scenarios against a plain GPT agent (no RAG), the agent
invented 2023 Premier League match results (Haaland, Saka, fake match dates)
and scored **5/5 on faithfulness** because the evaluator couldn't detect the
fabrication from general knowledge alone.

By routing through a ChromaDB RAG agent with explicit similarity scores in
every response, arksim can now distinguish:
- A grounded answer (similarity 0.45, cites real 2005 article)
- A hallucinated answer (no similarity scores, invents modern players)

### Agent Architecture

```
arksim POST /chat/completions
        |
        v
FastAPI server (server.py)
        |
        v
NewsAnalystAgent.invoke_sync()
  |
  |-- classify_query tool      -- detect BBC category from keywords
  |-- search_by_category tool  -- ChromaDB filtered by category
  |-- search_articles tool     -- ChromaDB broad semantic search
  |-- cross_reference tool     -- two searches + overlap detection
  |-- news_brief tool          -- format retrieved articles into brief
        |
        v
ChromaDB VectorDB/
  500 BBC articles (100 per category)
  text-embedding-3-small vectors
  cosine similarity search
```

### Dataset

**BBC News Articles** (2004-2005) — 2225 articles across 5 categories.

Download: https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

```
Category       Articles   Row Range in CSV
business         510      1    - 510   (sorted first)
entertainment    386      511  - 896
politics         417      897  - 1313
sport            511      1314 - 1824
tech             401      1825 - 2225
```

> The CSV is sorted alphabetically by category. `--max 200` gives 200 business
> articles only. Use `--balanced 100` for equal coverage across all categories.

---

## Test Scenarios

| # | Scenario | Simulated User | Tests |
|---|---|---|---|
| 1 | `semantic_sports_query` | Alex, sports journalist | search_by_category('sport'), real Premier League 2005 articles |
| 2 | `cross_topic_comparison` | Sam, policy analyst | cross_reference tool with Gordon Brown + UK economy |
| 3 | `low_similarity_edge_case` | Jordan, fintech researcher | honesty about blockchain/NFT/crypto (0 articles exist) |
| 4 | `entertainment_deep_dive` | Riley, entertainment journalist | BAFTA 2005: Blanchett, DiCaprio, Shrek 2 citations |
| 5 | `multi_turn_refinement` | Morgan, telecom analyst | 3-turn narrowing: broad tech -> 3G mobile -> revenue figures |

---

## Evaluation Metrics

### Built-in (arksim standard)

| Metric | What it measures |
|---|---|
| `faithfulness` | Did the agent stay factually consistent with available knowledge? |
| `helpfulness` | Did the response actually address the user's need? |
| `coherence` | Was the response logically structured and easy to follow? |
| `relevance` | Did the response answer the question asked? |
| `goal_completion` | Did the user achieve their stated goal by end of conversation? |

### Custom (domain-specific)

| Metric | Type | What it catches |
|---|---|---|
| `retrieval_grounding` | Quant (0-5) | Generic answers not grounded in retrieved articles |
| `similarity_transparency` | Quant (0-5) | Agent hiding low similarity scores from the user |
| `query_strategy_quality` | Quant (0-5) | Using broad search when category filter was obvious |
| `hallucination_detection` | Qual (label) | Facts invented from training data not in ChromaDB |
| `knowledge_gap_honesty` | Qual (label) | Not flagging low/zero similarity results clearly |
| `category_routing_accuracy` | Qual (label) | Wrong category filter applied to a query |

---

## Setup and Run

### Install

```powershell
python -m pip install arksim chromadb openai fastapi uvicorn
```

### Build the index (balanced, one time)

```powershell
cd examples/news-analyst

# Delete old index if rebuilding
Remove-Item -Recurse -Force agent_server\VectorDB

# Build balanced: 100 articles per category = 500 total (~$0.006)
python build_index.py --csv agent_server/data/articles.csv --balanced 100
```

### Terminal 1 — Start agent server

```powershell
$env:OPENAI_API_KEY = "sk-..."
python -m agent_server.chat_completions.server
```

### Terminal 2 — Run arksim

```powershell
$env:OPENAI_API_KEY = "sk-..."
arksim simulate-evaluate config_chromadb.yaml
```

Open `result/evaluation_chromadb/final_report.html` in your browser.

---

## Key Lessons Learned

**1. The knowledge field drives everything**
The `knowledge` array in scenarios.json serves as the evaluator's ground truth.
With 1 item arksim injects it into the simulated user — causing premature STOP.
With 2+ items arksim keeps it evaluator-only. Always use 2 knowledge items.

**2. Config mode determines whether ChromaDB is actually used**
- `config.yaml` → direct OpenAI API → GPT answers from 2024 training data → hallucinations get 5/5
- `config_chromadb.yaml` → local server → real ChromaDB tool calls → grounded answers

**3. Balanced indexing is critical for representative evaluation**
The BBC CSV is sorted by category. `--max 200` = 200 business articles, 0 sport.
The sports scenario will always fail if sport articles aren't in the index.

**4. `agent_behavior_failure` gates qualitative custom metrics**
If `agent_behavior_failure` is in `metrics_to_run`, arksim skips custom qualitative
metrics when built-in scores are high. Remove it to ensure `hallucination_detection`
always runs on every turn.

**5. Similarity scores are the hallucination signal**
A grounded agent always reports cosine similarity (0.0-1.0) for every result.
An agent answering from training data never has similarity scores to report.
This is the key signal the `hallucination_detection` metric looks for.
