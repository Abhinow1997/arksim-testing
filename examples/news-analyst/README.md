# BBC News Analyst Agent - ChromaDB RAG + arksim Evaluation

A production-style RAG agent built on **ChromaDB** and tested with the **arksim** simulation-evaluation framework. The agent searches a BBC news archive (2004-2005) using OpenAI semantic embeddings and answers questions grounded strictly in retrieved articles.

Built as part of the **DAMG 7245 - Big Data Intelligence Analytics** course at Northeastern University.

---

## What This Does

```
User Query
    |
    v
classify_query         -- detect BBC category (sport/politics/business/tech/entertainment)
    |
    v
search_by_category     -- ChromaDB cosine similarity search (filtered)
 OR search_articles    -- ChromaDB broad semantic search
    |
    v (optional)
cross_reference        -- two-query overlap for comparative questions
    |
    v
news_brief             -- structured brief from retrieved articles only
    |
    v
arksim Simulator       -- 5 adversarial simulated-user conversations
    |
    v
arksim Evaluator       -- 6 built-in + 6 custom metrics -> HTML report
```

---

## Dataset

**BBC News Articles** - 2225 articles across 5 categories from 2004-2005.

Download from Kaggle: https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

Place `articles.csv` in `agent_server/data/`.

| Category      | Articles | Row Range  |
|---------------|----------|------------|
| business      | 510      | 1 - 510    |
| entertainment | 386      | 511 - 896  |
| politics      | 417      | 897 - 1313 |
| sport         | 511      | 1314 - 1824|
| tech          | 401      | 1825 - 2225|

> **Important:** The CSV is sorted by category. Using `--max 200` gives only business articles. Always use `--balanced` for representative coverage.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

### 3. Download the dataset and place it

```
agent_server/data/articles.csv
```

### 4. Build the ChromaDB index

```powershell
# Balanced 500 articles (100 per category) - recommended
python build_index.py --csv agent_server/data/articles.csv --balanced 100

# Full 2225 articles (~$0.025 in embedding costs)
python build_index.py --csv agent_server/data/articles.csv --full

# If rebuilding, delete the old index first
Remove-Item -Recurse -Force agent_server\VectorDB
python build_index.py --csv agent_server/data/articles.csv --balanced 100
```

### 5. Start the agent server (Terminal 1)

```powershell
python -m agent_server.chat_completions.server
```

### 6. Run arksim evaluation (Terminal 2)

```powershell
arksim simulate-evaluate config_chromadb.yaml
```

Open `results/evaluation_chromadb/final_report.html` in your browser.

---

## Project Structure

```
news-analyst/
|-- build_index.py              # Build/rebuild ChromaDB index from CSV
|-- config.yaml                 # Direct OpenAI mode (no ChromaDB, for baseline)
|-- config_chromadb.yaml        # Grounded ChromaDB mode (use this)
|-- custom_agent.py             # arksim BaseAgent wrapper
|-- custom_metrics.py           # 6 domain-specific evaluation metrics
|-- requirements.txt            # Python dependencies
|-- run_pipeline.py             # Programmatic pipeline runner
|-- scenarios.json              # 5 adversarial test scenarios
|-- start_server.ps1            # PowerShell server startup script
|-- agent_server/
|   |-- chat_completions/
|   |   |-- server.py           # FastAPI Chat Completions endpoint
|   |-- core/
|   |   |-- agent.py            # NewsAnalystAgent (OpenAI function calling loop)
|   |   |-- chromadb_retriever.py  # ChromaDB wrapper (from notebook)
|   |-- data/                   # Place articles.csv here (not tracked in git)
|   |-- VectorDB/               # ChromaDB index files (not tracked in git)
|   |-- requirements.txt        # Server-specific dependencies
```

---

## Test Scenarios

| Scenario | Tests |
|---|---|
| `semantic_sports_query` | Category-filtered search for Premier League football (2004-05 season) |
| `cross_topic_comparison` | Cross-reference tool: Gordon Brown budget + UK economy overlap |
| `low_similarity_edge_case` | Honest handling of out-of-scope topics (blockchain/NFT/crypto) |
| `entertainment_deep_dive` | BAFTA 2005 article citation (Blanchett, DiCaprio, Shrek 2) |
| `multi_turn_refinement` | Stateful 3-turn refinement: broad tech -> 3G mobile -> revenue figures |

---

## Custom Evaluation Metrics

### Quantitative (0-5 scale)
| Metric | What It Catches |
|---|---|
| `retrieval_grounding` | Did the agent cite actual article content or answer generically? |
| `similarity_transparency` | Did the agent report cosine similarity scores and flag low confidence? |
| `query_strategy_quality` | Did the agent use category filters instead of always doing broad search? |

### Qualitative (categorical labels)
| Metric | Labels |
|---|---|
| `hallucination_detection` | `clean` / `hallucinated` / `uncertain` |
| `knowledge_gap_honesty` | `handled_honestly` / `handled_poorly` / `not_applicable` |
| `category_routing_accuracy` | `optimal` / `suboptimal` / `incorrect` / `not_applicable` |

---

## Key Design Decisions

**ChromaDB over FAISS:** The project notebook (`chromadb_news_articles.ipynb`) uses ChromaDB with `PersistentClient` and `OpenAIEmbeddingFunction`. This example preserves that exact pattern end-to-end.

**Two config modes:**
- `config.yaml` — calls OpenAI API directly. GPT answers from training data. No ChromaDB. Use only for baseline/persona testing.
- `config_chromadb.yaml` — routes through the local FastAPI server. Forces real ChromaDB tool calls. Use for grounded evaluation.

**Balanced indexing matters:** The BBC CSV is alphabetically sorted by category. `--max 200` gives 200 business articles and zero sport/tech articles. Always use `--balanced 100` or higher.

---

## Based On

- [arksim](https://pypi.org/project/arksim/) - Agent simulation and evaluation framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [BBC News Archive](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) - Kaggle dataset
- Course: DAMG 7245 Big Data Intelligence Analytics, Northeastern University
