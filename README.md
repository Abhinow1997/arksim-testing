# arksim-testing

Exploring the **arksim** agent simulation and evaluation framework as part of
DAMG 7245 Big Data Intelligence Analytics at Northeastern University.

---

## What is arksim?

arksim lets you test AI agents by running **simulated conversations** and
**automatically scoring** the results — no manual review needed.

```
Simulated User  <-->  Your Agent  -->  Evaluator  -->  HTML Report
  (LLM playing       (your RAG,       (LLM judges      (scores per
   a persona)         chatbot,         each turn)        turn + goal)
                       etc.)
```

Core concept: define a test scenario once, arksim drives the conversation
and tells you exactly where your agent passed or failed.

---

## Examples

### news-analyst (primary)

A BBC News RAG agent backed by ChromaDB, tested with arksim.

**What it demonstrates:**
- Catching hallucinations that standard evaluators miss
- Custom metrics for RAG-specific failure modes (retrieval grounding, similarity transparency)
- The difference between direct LLM mode and grounded tool-call mode
- Balanced dataset indexing for representative evaluation

**Stack:** ChromaDB + OpenAI embeddings + FastAPI + arksim

See [examples/news-analyst/README.md](examples/news-analyst/README.md)

### bank-insurance / e-commerce

Official arksim examples included for reference.

---

## Quick Start

```powershell
cd examples/news-analyst
python -m pip install arksim chromadb openai fastapi uvicorn

# Build index
python build_index.py --csv agent_server/data/articles.csv --balanced 100

# Terminal 1
$env:OPENAI_API_KEY = "sk-..."
python -m agent_server.chat_completions.server

# Terminal 2
$env:OPENAI_API_KEY = "sk-..."
arksim simulate-evaluate config_chromadb.yaml
```

---

## Dataset

BBC News Articles (2004-2005): https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

Place `articles.csv` in `examples/news-analyst/agent_server/data/`
(not tracked in git — download separately)
