# SPDX-License-Identifier: Apache-2.0
"""
News Analyst Agent — Pure OpenAI Function Calling
===================================================
Rewritten to use the standard `openai` package only.
No openai-agents SDK required.

Tool chain per query (enforced via system prompt):
  Step 1 – classify_query    : detect category + search strategy
  Step 2 – search_articles   : ChromaDB broad semantic search
       OR  search_by_category : ChromaDB category-filtered search
  Step 3 – cross_reference   : two-query overlap (when comparing topics)
  Step 4 – news_brief        : assemble structured brief from retrieved articles

The agent runs a proper tool-calling loop:
  LLM → tool_calls → execute real ChromaDB functions → feed results back → repeat
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import openai

from .chromadb_retriever import ChromaDBRetriever

# ── Database paths ─────────────────────────────────────────────────────────────
_AGENT_SERVER_DIR = Path(__file__).parent.parent
_DB_PATH = str(_AGENT_SERVER_DIR / "VectorDB")
_CSV_PATH = str(_AGENT_SERVER_DIR / "data" / "articles.csv")

_retriever: ChromaDBRetriever | None = None


def _get_retriever() -> ChromaDBRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever
    try:
        _retriever = ChromaDBRetriever.load(db_path=_DB_PATH)
        return _retriever
    except ValueError:
        pass
    if os.path.exists(_CSV_PATH):
        _retriever = ChromaDBRetriever.from_csv(
            csv_path=_CSV_PATH, db_path=_DB_PATH, encoding="iso-8859-1"
        )
        return _retriever
    raise RuntimeError(
        f"ChromaDB index not found at {_DB_PATH}. "
        "Run: python build_index.py --csv agent_server/data/articles.csv"
    )


# ── BBC Categories ─────────────────────────────────────────────────────────────
BBC_CATEGORIES = {"sport", "politics", "business", "tech", "entertainment"}

TOPIC_TO_CATEGORY: dict[str, str] = {
    "football": "sport", "cricket": "sport", "tennis": "sport",
    "rugby": "sport", "olympics": "sport", "premier league": "sport",
    "election": "politics", "parliament": "politics", "government": "politics",
    "minister": "politics", "labour": "politics", "conservative": "politics",
    "economy": "business", "gdp": "business", "inflation": "business",
    "market": "business", "stock": "business", "profit": "business",
    "software": "tech", "internet": "tech", "mobile": "tech",
    "broadband": "tech", "google": "tech", "apple": "tech", "microsoft": "tech",
    "film": "entertainment", "music": "entertainment", "oscar": "entertainment",
    "bafta": "entertainment", "celebrity": "entertainment",
}


# ── Tool implementations (plain functions — no SDK decorator needed) ───────────

def _tool_classify_query(query: str) -> dict:
    """Classify query → detect category + strategy."""
    q_lower = query.lower()
    detected = None
    for kw, cat in TOPIC_TO_CATEGORY.items():
        if kw in q_lower:
            detected = cat
            break

    needs_xref = any(w in q_lower for w in ["compare", "versus", "vs", "both", "overlap"])
    reformulated = [query]
    if len(query.split()) <= 3:
        reformulated += [f"news about {query}", f"report on {query}"]

    return {
        "original_query": query,
        "detected_category": detected,
        "available_categories": sorted(BBC_CATEGORIES),
        "search_strategy": "category_filtered" if detected else "semantic_broad",
        "needs_cross_reference": needs_xref,
        "reformulated_queries": reformulated,
        "tip": (
            f"Use search_by_category with '{detected}'"
            if detected else "Use search_articles for broad semantic search"
        ),
    }


def _tool_search_articles(query: str, k: int = 5) -> dict:
    """Broad semantic search across all articles in ChromaDB."""
    retriever = _get_retriever()
    k = max(1, min(k, 10))
    results = retriever.retrieve(query, k=k)

    if not results:
        return {
            "query": query,
            "total_indexed": retriever.count(),
            "results": [],
            "warning": "No articles found. Index may be empty.",
        }

    return {
        "query": query,
        "total_indexed": retriever.count(),
        "results": [
            {
                "rank": i + 1,
                "similarity": r["similarity"],
                "category": r["category"],
                "title": r["title"],
                # Full content so the LLM can cite specific facts
                "content": r["content"],
            }
            for i, r in enumerate(results)
        ],
    }


def _tool_search_by_category(query: str, category: str, k: int = 5) -> dict:
    """Category-filtered semantic search in ChromaDB."""
    retriever = _get_retriever()
    category = category.lower().strip()

    if category not in BBC_CATEGORIES:
        return {
            "error": f"Unknown category '{category}'.",
            "valid_categories": sorted(BBC_CATEGORIES),
        }

    k = max(1, min(k, 10))
    results = retriever.retrieve_by_category(query, category=category, k=k)

    if not results:
        return {
            "query": query,
            "category": category,
            "results": [],
            "warning": f"No articles in category '{category}' match this query.",
        }

    return {
        "query": query,
        "category_filter": category,
        "results": [
            {
                "rank": i + 1,
                "similarity": r["similarity"],
                "category": r["category"],
                "title": r["title"],
                "content": r["content"],
            }
            for i, r in enumerate(results)
        ],
    }


def _tool_cross_reference(query_a: str, query_b: str) -> dict:
    """Run two searches and surface overlapping articles."""
    retriever = _get_retriever()
    res_a = retriever.retrieve(query_a, k=5)
    res_b = retriever.retrieve(query_b, k=5)

    ids_a = {r["content"][:80] for r in res_a}
    ids_b = {r["content"][:80] for r in res_b}

    return {
        "query_a": {
            "query": query_a,
            "results": [
                {"similarity": r["similarity"], "category": r["category"], "content": r["content"]}
                for r in res_a
            ],
        },
        "query_b": {
            "query": query_b,
            "results": [
                {"similarity": r["similarity"], "category": r["category"], "content": r["content"]}
                for r in res_b
            ],
        },
        "overlap_count": len(ids_a & ids_b),
        "note": "overlap_count > 0 means the same article was retrieved by both queries.",
    }


def _tool_news_brief(topic: str, search_results_json: str, user_question: str) -> dict:
    """Format retrieved articles into a structured brief prompt."""
    try:
        data = json.loads(search_results_json)
        results = data.get("results", [])
    except (json.JSONDecodeError, AttributeError):
        return {"error": "Invalid search_results_json — run a search tool first."}

    if not results:
        return {"error": "No results to summarise. Run search_articles or search_by_category first."}

    # Build article evidence block
    evidence = ""
    for r in results[:5]:
        sim = r.get("similarity", "?")
        cat = r.get("category", "?")
        content = r.get("content", "")[:600]
        evidence += f"\n[Similarity={sim} | Category={cat}]\n{content}\n---\n"

    return {
        "brief_instructions": (
            f"Write a structured news brief on: {topic}\n"
            f"User question: {user_question}\n\n"
            f"SOURCE ARTICLES FROM CHROMADB (use ONLY these — do not use training knowledge):\n"
            f"{evidence}\n\n"
            "Brief structure:\n"
            "1. HEADLINE SUMMARY (1 sentence from the articles)\n"
            "2. KEY FACTS (3-4 bullets — specific names, numbers, dates from articles)\n"
            "3. CONTEXT (2 sentences — background from articles)\n"
            "4. DATA GAPS: if similarity scores are below 0.3 for any result, explicitly note this\n"
            "5. SIMILARITY SCORES: list all scores so user can judge result quality\n\n"
            "CRITICAL: every fact must be traceable to the above articles. "
            "Never use names, scores, or dates from your training data."
        ),
        "top_similarity": results[0].get("similarity", 0),
        "source_count": len(results),
        "low_confidence_warning": (
            "SIMILARITY SCORES ARE LOW (<0.3) — results may not be relevant"
            if results[0].get("similarity", 1) < 0.3 else None
        ),
    }


# ── OpenAI function schemas ────────────────────────────────────────────────────
# These tell the LLM what tools exist and how to call them.

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "classify_query",
            "description": (
                "ALWAYS call this first. Classifies the user query to detect "
                "the news category (sport/politics/business/tech/entertainment) "
                "and determine the best search strategy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's news query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_articles",
            "description": (
                "Semantic search over ALL indexed BBC news articles in ChromaDB. "
                "Use when the category is unclear or the query is broad."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "Number of results (1-10).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_category",
            "description": (
                "Category-filtered semantic search in ChromaDB. "
                "Use when the category is clearly identified (e.g. football → sport). "
                "More precise than search_articles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["sport", "politics", "business", "tech", "entertainment"],
                    },
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_reference",
            "description": (
                "Run TWO separate ChromaDB searches and find overlapping articles. "
                "Use only when the user asks to compare or connect two topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_a": {"type": "string", "description": "First topic."},
                    "query_b": {"type": "string", "description": "Second topic."},
                },
                "required": ["query_a", "query_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news_brief",
            "description": (
                "Format ChromaDB search results into a structured news brief. "
                "Always call this last, after you have search results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Main topic of the brief."},
                    "search_results_json": {
                        "type": "string",
                        "description": "JSON string of the search tool output.",
                    },
                    "user_question": {
                        "type": "string",
                        "description": "The original user question.",
                    },
                },
                "required": ["topic", "search_results_json", "user_question"],
            },
        },
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _dispatch_tool(name: str, arguments: str) -> str:
    """Execute a tool by name and return JSON string result."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON arguments for tool '{name}'"})

    try:
        if name == "classify_query":
            result = _tool_classify_query(**args)
        elif name == "search_articles":
            result = _tool_search_articles(**args)
        elif name == "search_by_category":
            result = _tool_search_by_category(**args)
        elif name == "cross_reference":
            result = _tool_cross_reference(**args)
        elif name == "news_brief":
            result = _tool_news_brief(**args)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": f"Tool '{name}' raised: {exc}"}

    return json.dumps(result, indent=2)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a News Research Analyst with access to a ChromaDB index of 
BBC news articles (sport, politics, business, tech, entertainment) from 2004-2005.

MANDATORY WORKFLOW FOR EVERY QUERY:
1. classify_query  → always call this first to identify category and strategy
2. search_by_category (if category detected) OR search_articles (if broad/unclear)
3. cross_reference → only if the user asks to compare or connect two topics
4. news_brief → always call this last to format the final response

CRITICAL RULES:
- NEVER answer from your training knowledge — ONLY use what ChromaDB returns
- ALWAYS include the similarity scores in your final response
- If top similarity < 0.3, explicitly say: "Low confidence — results may not be relevant"
- If asked about topics from after 2005, say: "This topic is not in the 2004-2005 BBC archive"
- Cite specific article content (exact wording, names, numbers) in every response

The BBC archive covers: Premier League football (2004-05 season), UK politics under 
Blair/Brown, FTSE/UK business, early internet/mobile tech, and entertainment/BAFTA.
Players like Haaland, Saka, and events from 2022+ DO NOT exist in this archive."""


# ── Agent class ────────────────────────────────────────────────────────────────

class NewsAnalystAgent:
    """
    Stateful News Analyst Agent using standard OpenAI function calling.

    Runs a synchronous tool-calling loop:
      1. Send message history to GPT
      2. If tool_calls in response → execute ChromaDB functions → add results
      3. Repeat until GPT produces a final text response (finish_reason='stop')
    """

    MAX_TOOL_ITERATIONS = 10   # safety limit to prevent infinite loops

    def __init__(
        self,
        context_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        model: str = "gpt-4o",
    ) -> None:
        self._client = openai.OpenAI()   # sync client — works in both sync and async contexts
        self.context_id = context_id or str(uuid.uuid4())
        self._history: list[dict[str, Any]] = list(history) if history else []
        self._model = model

    def invoke_sync(self, question: str) -> str:
        """
        Process a user question through the full tool-calling loop.
        Returns the agent's final answer as a string.
        """
        self._history.append({"role": "user", "content": question})

        # Build working message list: system + full history
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + self._history

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            choice = response.choices[0]
            msg = choice.message

            # Add assistant message to working history
            messages.append(msg.model_dump(exclude_unset=False))

            # ── Done: no tool calls ────────────────────────────────────────
            if choice.finish_reason == "stop" or not msg.tool_calls:
                answer = msg.content or ""
                self._history.append({"role": "assistant", "content": answer})
                return answer

            # ── Execute each tool call ─────────────────────────────────────
            for tc in msg.tool_calls:
                tool_result = _dispatch_tool(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        # Safety fallback if loop exhausted
        fallback = "Reached maximum tool iterations. Please try a more specific query."
        self._history.append({"role": "assistant", "content": fallback})
        return fallback

    async def invoke(self, question: str) -> str:
        """Async wrapper — runs the synchronous invoke in the current thread."""
        # OpenAI's sync client is fine here; arksim's executor handles the thread
        return self.invoke_sync(question)
