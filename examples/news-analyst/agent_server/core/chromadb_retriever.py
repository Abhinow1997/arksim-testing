# SPDX-License-Identifier: Apache-2.0
"""
ChromaDB-based News Article Retriever
=======================================
Directly based on the chromadb_news_articles.ipynb notebook pattern:

  CSV  ──► Polars DataFrame
         ──► OpenAI text-embedding-3-small  (1536-dim vectors)
         ──► ChromaDB PersistentClient      (HNSW cosine index)
         ──► Semantic query → top-k articles

Key design decisions mirroring the notebook:
  - PersistentClient (not in-memory) — survives process restarts
  - OpenAIEmbeddingFunction registered at collection level
  - cosine distance space (hnsw:space = cosine)
  - Idempotent build: skip re-indexing if collection already populated
  - Supports both CSV ingestion (Polars) and plain text document lists
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "news_articles"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DB_PATH = "VectorDB"    # matches notebook's PersistentClient path


class ChromaDBRetriever:
    """
    ChromaDB-based semantic retriever for news articles.

    Mirrors the notebook pipeline:
      1. PersistentClient saves index to disk (survives restarts)
      2. OpenAIEmbeddingFunction handles embed-at-query-time automatically
      3. HNSW cosine index for fast approximate nearest-neighbor search

    Usage:
        # Build once from CSV
        retriever = ChromaDBRetriever.from_csv("articles.csv", db_path="VectorDB")

        # Load existing index on restart
        retriever = ChromaDBRetriever.load(db_path="VectorDB")

        # Query
        results = retriever.retrieve("economic inflation rates", k=3)
    """

    def __init__(self, db_path: str, api_key: str | None = None) -> None:
        """
        Args:
            db_path: Directory where ChromaDB stores chroma.sqlite3 + HNSW files.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        self.db_path = db_path
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        # Embedding function — registered at collection level (notebook Step 4 pattern)
        self._openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self._api_key,
            model_name=EMBEDDING_MODEL,
        )

        # PersistentClient — survives restarts (notebook Step 8 pattern)
        self._client = chromadb.PersistentClient(path=db_path)

        # Get or create the collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._openai_ef,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB collection '%s' loaded — %d documents indexed",
            COLLECTION_NAME,
            self._collection.count(),
        )

    # ── Build from CSV (Polars, notebook Step 3 pattern) ────────────────────

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        db_path: str = DEFAULT_DB_PATH,
        text_column: str | None = None,
        title_column: str | None = None,
        category_column: str | None = None,
        encoding: str = "iso-8859-1",
        max_articles: int | None = None,
        api_key: str | None = None,
    ) -> "ChromaDBRetriever":
        """
        Build a ChromaDB index from a CSV file using Polars (notebook Step 3-6 pattern).

        Automatically detects BBC-format columns (ArticleId, Article, Category).
        For other CSVs pass text_column explicitly.

        Args:
            csv_path: Path to articles CSV.
            db_path: Where to persist the ChromaDB index.
            text_column: Column containing article text. Auto-detected if None.
            title_column: Optional column for article titles (stored as metadata).
            category_column: Optional column for article category (stored as metadata).
            encoding: CSV encoding — BBC dataset uses 'iso-8859-1'.
            max_articles: Limit articles for dev/testing (None = all).
            api_key: OpenAI API key.

        Returns:
            Ready-to-query ChromaDBRetriever.
        """
        retriever = cls(db_path=db_path, api_key=api_key)

        # Skip re-indexing if already populated (notebook idempotency pattern)
        if retriever._collection.count() > 0:
            logger.info(
                "Collection already has %d articles — skipping re-indexing",
                retriever._collection.count(),
            )
            return retriever

        # ── Load CSV with Python's csv module ────────────────────────────────
        # Polars chunked reader chokes on embedded newlines in the content
        # field. Python's csv.DictReader handles them correctly because it
        # tracks quoting state across line boundaries.
        logger.info("Loading CSV: %s", csv_path)
        import csv

        rows: list[dict[str, str]] = []
        with open(csv_path, encoding=encoding, errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                rows.append(row)
                if max_articles and len(rows) >= max_articles:
                    break

        if not rows:
            raise ValueError(f"No rows found in CSV: {csv_path}")

        logger.info("Loaded %d articles, columns: %s", len(rows), list(rows[0].keys()))

        # ── Auto-detect column names ─────────────────────────────────────────
        cols_lower = {c.lower(): c for c in rows[0].keys()}

        if text_column is None:
            for candidate in ["content", "article", "text", "body", "description"]:
                if candidate in cols_lower:
                    text_column = cols_lower[candidate]
                    break
            if text_column is None:
                text_column = list(rows[0].keys())[-1]   # last column as fallback
        logger.info("Using text column: '%s'", text_column)

        if title_column is None:
            for candidate in ["title", "headline", "name", "subject"]:
                if candidate in cols_lower:
                    title_column = cols_lower[candidate]
                    break

        if category_column is None:
            for candidate in ["category", "topic", "section", "label", "class"]:
                if candidate in cols_lower:
                    category_column = cols_lower[candidate]
                    break

        # ── Prepare documents and metadata ───────────────────────────────────
        ids = [f"ID{i + 1}" for i in range(len(rows))]
        documents = [str(row.get(text_column, "")).strip() for row in rows]

        metadatas = []
        for row in rows:
            meta: dict[str, str] = {}
            if title_column:
                meta["title"] = str(row.get(title_column, "")).strip()
            if category_column:
                meta["category"] = str(row.get(category_column, "")).strip()
            meta["source"] = csv_path
            metadatas.append(meta)

        # ── Add to ChromaDB (embedding happens automatically via openai_ef) ──
        # Batch in chunks of 100 to avoid API rate limits
        batch_size = 100
        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            retriever._collection.add(
                documents=documents[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )
            logger.info("Indexed articles %d–%d / %d", start + 1, end, total)

        logger.info(
            "ChromaDB index built — %d articles in '%s'",
            retriever._collection.count(),
            db_path,
        )
        return retriever

    @classmethod
    def from_documents(
        cls,
        documents: list[dict[str, Any]],
        db_path: str = DEFAULT_DB_PATH,
        api_key: str | None = None,
    ) -> "ChromaDBRetriever":
        """
        Build index from a list of dicts with 'content', optional 'title', 'category'.
        Used when you have pre-processed text (e.g., from the pe-orgair data files).

        Args:
            documents: List of dicts, each with at minimum a 'content' key.
            db_path: ChromaDB persistence directory.
            api_key: OpenAI API key.
        """
        retriever = cls(db_path=db_path, api_key=api_key)

        if retriever._collection.count() > 0:
            logger.info(
                "Collection already has %d docs — skipping re-indexing",
                retriever._collection.count(),
            )
            return retriever

        texts = [d["content"] for d in documents]
        ids = [f"DOC{i}" for i in range(len(documents))]
        metadatas = [
            {
                "title": d.get("title", ""),
                "category": d.get("category", ""),
                "source": d.get("source", ""),
            }
            for d in documents
        ]

        batch_size = 100
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            retriever._collection.add(
                documents=texts[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info("Indexed %d documents", retriever._collection.count())
        return retriever

    @classmethod
    def load(
        cls,
        db_path: str = DEFAULT_DB_PATH,
        api_key: str | None = None,
    ) -> "ChromaDBRetriever":
        """
        Load an existing ChromaDB index from disk.
        Raises ValueError if the collection is empty (not yet built).
        """
        retriever = cls(db_path=db_path, api_key=api_key)
        if retriever._collection.count() == 0:
            raise ValueError(
                f"ChromaDB collection at '{db_path}' is empty. "
                "Build the index first with ChromaDBRetriever.from_csv() or .from_documents()."
            )
        return retriever

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Semantic search over indexed articles (notebook Step 7 pattern).

        The query is embedded automatically by openai_ef (same model used at
        index time — critical for meaningful cosine similarity scores).

        Args:
            query: Natural language question or keyword query.
            k: Number of results to return.

        Returns:
            List of dicts: {content, title, category, source, similarity}
            Sorted by descending similarity (highest first).
        """
        if self._collection.count() == 0:
            logger.warning("Collection is empty — no results available")
            return []

        k = min(k, self._collection.count())

        results = self._collection.query(
            query_texts=[query],   # openai_ef embeds automatically
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )

        output: list[dict[str, Any]] = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # ChromaDB cosine distance → similarity (notebook: similarity = 1.0 - dist)
            similarity = round(1.0 - dist, 4)
            output.append({
                "content": doc,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "source": meta.get("source", ""),
                "similarity": similarity,
            })

        return output

    def retrieve_by_category(
        self, query: str, category: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Semantic search filtered to a specific news category.
        Uses ChromaDB's where clause for metadata filtering.

        Args:
            query: Natural language query.
            category: Category filter (e.g., 'sport', 'politics', 'business').
            k: Number of results.
        """
        k = min(k, self._collection.count())
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where={"category": {"$eq": category}},
            include=["documents", "distances", "metadatas"],
        )
        output = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            output.append({
                "content": doc,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "source": meta.get("source", ""),
                "similarity": round(1.0 - dist, 4),
            })
        return output

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of indexed documents."""
        return self._collection.count()

    def info(self) -> dict[str, Any]:
        """Return collection metadata and stats."""
        return {
            "collection_name": COLLECTION_NAME,
            "db_path": self.db_path,
            "document_count": self._collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "distance_metric": "cosine",
        }
