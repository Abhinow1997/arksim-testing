# SPDX-License-Identifier: Apache-2.0
"""FAISS-based document retrieval — identical to bank-insurance retriever."""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from .loader import CrawledObject, Loader

logger = logging.getLogger(__name__)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


class FaissRetriever:
    def __init__(self, documents: list[dict[str, Any]], index_dir: str) -> None:
        self.documents = documents
        self.index_dir = Path(index_dir)
        self.index = self._load_or_build()

    def _load_or_build(self) -> faiss.Index:
        index_file = self.index_dir / "index.faiss"
        docs_file = self.index_dir / "docs.pkl"
        if index_file.exists() and docs_file.exists():
            logger.info("Loading existing FAISS index from %s", self.index_dir)
            index = faiss.read_index(str(index_file))
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)
            return index
        if not self.documents:
            raise ValueError("No documents to build index from.")
        logger.info("Building FAISS index from %d documents...", len(self.documents))
        texts = [d["content"] for d in self.documents]
        embeddings = np.array(_embeddings.embed_documents(texts), dtype=np.float32)
        faiss.normalize_L2(embeddings)
        index: faiss.Index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_file))
        with open(docs_file, "wb") as f:
            pickle.dump(self.documents, f)
        logger.info("FAISS index saved to %s", self.index_dir)
        return index

    async def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        embedding = np.array(await _embeddings.aembed_query(query), dtype=np.float32)
        vec = embedding.reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, k)
        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "content": doc["content"],
                    "title": doc.get("metadata", {}).get("title", ""),
                    "source": doc.get("metadata", {}).get("source", ""),
                    "confidence": float(dist),
                })
        return results

    @classmethod
    def load(cls, database_path: str) -> "FaissRetriever":
        pkl_path = Path(database_path) / "agent_knowledge.pkl"
        index_dir = str(Path(database_path) / "index")
        with open(pkl_path, "rb") as f:
            raw_docs: list[CrawledObject] = pickle.load(f)
        documents = [
            {"content": doc.content, "metadata": getattr(doc, "metadata", {})}
            for doc in raw_docs
            if getattr(doc, "content", None) and not getattr(doc, "is_error", False)
        ]
        return cls(documents=documents, index_dir=index_dir)


def build_rag(folder_path: str, rag_docs: list[dict[str, Any]]) -> None:
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, "agent_knowledge.pkl")
    loader = Loader()
    docs: list[Any] = []
    if Path(filepath).exists():
        print(f"Loading existing knowledge from {filepath}")
        with open(filepath, "rb") as f:
            docs = pickle.load(f)
        return
    print("Building new knowledge base...")
    for doc in rag_docs:
        source: str = doc.get("source")
        if doc.get("type") == "local":
            if source.startswith("./"):
                source = os.path.join(folder_path, source.lstrip("./"))
            file_list: list[str] = []
            try:
                if os.path.isfile(source):
                    file_list = [source]
                elif os.path.isdir(source):
                    for root, _, files in os.walk(source):
                        for file in files:
                            if not file.startswith("."):
                                file_list.append(os.path.join(root, file))
            except Exception:
                continue
            if file_list:
                docs.extend(loader.to_crawled_local_objs(file_list))
        elif doc.get("type") == "text":
            docs.extend(loader.to_crawled_text([source]))
    chunked_docs = Loader.chunk(docs)
    Loader.save(filepath, chunked_docs)
