"""
build_index.py - Build ChromaDB index from BBC news articles CSV.

DATASET FACTS (articles.csv):
  Total articles : 2225
  Sorted order   : business (1-510) -> entertainment (511-896)
                   -> politics (897-1313) -> sport (1314-1824)
                   -> tech (1825-2225)

  CRITICAL: --max 200 gives you 200 BUSINESS articles only (no sport/tech/etc).
            Always use --balanced for representative coverage.

RECOMMENDED USAGE:
  # Balanced 500 (100 per category) - best for testing, costs ~$0.006
  python build_index.py --csv agent_server/data/articles.csv --balanced 100

  # Full dataset - complete coverage, costs ~$0.025
  python build_index.py --csv agent_server/data/articles.csv --full

  # Delete old index first if rebuilding
  Remove-Item -Recurse -Force agent_server/VectorDB
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_balanced(csv_path: str, per_category: int, encoding: str = "iso-8859-1") -> list[dict]:
    """Load a balanced sample with equal articles per category."""
    cats: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cat = row.get("category", "").strip()
            if cat:
                cats[cat].append(row)

    print("Dataset category breakdown:")
    rows = []
    for cat in sorted(cats.keys()):
        available = len(cats[cat])
        take = min(per_category, available)
        rows.extend(cats[cat][:take])
        print(f"  {cat:<15}: {available} available, taking {take}")
    print(f"\nTotal articles to index: {len(rows)}")
    return rows


def load_all(csv_path: str, encoding: str = "iso-8859-1") -> list[dict]:
    """Load all articles from the CSV."""
    rows = []
    with open(csv_path, encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def build(csv_path: str, db_path: str, rows: list[dict]) -> None:
    from agent_server.core.chromadb_retriever import ChromaDBRetriever

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    print(f"\nPersistence path: {db_path}")

    # Check if index already exists
    db = Path(db_path) / "chroma.sqlite3"
    if db.exists():
        print(f"\nWARNING: Index already exists at {db_path}")
        print("Delete it first to rebuild:")
        print(f"  Remove-Item -Recurse -Force {db_path}")
        print("\nLoading existing index instead...")
        retriever = ChromaDBRetriever.load(db_path=db_path, api_key=api_key)
        print(f"Existing index has {retriever.count()} articles.")
        return

    # Build from the pre-loaded rows using from_documents
    docs = [
        {
            "content": r.get("content", "").strip(),
            "title": r.get("title", "").strip(),
            "category": r.get("category", "").strip(),
            "source": r.get("filename", csv_path),
        }
        for r in rows
        if r.get("content", "").strip()
    ]
    retriever = ChromaDBRetriever.from_documents(
        documents=docs,
        db_path=db_path,
        api_key=api_key,
    )

    info = retriever.info()
    print("\nIndex built successfully!")
    print(f"  Collection  : {info['collection_name']}")
    print(f"  Documents   : {info['document_count']}")
    print(f"  Model       : {info['embedding_model']}")
    print(f"  Saved to    : {info['db_path']}")

    # Quick test query per category
    print("\nQuick verification queries:")
    tests = [
        ("football Premier League Chelsea Arsenal", "sport"),
        ("Gordon Brown budget chancellor economy", "politics"),
        ("Nokia mobile phone 3G network", "tech"),
        ("BAFTA film Oscar ceremony nominees", "entertainment"),
        ("profit revenue quarterly earnings", "business"),
    ]
    for query, expected_cat in tests:
        results = retriever.retrieve_by_category(query, category=expected_cat, k=2)
        if results:
            top = results[0]
            print(f"  [{expected_cat}] '{query[:35]}...' -> sim={top['similarity']:.3f} | {top['title'][:45]}")
        else:
            print(f"  [{expected_cat}] '{query[:35]}...' -> NO RESULTS (check indexing)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB index from BBC news CSV")
    parser.add_argument("--csv", required=True, help="Path to articles.csv")
    parser.add_argument("--db", default="agent_server/VectorDB", help="ChromaDB persistence path")
    parser.add_argument("--balanced", type=int, metavar="N",
                        help="Take N articles per category (balanced). Recommended: 100-200.")
    parser.add_argument("--full", action="store_true",
                        help="Index all 2225 articles (complete dataset, ~$0.025 cost)")
    parser.add_argument("--max", type=int,
                        help="(LEGACY - DO NOT USE: gives only business articles). Use --balanced instead.")
    args = parser.parse_args()

    if args.max and not args.balanced and not args.full:
        print("WARNING: --max gives only business articles (CSV is sorted by category).")
        print("         Use --balanced 100 for equal coverage across all 5 categories.")
        print()

    print("=" * 60)
    print("BBC News Article ChromaDB Index Builder")
    print("=" * 60)
    print(f"CSV source: {args.csv}")

    if args.full:
        print("Mode: FULL (all 2225 articles)")
        rows = load_all(args.csv)
    elif args.balanced:
        print(f"Mode: BALANCED ({args.balanced} per category)")
        rows = load_balanced(args.csv, per_category=args.balanced)
    elif args.max:
        print(f"Mode: LEGACY MAX ({args.max} rows from top of CSV = business only)")
        rows_all = load_all(args.csv)
        rows = rows_all[:args.max]
    else:
        print("ERROR: Specify --balanced N, --full, or --max N")
        print("  Recommended: python build_index.py --csv articles.csv --balanced 100")
        sys.exit(1)

    build(csv_path=args.csv, db_path=args.db, rows=rows)
