"""Command-line ingestion: index documents without launching the UI.

Usage:
    python ingest.py data/sample_docs
    python ingest.py path/to/file.pdf path/to/another.pdf
    python ingest.py --reset data/sample_docs
"""

from __future__ import annotations

import argparse
import sys

from src.config import settings
from src.ingestion import build_chunks
from src.vectorstore import add_chunks, collection_size, reset_collection


def main() -> int:
    parser = argparse.ArgumentParser(description="Index financial documents into ChromaDB.")
    parser.add_argument("paths", nargs="+", help="Files or directories to index.")
    parser.add_argument("--reset", action="store_true", help="Clear the index first.")
    args = parser.parse_args()

    if not settings.has_openai_key:
        print("ERROR: OPENAI_API_KEY is not set. See .env.example.", file=sys.stderr)
        return 1

    if args.reset:
        print("Clearing existing index…")
        reset_collection()

    print("Loading & chunking documents…")
    chunks = build_chunks(args.paths)
    if not chunks:
        print("No supported documents found (.pdf, .txt, .md).", file=sys.stderr)
        return 1

    print(f"Embedding & storing {len(chunks)} chunks…")
    add_chunks(chunks)
    print(f"Done. Index now holds {collection_size()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
