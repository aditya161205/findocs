"""Central configuration.

All tunables live here so they can be explained in one place and overridden via
environment variables (loaded from a local ``.env`` file if present).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
VECTORSTORE_DIR = PROJECT_ROOT / ".chroma"


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Runtime settings. Read once at import time; override via env vars."""

    # --- LLM / embeddings provider ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Chunking ---
    # 1,000 chars (~250 tokens) keeps a single factsheet table or paragraph intact;
    # 150-char overlap preserves context that straddles a chunk boundary.
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))

    # --- Retrieval ---
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 4))
    # Relevance below this is treated as "not in the documents".
    min_relevance: float = field(default_factory=lambda: _get_float("MIN_RELEVANCE", 0.20))

    # --- Vector store ---
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "financial_docs")
    )
    persist_dir: str = field(default_factory=lambda: os.getenv("PERSIST_DIR", str(VECTORSTORE_DIR)))

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())


settings = Settings()
