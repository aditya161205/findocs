"""Central configuration.

All tunables live here so they can be explained in one place and overridden via
environment variables (loaded from a local ``.env`` file if present).

Two LLM providers are supported:
  * ``openai`` — text-embedding-3-small + gpt-4o-mini
  * ``google`` — Gemini, which has a free tier (no credit card)
The provider is auto-detected from whichever API key is present, or can be
forced with ``LLM_PROVIDER``.
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

# Per-provider default model names.
DEFAULT_MODELS = {
    "openai": {"chat": "gpt-4o-mini", "embedding": "text-embedding-3-small"},
    "google": {"chat": "gemini-2.5-flash", "embedding": "models/gemini-embedding-001"},
}


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


def _detect_provider() -> str:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced in DEFAULT_MODELS:
        return forced
    # Auto-detect: prefer the free option (Google) if its key is present.
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    return "openai"


@dataclass
class Settings:
    """Runtime settings. Read once at import time; override via env vars."""

    # --- Provider / keys ---
    provider: str = field(default_factory=_detect_provider)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    # --- Models (default per provider, overridable) ---
    _chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", ""))
    _embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", ""))

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
    def chat_model(self) -> str:
        return self._chat_model or DEFAULT_MODELS[self.provider]["chat"]

    @property
    def embedding_model(self) -> str:
        return self._embedding_model or DEFAULT_MODELS[self.provider]["embedding"]

    @property
    def api_key(self) -> str:
        return self.google_api_key if self.provider == "google" else self.openai_api_key

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


settings = Settings()
