"""Vector store: a thin wrapper around a persistent ChromaDB collection.

Embeddings are computed with OpenAI and stored on disk so the index survives
restarts (no re-embedding on every run).
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import settings


def get_embeddings() -> OpenAIEmbeddings:
    if not settings.has_openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def get_vectorstore() -> Chroma:
    """Return the persistent collection, creating it on first use.

    Cosine distance is used so relevance scores are comparable across queries.
    """
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Document]) -> int:
    """Embed and persist a batch of chunks. Returns the number added."""
    if not chunks:
        return 0
    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)


def reset_collection() -> None:
    """Delete every vector in the collection (used by the 'Clear index' button)."""
    store = get_vectorstore()
    try:
        store.delete_collection()
    except Exception:
        # Collection may not exist yet — nothing to clear.
        pass


def collection_size() -> int:
    """Number of embedded chunks currently stored."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0
