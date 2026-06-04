"""Document loading and chunking.

Turns raw files (PDF / TXT / MD) into small, overlapping ``Document`` chunks that
each carry enough metadata (source file + page number) to build a citation later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from .config import settings

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _load_file(path: Path) -> list[Document]:
    """Load a single file into one ``Document`` per page (PDF) or per file (text)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {suffix} ({path.name})")

    # Normalise metadata so downstream code can rely on consistent keys.
    for d in docs:
        d.metadata["source"] = path.name
        # PyPDFLoader uses a 0-based "page"; expose a human 1-based page too.
        if "page" in d.metadata:
            d.metadata["page_label"] = d.metadata.get("page_label", d.metadata["page"] + 1)
    return docs


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    """Load every supported file in ``paths`` (files or directories)."""
    documents: list[Document] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in SUPPORTED_SUFFIXES]
        else:
            files = [path]
        for f in files:
            documents.extend(_load_file(f))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks with stable, citable metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        # Split on natural boundaries first, falling back to characters.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Tag each chunk with a per-source index — useful for debugging & dedup.
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = per_source_counter.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        per_source_counter[src] = idx + 1
    return chunks


def build_chunks(paths: Iterable[str | Path]) -> list[Document]:
    """Convenience: load + chunk in one call."""
    return chunk_documents(load_documents(paths))
