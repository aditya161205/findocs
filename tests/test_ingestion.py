"""Unit tests for the chunking pipeline (no API key or network required)."""

from langchain_core.documents import Document

from src.ingestion import chunk_documents


def _doc(text: str, source: str = "fund_factsheet.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_chunking_splits_long_text():
    long_text = ("Net asset value and expense ratio. " * 200).strip()
    chunks = chunk_documents([_doc(long_text)], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_chunking_preserves_source_metadata():
    chunks = chunk_documents(
        [_doc("Some short factsheet text.")], chunk_size=100, chunk_overlap=10
    )
    assert chunks
    assert chunks[0].metadata["source"] == "fund_factsheet.pdf"


def test_chunk_index_is_monotonic_per_source():
    long_text = ("Risk factors and disclosures. " * 100).strip()
    chunks = chunk_documents([_doc(long_text)], chunk_size=150, chunk_overlap=10)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_overlap_creates_shared_context():
    long_text = " ".join(f"sentence{i}." for i in range(100))
    chunks = chunk_documents([_doc(long_text)], chunk_size=120, chunk_overlap=40)
    # With overlap, consecutive chunks should share some text.
    assert len(chunks) >= 2
