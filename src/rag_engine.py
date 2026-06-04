"""The RAG engine: retrieve → assemble context → generate a cited answer.

This module is deliberately small and readable — it's the part you'll walk an
interviewer through. The flow is:

    question
      → embed & search ChromaDB (semantic search)         [retrieve]
      → keep chunks above a relevance floor                [filter]
      → format them as numbered, citable context           [assemble]
      → ask the LLM to answer using ONLY that context      [generate]
      → return answer + citations + a confidence score     [package]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import settings
from .prompts import QA_PROMPT
from .vectorstore import get_vectorstore

NO_ANSWER = "I could not find this in the provided documents."


@dataclass
class Source:
    """One retrieved chunk, ready to be shown as a citation."""

    number: int
    source: str
    page: int | None
    relevance: float  # cosine relevance in [0, 1]; higher is better
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} (p.{self.page})" if self.page is not None else self.source


@dataclass
class Answer:
    """The full result returned to the UI / caller."""

    text: str
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0  # 0–1
    grounded: bool = True

    @property
    def confidence_pct(self) -> int:
        return round(self.confidence * 100)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.75:
            return "High"
        if self.confidence >= 0.5:
            return "Medium"
        return "Low"


def _page_of(doc: Document) -> int | None:
    meta = doc.metadata
    if "page_label" in meta:
        return meta["page_label"]
    if "page" in meta:
        return meta["page"] + 1
    return None


def _retrieve(question: str, top_k: int) -> list[tuple[Document, float]]:
    """Semantic search returning (chunk, relevance) pairs, best first."""
    store = get_vectorstore()
    # relevance scores are normalised to [0, 1] (1 = identical) for cosine space.
    return store.similarity_search_with_relevance_scores(question, k=top_k)


def _format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        header = f"[{s.number}] Source: {s.label}"
        blocks.append(f"{header}\n{s.text}")
    return "\n\n".join(blocks)


def _confidence(sources: list[Source]) -> float:
    """Turn retrieval relevance into a single, explainable confidence score.

    Heuristic (transparent on purpose):
      * Base it on the BEST matching chunk (top relevance) — that's what most
        determines whether the answer exists in the corpus.
      * Add a small bonus when several chunks agree (corroboration), measured by
        the mean relevance of the rest.
    Both components come straight from cosine similarity, so the number is
    defensible rather than a black box.
    """
    if not sources:
        return 0.0
    rels = [s.relevance for s in sources]
    top = rels[0]
    corroboration = sum(rels[1:]) / len(rels[1:]) if len(rels) > 1 else top
    score = 0.8 * top + 0.2 * corroboration
    return max(0.0, min(1.0, score))


class RAGEngine:
    """Stateless query engine over the persisted vector store."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=0,  # factual extraction — no creativity
            api_key=settings.openai_api_key,
        )

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        top_k = top_k or settings.top_k
        hits = _retrieve(question, top_k)

        # Keep only sufficiently relevant chunks.
        kept = [(d, r) for d, r in hits if r >= settings.min_relevance]
        if not kept:
            return Answer(text=NO_ANSWER, sources=[], confidence=0.0, grounded=False)

        sources = [
            Source(
                number=i + 1,
                source=doc.metadata.get("source", "unknown"),
                page=_page_of(doc),
                relevance=round(rel, 4),
                text=doc.page_content.strip(),
            )
            for i, (doc, rel) in enumerate(kept)
        ]

        context = _format_context(sources)
        messages = QA_PROMPT.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        answer_text = response.content.strip()

        grounded = NO_ANSWER.lower() not in answer_text.lower()
        confidence = _confidence(sources) if grounded else 0.0

        return Answer(
            text=answer_text,
            sources=sources if grounded else [],
            confidence=confidence,
            grounded=grounded,
        )
