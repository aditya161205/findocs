# Architecture & Design Rationale

This document explains *why* each piece is built the way it is. It doubles as an
**interview cheat-sheet** — every design choice below is something you can defend.

## The RAG pipeline, end to end

RAG = **Retrieval-Augmented Generation**. Instead of asking an LLM to answer from
its (frozen, generic) memory, we *retrieve* the most relevant passages from the
user's own documents and *augment* the prompt with them. The LLM then answers
from that grounded context only.

This solves three problems with using a raw LLM on financial docs:

1. **No private knowledge** — the model has never seen this specific factsheet.
2. **Hallucination** — LLMs invent plausible-but-wrong numbers; grounding + "say
   you don't know" instructions curb that.
3. **Traceability** — regulators/analysts need to know *where* an answer came
   from. We return citations.

### 1. Ingestion (`src/ingestion.py`)

- **Loaders:** `pypdf` for PDFs (one `Document` per page), text loaders for
  `.txt`/`.md`. Per-page loading is what lets us cite a **page number** later.
- **Metadata normalisation:** every chunk carries `source` (filename) and a
  1-based `page_label`. Citations are only possible because this metadata rides
  along with each chunk.

### 2. Chunking (`src/ingestion.py`)

- **Why chunk at all?** Embeddings represent a bounded amount of text well; a
  whole 100-page report embedded as one vector is useless for pinpoint retrieval.
  Chunks are the unit of retrieval.
- **`RecursiveCharacterTextSplitter`** splits on natural boundaries first
  (paragraph → line → sentence → word), so we rarely cut mid-sentence.
- **Size = 1000 chars (~250 tokens):** big enough to keep a factsheet table or a
  circular clause intact, small enough that retrieval stays precise.
- **Overlap = 150 chars:** a fact that straddles a boundary (e.g. a sentence
  split across two chunks) still appears whole in at least one chunk.

> **Trade-off to mention:** larger chunks = more context per hit but noisier and
> more tokens; smaller chunks = sharper retrieval but risk losing context.

### 3. Embeddings + Vector store (`src/vectorstore.py`)

- **Embeddings** (`text-embedding-3-small`) map text to vectors where semantic
  similarity ≈ geometric closeness. This is what makes *semantic* search work:
  "cost of the fund" can match a chunk about "expense ratio" even with no shared
  words.
- **ChromaDB**, persisted to disk, configured with **cosine** distance so
  relevance scores are comparable across queries and normalisable to `[0, 1]`.
- Index is persisted (`.chroma/`) so we embed once, not on every run.

> **Why a vector DB and not just a Python list + cosine?** Approximate
> nearest-neighbour (HNSW) search scales to millions of chunks with sub-linear
> lookups, plus metadata filtering and persistence for free.

### 4. Retrieval + filtering (`src/rag_engine.py`)

- **Top-k semantic search** returns the k closest chunks with relevance scores.
- A **relevance floor** (`MIN_RELEVANCE`) drops weak matches. If nothing clears
  the floor, we short-circuit to *"I could not find this in the provided
  documents"* — no LLM call, no hallucination risk.

### 5. Generation (`src/prompts.py`, `src/rag_engine.py`)

- The retrieved chunks are formatted as **numbered sources** and injected into
  the prompt.
- The system prompt enforces three things: answer **only** from context, **cite
  `[n]`** for each claim, and **refuse** when the answer isn't present.
- **`temperature = 0`** — this is factual extraction, not creative writing.

### 6. Confidence score (`_confidence` in `src/rag_engine.py`)

A deliberately **transparent** heuristic rather than a black box:

```
confidence = 0.8 * (top chunk relevance) + 0.2 * (mean relevance of the rest)
```

- The **top match** dominates — it most determines whether the answer exists.
- The **corroboration term** nudges confidence up when several chunks agree.
- Both inputs are raw cosine relevances, so the number is explainable and
  bounded to `[0, 1]`, then bucketed into High / Medium / Low.

> **Honest framing for an interview:** this is a *retrieval-confidence* proxy,
> not a calibrated probability the answer is correct. A natural "what would you
> improve" answer: add an LLM-based groundedness/faithfulness check, or compare
> the answer against the cited spans (e.g. an NLI model).

## Where to take it next

- **Hybrid search** (BM25 keyword + dense vectors) for exact tickers/figures.
- **Re-ranking** the top-k with a cross-encoder before generation.
- **Citation verification** — check each `[n]` claim is entailed by its source.
- **Eval harness** — a small Q&A set scored for answer + citation accuracy.
- **Multi-tenant isolation** — per-user collections for true "private" docs.
