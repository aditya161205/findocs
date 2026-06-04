# FinDocs RAG — Q&A over Private Financial Documents

Upload mutual-fund factsheets, annual reports, or RBI/SEBI circulars, ask
questions in plain English, and get **precise answers with source citations and
a confidence score** — grounded strictly in your own documents.

> Built to demonstrate a production-shaped **Retrieval-Augmented Generation
> (RAG)** pipeline: document ingestion → chunking → embeddings → vector search →
> grounded generation with citations.

---

## Features

- **Ask natural-language questions** over a private corpus of financial PDFs/text.
- **Source citations** — every answer points back to the file and page it came from.
- **Confidence score** derived transparently from retrieval relevance (no black box).
- **Refuses to hallucinate** — if the answer isn't in your documents, it says so.
- **Persistent vector index** (ChromaDB) so you don't re-embed on every run.
- **Two ways to use it:** a Streamlit web UI, or a CLI for batch indexing.

## Stack

| Layer            | Choice                                            |
| ---------------- | ------------------------------------------------- |
| Orchestration    | **LangChain**                                     |
| Vector DB        | **ChromaDB** (persistent, cosine similarity)      |
| Embeddings + LLM | **OpenAI** (`text-embedding-3-small`, `gpt-4o-mini`) |
| PDF parsing      | **pypdf**                                         |
| UI               | **Streamlit**                                     |

## How it works

```
                 ┌──────────────┐      ┌──────────────┐
   PDF/TXT/MD ──▶│  Ingestion   │ ───▶ │   Chunking   │   (1000 chars, 150 overlap)
                 │ (pypdf etc.) │      │ Recursive    │
                 └──────────────┘      └──────┬───────┘
                                              │ embed (OpenAI)
                                              ▼
                                       ┌──────────────┐
                                       │   ChromaDB   │  persistent vector store
                                       └──────┬───────┘
            question ──▶ embed ──▶ semantic search (top-k, cosine)
                                              │
                                              ▼
                              keep chunks ≥ relevance floor
                                              │
                                              ▼
                       numbered context ──▶ LLM (temp 0, "cite [n], no guessing")
                                              │
                                              ▼
                       Answer + citations [1][2] + confidence %
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design rationale and
the talking points you can use to explain each decision.

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure your OpenAI key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Index the bundled sample docs (or your own files)
python ingest.py data/sample_docs

# 4. Launch the UI
streamlit run app.py
```

Then ask things like:

- *"What is the expense ratio of the direct plan and the 1-year return?"*
- *"How often must high-risk customers' KYC be updated?"*
- *"Who is the fund manager and since when?"*

You can also upload your own documents directly from the sidebar.

## Tests

The chunking and citation/confidence logic are unit-tested and need **no API key
or network**:

```bash
pytest -q
```

## Project layout

```
.
├── app.py               # Streamlit UI
├── ingest.py            # CLI: index documents into ChromaDB
├── src/
│   ├── config.py        # all tunables (chunk size, top-k, models, thresholds)
│   ├── ingestion.py     # load + chunk documents (citable metadata)
│   ├── vectorstore.py   # ChromaDB wrapper (embed / add / search / reset)
│   ├── prompts.py       # grounded, citation-enforcing prompt
│   └── rag_engine.py    # retrieve → assemble → generate → score
├── data/sample_docs/    # synthetic factsheet + RBI-style circular (safe to demo)
├── tests/
└── docs/ARCHITECTURE.md
```

## Notes

- The bundled documents in `data/sample_docs/` are **synthetic** and for demo only.
  Swap in real SEBI/RBI circulars or any company's 10-K to make it shine.
- Your `.env` and the `.chroma/` index are git-ignored — no secrets or data leak
  into the repo.

## License

MIT — see [`LICENSE`](LICENSE).
