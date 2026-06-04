# FinDocs RAG

A small Q&A app I built for asking questions over financial documents. You give
it some PDFs (mutual fund factsheets, annual reports, RBI/SEBI circulars, a
company's 10-K, whatever), and then you can ask questions in plain English and
get answers back with the source it pulled them from and how confident it is.

I made it mostly to get hands-on with RAG end to end: parsing docs, chunking
them, embedding into a vector DB, and wiring up retrieval + an LLM so the answers
actually stay grounded in the documents instead of being made up.

## What it does

- Answers questions using only the documents you give it.
- Shows where each answer came from (file name + page number).
- Gives a confidence score so you can tell when it's unsure.
- Says "I couldn't find this in the documents" instead of guessing when the
  answer isn't there.
- Keeps the index on disk, so you don't have to re-embed everything every time
  you restart.
- Works either through a small Streamlit UI or from the command line.

## Stack

- Python
- LangChain for the RAG plumbing
- ChromaDB as the vector store
- OpenAI for embeddings and the chat model
- pypdf for reading PDFs
- Streamlit for the UI

## How it works

The flow is pretty straightforward:

1. Load the files and split them into ~1000-character chunks with a bit of
   overlap, keeping track of which file and page each chunk came from.
2. Embed the chunks and store them in ChromaDB.
3. When you ask a question, embed it and pull back the most similar chunks.
4. Drop anything that isn't similar enough. If nothing's left, just say it
   wasn't found instead of calling the model.
5. Hand the remaining chunks to the model with instructions to answer only from
   them and to cite each one.
6. Return the answer with its citations and a confidence number.

The confidence score is intentionally simple so I can explain it: it's mostly
based on how well the best chunk matched the question, plus a little bump if the
other chunks agree. It's a measure of how good the retrieval was, not a
guarantee the answer is correct.

There's more detail and the reasoning behind each choice in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# put your OpenAI key in .env

# index the sample docs (or point it at your own files)
python ingest.py data/sample_docs

streamlit run app.py
```

Some questions that work on the sample docs:

- What's the expense ratio of the direct plan and the 1-year return?
- How often do high-risk customers need their KYC updated?
- Who's the fund manager and since when?

You can also just upload files from the sidebar instead of using the CLI.

## Tests

The chunking and the confidence/citation logic have unit tests that don't need
an API key or network:

```bash
pytest -q
```

## Layout

```
app.py            Streamlit UI
ingest.py         CLI for indexing documents
src/
  config.py       settings (chunk size, top-k, models, thresholds)
  ingestion.py    loading + chunking
  vectorstore.py  ChromaDB wrapper
  prompts.py      the prompt
  rag_engine.py   retrieval + answer generation
data/sample_docs/ a couple of sample docs to try it on
tests/
docs/
```

## A couple of notes

The sample documents under `data/sample_docs/` are fake ones I wrote up so the
app has something to run on out of the box. Swap in real documents to actually
use it.

The `.env` file and the local Chroma index are gitignored, so no keys or indexed
data end up in the repo.
