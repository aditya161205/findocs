"""Streamlit front-end for FinDocs RAG.

Upload financial documents → ask questions → get cited, confidence-scored answers.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.config import settings
from src.ingestion import build_chunks
from src.rag_engine import RAGEngine
from src.vectorstore import add_chunks, collection_size, reset_collection

st.set_page_config(page_title="FinDocs RAG", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner=False)
def get_engine() -> RAGEngine:
    return RAGEngine()


def _ingest_uploaded_files(uploaded_files) -> int:
    """Persist uploads to a temp dir, chunk them, and add to the vector store."""
    added = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        paths = []
        for uf in uploaded_files:
            dest = tmp_dir / uf.name
            dest.write_bytes(uf.getbuffer())
            paths.append(dest)
        chunks = build_chunks(paths)
        added = add_chunks(chunks)
    return added


# ----------------------------- Sidebar ------------------------------------- #
with st.sidebar:
    st.title("📊 FinDocs RAG")
    st.caption("Ask questions across your private financial documents.")

    if not settings.has_openai_key:
        st.error("`OPENAI_API_KEY` not set. Add it to a `.env` file and restart.")

    st.subheader("1 · Add documents")
    uploads = st.file_uploader(
        "Factsheets, annual reports, RBI/SEBI circulars, 10-Ks",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", type="primary", disabled=not uploads):
        with st.spinner("Chunking & embedding…"):
            n = _ingest_uploaded_files(uploads)
        st.success(f"Indexed {n} chunks from {len(uploads)} file(s).")

    st.divider()
    st.metric("Chunks in index", collection_size())
    if st.button("Clear index"):
        reset_collection()
        st.cache_resource.clear()
        st.success("Index cleared.")
        st.rerun()

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, settings.top_k)
    st.caption(f"Model: `{settings.chat_model}` · Embeddings: `{settings.embedding_model}`")


# ------------------------------ Main --------------------------------------- #
st.header("Ask a question")
st.caption(
    "Answers are generated **only** from the indexed documents, with source "
    "citations and a confidence score."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. What is the expense ratio and what was the 1-year return?",
)

ask = st.button("Ask", type="primary", disabled=not question)

if ask:
    if collection_size() == 0:
        st.warning("No documents indexed yet — upload and index some on the left first.")
    elif not settings.has_openai_key:
        st.error("Cannot answer without `OPENAI_API_KEY`.")
    else:
        with st.spinner("Searching documents & composing answer…"):
            answer = get_engine().ask(question, top_k=top_k)

        # --- Answer + confidence ---
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown("### Answer")
            st.markdown(answer.text)
        with col_b:
            st.markdown("### Confidence")
            st.metric(answer.confidence_label, f"{answer.confidence_pct}%")
            st.progress(answer.confidence)

        # --- Citations ---
        if answer.sources:
            st.markdown("### Sources")
            for s in answer.sources:
                with st.expander(f"[{s.number}] {s.label} · relevance {s.relevance:.2f}"):
                    st.write(s.text)
        elif not answer.grounded:
            st.info(
                "The answer wasn't found in the indexed documents. Try rephrasing, "
                "or index a document that contains the information."
            )
