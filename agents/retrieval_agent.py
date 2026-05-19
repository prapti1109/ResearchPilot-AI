"""
agents/retrieval_agent.py — Agent 2: Parse PDFs, embed chunks, index in FAISS.
"""
from __future__ import annotations

from loguru import logger

from config import settings
from rag.chunker import chunk_pages
from rag.pdf_loader import load_pdf
from rag.retriever import RetrievedChunk, retrieve
from rag.vectordb import get_vector_store
from workflows.states import RetrievedChunkModel, TextChunkModel, WorkflowState


def run_retrieval_agent(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: Retrieval Agent.

    Processes each downloaded PDF through the full RAG pipeline:
    load → clean → chunk → embed → FAISS index.

    Then performs a semantic search for the research topic to populate
    state.retrieved_context.
    """
    state.current_agent = "retrieval_agent"
    state.log("📚 Retrieval Agent: Processing PDFs into vector database...")
    logger.info(f"[RetrievalAgent] session={state.session_id}")

    # Filter to successfully downloaded papers
    valid_papers = [p for p in state.papers if p.download_ok and p.local_pdf_path]
    if not valid_papers:
        state.add_error("No PDFs available to index.")
        return state

    store = get_vector_store(state.session_id)
    all_chunks: list[TextChunkModel] = []

    # ── Process each PDF ───────────────────────────────────────────────────────
    for paper in valid_papers:
        state.log(f"  📄 Parsing: {paper.title[:55]}...")
        try:
            pages = load_pdf(paper.local_pdf_path, source_id=paper.arxiv_id)
            if not pages:
                logger.warning(f"[RetrievalAgent] No text in {paper.arxiv_id}")
                continue

            chunks = chunk_pages(
                pages,
                chunk_size=state.chunk_size,
                chunk_overlap=state.chunk_overlap,
            )

            # Add to FAISS
            store.add_chunks(chunks)

            # Convert to state models
            for c in chunks:
                all_chunks.append(TextChunkModel(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    page_num=c.page_num,
                    text=c.text,
                    char_count=c.char_count,
                ))

            state.log(f"  ✅ Indexed {len(chunks)} chunks from {paper.arxiv_id}")
        except Exception as exc:
            logger.error(f"[RetrievalAgent] Error processing {paper.arxiv_id}: {exc}")
            state.log(f"  ⚠️  Failed to process {paper.arxiv_id}: {exc}")

    # Persist the FAISS index
    store.save()

    state.chunks = all_chunks
    state.metrics.total_chunks = len(all_chunks)
    state.log(f"✅ Vector DB ready: {len(all_chunks)} chunks from "
              f"{len(valid_papers)} papers.")

    # ── Semantic retrieval for the topic ──────────────────────────────────────
    state.log(f"🔎 Running semantic retrieval for: '{state.topic}'")
    try:
        raw_results = retrieve(
            query=state.topic,
            session_id=state.session_id,
            k=state.retrieval_top_k,
        )
        state.retrieved_context = [
            RetrievedChunkModel(
                chunk_id=r.chunk_id,
                source=r.source,
                page_num=r.page_num,
                text=r.text,
                score=r.score,
            )
            for r in raw_results
        ]
        state.log(f"✅ Retrieved {len(state.retrieved_context)} relevant chunks.")
    except Exception as exc:
        state.add_error(f"Retrieval failed: {exc}")

    return state
