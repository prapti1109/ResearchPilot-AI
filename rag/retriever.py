"""
rag/retriever.py — Semantic retriever on top of FAISSVectorStore.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from rag.vectordb import get_vector_store


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    page_num: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page_num": self.page_num,
            "text": self.text,
            "score": round(self.score, 4),
        }


def retrieve(
    query: str,
    session_id: str,
    k: int = 5,
    source_filter: Optional[str] = None,
    min_score: float = 0.0,
) -> List[RetrievedChunk]:
    """
    Retrieve top-k semantically relevant chunks for a query.

    Args:
        query: User query or sub-question.
        session_id: Active research session.
        k: Max chunks to return.
        source_filter: Restrict to a specific paper source ID.
        min_score: Minimum cosine similarity score threshold.

    Returns:
        List of RetrievedChunk ordered by relevance (highest first).
    """
    store = get_vector_store(session_id)
    raw = store.search(query, k=k, source_filter=source_filter)

    results: List[RetrievedChunk] = []
    for chunk, score in raw:
        if score < min_score:
            continue
        results.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page_num=chunk.page_num,
            text=chunk.text,
            score=score,
        ))

    logger.debug(f"Retriever: '{query[:50]}' → {len(results)} chunks "
                 f"(session={session_id})")
    return results


def format_context(chunks: List[RetrievedChunk], max_chars: int = 4000) -> str:
    """
    Concatenate retrieved chunks into a prompt-ready context string.
    Truncates to max_chars to avoid overflowing LLM context window.
    """
    parts: List[str] = []
    total = 0
    for chunk in chunks:
        header = f"[Source: {chunk.source}, Page {chunk.page_num}]"
        block = f"{header}\n{chunk.text}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "...")
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)
