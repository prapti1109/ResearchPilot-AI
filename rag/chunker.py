"""
rag/chunker.py — Text chunker for the RAG pipeline.
Splits PageContent objects into overlapping chunks for embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from loguru import logger

from rag.pdf_loader import PageContent


@dataclass
class TextChunk:
    chunk_id: str       # "{source}_p{page}_{index}"
    source: str
    page_num: int
    text: str
    char_count: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page_num": self.page_num,
            "text": self.text,
            "char_count": self.char_count,
        }


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Simple character-level sliding window splitter.
    Tries to split at sentence boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:]
        else:
            # Try to find a sentence boundary (. or \n) within last 20% of chunk
            boundary_search_start = start + int(chunk_size * 0.8)
            best_boundary = -1
            for delim in [".\n", ".\n\n", ". ", "\n\n", "\n"]:
                idx = text.rfind(delim, boundary_search_start, end)
                if idx > best_boundary:
                    best_boundary = idx + len(delim)

            end = best_boundary if best_boundary > start else end
            chunk = text[start:end]

        chunk = chunk.strip()
        if len(chunk) > 20:  # skip tiny fragments
            chunks.append(chunk)

        start = end - chunk_overlap
        if start <= 0 or start >= len(text):
            break

    return chunks


def chunk_pages(
    pages: List[PageContent],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[TextChunk]:
    """
    Split a list of PageContent objects into TextChunk objects.

    Args:
        pages: Output of rag.pdf_loader.load_pdf()
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of TextChunk objects with metadata.
    """
    chunks: List[TextChunk] = []
    for page in pages:
        splits = _split_text(page.text, chunk_size, chunk_overlap)
        for i, text in enumerate(splits):
            chunk = TextChunk(
                chunk_id=f"{page.source}_p{page.page_num}_{i}",
                source=page.source,
                page_num=page.page_num,
                text=text,
                char_count=len(text),
            )
            chunks.append(chunk)

    logger.info(f"Chunker produced {len(chunks)} chunks from {len(pages)} pages "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks
