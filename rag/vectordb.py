"""
rag/vectordb.py — FAISS vector store with session persistence.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config import settings
from rag.chunker import TextChunk
from rag.embeddings import get_embedding_model


class FAISSVectorStore:
    """
    In-memory FAISS index backed by persistent storage per session.

    Workflow:
        store = FAISSVectorStore(session_id)
        store.add_chunks(chunks)            # embed + index
        results = store.search("query", k=5)
        store.save()                         # persist to disk
        store.load()                         # restore from disk
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._session_dir = settings.session_dir(session_id)
        self._index = None           # faiss.Index
        self._chunk_store: List[TextChunk] = []   # parallel list to index rows
        self._embedder = get_embedding_model()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _init_index(self) -> None:
        """Create a fresh FAISS flat L2 index."""
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")
        dim = self._embedder.dimension
        self._index = faiss.IndexFlatIP(dim)   # inner product (cosine on normalised vecs)
        logger.debug(f"FAISS IndexFlatIP created (dim={dim})")

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Embed chunks and add them to the FAISS index."""
        if not chunks:
            return

        if self._index is None:
            self._init_index()

        texts = [c.text for c in chunks]
        embeddings = self._embedder.encode(texts, batch_size=32, normalize=True)

        self._index.add(embeddings)
        self._chunk_store.extend(chunks)
        logger.info(f"FAISS: added {len(chunks)} chunks "
                    f"(total={self._index.ntotal})")

    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[TextChunk, float]]:
        """
        Semantic search over indexed chunks.

        Args:
            query: Natural language query.
            k: Number of top results to return.
            source_filter: If set, only return chunks from this source.

        Returns:
            List of (TextChunk, score) tuples ordered by relevance.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no results.")
            return []

        q_vec = self._embedder.encode_query(query)
        fetch_k = min(k * 3 if source_filter else k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, fetch_k)

        results: List[Tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunk_store):
                continue
            chunk = self._chunk_store[idx]
            if source_filter and chunk.source != source_filter:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break

        logger.debug(f"FAISS search '{query[:40]}' → {len(results)} results")
        return results

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal if self._index else 0

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist index + chunk store to the session directory."""
        if self._index is None:
            return
        import faiss
        index_path = self._session_dir / "faiss.index"
        meta_path = self._session_dir / "chunks.pkl"
        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self._chunk_store, f)
        logger.info(f"FAISS saved ({self._index.ntotal} vectors) → {self._session_dir}")

    def load(self) -> bool:
        """Restore index + chunk store from disk. Returns True if successful."""
        import faiss
        index_path = self._session_dir / "faiss.index"
        meta_path = self._session_dir / "chunks.pkl"
        if not index_path.exists() or not meta_path.exists():
            return False
        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self._chunk_store = pickle.load(f)
        logger.info(f"FAISS loaded ({self._index.ntotal} vectors) from {self._session_dir}")
        return True

    def clear(self) -> None:
        """Reset the index and chunk store in memory."""
        self._index = None
        self._chunk_store = []


_store_cache: Dict[str, FAISSVectorStore] = {}


def get_vector_store(session_id: str) -> FAISSVectorStore:
    """Return (or create) the FAISSVectorStore for a given session."""
    if session_id not in _store_cache:
        store = FAISSVectorStore(session_id)
        store.load()   # no-op if no saved index
        _store_cache[session_id] = store
    return _store_cache[session_id]
