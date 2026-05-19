"""
rag/embeddings.py — Sentence embedding pipeline using BAAI/bge-small-en.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from config import settings


class EmbeddingModel:
    """
    Singleton wrapper around sentence-transformers for BAAI/bge-small-en.

    The model is loaded lazily on first call to encode().
    """
    _instance: Optional["EmbeddingModel"] = None
    _model = None

    def __new__(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
            dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model ready — dim={dim}")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into embedding vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per inference batch.
            normalize: L2-normalize embeddings (recommended for cosine similarity).
            show_progress: Show tqdm progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._load()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.debug(f"Encoding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns shape (1, dim)."""
        # BGE models benefit from a prefix for queries
        prefixed = f"Represent this sentence for retrieval: {query}"
        return self.encode([prefixed])


def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
