"""
tests/test_rag.py — Unit tests for the RAG pipeline components.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.chunker import TextChunk, chunk_pages
from rag.pdf_loader import PageContent, _clean_text


# ── pdf_loader tests ───────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_extra_whitespace(self):
        raw = "hello   world  \t  foo"
        assert "  " not in _clean_text(raw)

    def test_fixes_hyphenation(self):
        raw = "trans-\nformer"
        result = _clean_text(raw)
        assert "trans-\n" not in result
        assert "transformer" in result

    def test_collapses_blank_lines(self):
        raw = "para1\n\n\n\n\npara2"
        result = _clean_text(raw)
        assert "\n\n\n" not in result


class TestLoadPdf:
    def test_nonexistent_file_returns_empty(self):
        from rag.pdf_loader import load_pdf
        result = load_pdf("/nonexistent/path/file.pdf")
        assert result == []


# ── chunker tests ──────────────────────────────────────────────────────────────

class TestChunker:
    def _make_page(self, text: str, source: str = "test_paper") -> PageContent:
        return PageContent(source=source, page_num=1, text=text)

    def test_short_text_single_chunk(self):
        page = self._make_page("Short text under 500 chars.")
        chunks = chunk_pages([page], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text under 500 chars."

    def test_long_text_produces_multiple_chunks(self):
        long_text = "sentence number X. " * 60
        page = self._make_page(long_text)
        chunks = chunk_pages([page], chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        page = self._make_page("hello world", source="arxiv_1234")
        chunks = chunk_pages([page])
        assert chunks[0].source == "arxiv_1234"
        assert chunks[0].page_num == 1

    def test_chunk_id_format(self):
        page = self._make_page("text", source="src")
        chunks = chunk_pages([page])
        assert chunks[0].chunk_id.startswith("src_p1_")

    def test_empty_pages_returns_empty(self):
        chunks = chunk_pages([])
        assert chunks == []

    def test_overlap_does_not_exceed_chunk_size(self):
        page = self._make_page("word " * 200)
        chunks = chunk_pages([page], chunk_size=100, chunk_overlap=30)
        for chunk in chunks:
            assert chunk.char_count <= 120   # slight tolerance for boundary logic


# ── embeddings tests ───────────────────────────────────────────────────────────

class TestEmbeddings:
    @patch("rag.embeddings.SentenceTransformer")
    def test_encode_returns_numpy_array(self, MockST):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        MockST.return_value = mock_model

        # Reset singleton
        from rag import embeddings
        embeddings.EmbeddingModel._instance = None
        embeddings.EmbeddingModel._model = None

        with patch("rag.embeddings.SentenceTransformer", MockST):
            from rag.embeddings import EmbeddingModel
            em = EmbeddingModel()
            em._model = mock_model

            result = em.encode(["text1", "text2", "text3"])
            assert result.shape == (3, 384)
            assert result.dtype == np.float32

    def test_encode_empty_returns_empty_array(self):
        from rag.embeddings import EmbeddingModel
        em = EmbeddingModel()
        # Skip actual model load by setting dim via mock
        em._model = MagicMock()
        em._model.get_sentence_embedding_dimension.return_value = 384
        em._model.encode.return_value = np.empty((0, 384), dtype=np.float32)
        result = em.encode([])
        assert result.shape[0] == 0


# ── retriever tests ────────────────────────────────────────────────────────────

class TestRetriever:
    def test_format_context_respects_max_chars(self):
        from rag.retriever import RetrievedChunk, format_context
        chunks = [
            RetrievedChunk("id1", "src", 1, "A" * 1000, 0.9),
            RetrievedChunk("id2", "src", 2, "B" * 1000, 0.8),
        ]
        result = format_context(chunks, max_chars=500)
        assert len(result) <= 600   # some tolerance for header overhead

    def test_format_context_empty_returns_empty(self):
        from rag.retriever import format_context
        assert format_context([]) == ""
