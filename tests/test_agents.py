"""
tests/test_agents.py — Unit tests for all 6 LangGraph agents (LLM mocked).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from workflows.states import (
    ClaimVerdict,
    PaperMetadataModel,
    PaperSummary,
    ReportSections,
    WorkflowState,
    WorkflowStatus,
)


def make_state(**kwargs) -> WorkflowState:
    defaults = dict(topic="Test Topic", max_papers=2)
    defaults.update(kwargs)
    return WorkflowState(**defaults)


def make_paper(arxiv_id="1234.5678", download_ok=True) -> PaperMetadataModel:
    return PaperMetadataModel(
        arxiv_id=arxiv_id,
        title="Test Paper on AI",
        authors=["Alice Smith", "Bob Jones"],
        abstract="This paper proposes a novel approach to AI.",
        published="2024-01-15",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        local_pdf_path="/tmp/test.pdf",
        download_ok=download_ok,
    )


# ── Research Agent ─────────────────────────────────────────────────────────────

class TestResearchAgent:
    @patch("agents.research_agent.search_papers")
    @patch("agents.research_agent.download_pdfs")
    def test_populates_papers(self, mock_dl, mock_search):
        from tools.arxiv_tool import PaperMetadata
        mock_paper = PaperMetadata(
            arxiv_id="1234.5678",
            title="Test Paper",
            authors=["A. Author"],
            abstract="Abstract text",
            published="2024-01-01",
            pdf_url="http://example.com/test.pdf",
            entry_url="http://arxiv.org/abs/1234.5678",
        )
        mock_paper.local_pdf_path = "/tmp/test.pdf"
        mock_search.return_value = [mock_paper]
        mock_dl.return_value = [mock_paper]

        from agents.research_agent import run_research_agent
        state = make_state()
        result = run_research_agent(state)

        assert len(result.papers) == 1
        assert result.papers[0].arxiv_id == "1234.5678"

    @patch("agents.research_agent.search_papers", return_value=[])
    @patch("agents.research_agent.download_pdfs", return_value=[])
    def test_no_results_adds_error(self, mock_dl, mock_search):
        from agents.research_agent import run_research_agent
        state = make_state()
        result = run_research_agent(state)
        assert len(result.errors) > 0


# ── Verification Agent ─────────────────────────────────────────────────────────

class TestVerificationAgent:
    @patch("agents.verification_agent.get_llm")
    @patch("agents.verification_agent.retrieve", return_value=[])
    def test_verifies_claims(self, mock_retrieve, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "VERDICT: VERIFIED\nCONFIDENCE: 0.85\n"
            "EXPLANATION: Claim is supported by retrieved evidence."
        )
        mock_get_llm.return_value = mock_llm

        from agents.verification_agent import run_verification_agent
        state = make_state()
        state.summaries = [
            PaperSummary(
                arxiv_id="1234",
                title="Test",
                key_contributions="CNN achieves 95% accuracy on benchmark.",
                results="Improved accuracy by 10%.",
            )
        ]
        result = run_verification_agent(state)
        assert len(result.verified_findings) > 0
        assert result.verified_findings[0].verdict == ClaimVerdict.VERIFIED

    def test_empty_summaries_skips_gracefully(self):
        from agents.verification_agent import run_verification_agent
        state = make_state()
        result = run_verification_agent(state)
        assert result.verified_findings == []


# ── Summarizer Agent ───────────────────────────────────────────────────────────

class TestSummarizerAgent:
    @patch("agents.summarizer_agent.get_llm")
    @patch("agents.summarizer_agent.retrieve", return_value=[])
    def test_generates_summaries(self, mock_retrieve, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "KEY CONTRIBUTIONS: Novel approach to X.\n"
            "METHODOLOGY: Deep learning with transformers.\n"
            "RESULTS: 95% accuracy achieved.\n"
            "LIMITATIONS: Only tested on small datasets."
        )
        mock_get_llm.return_value = mock_llm

        from agents.summarizer_agent import run_summarizer_agent
        state = make_state()
        state.papers = [make_paper()]
        result = run_summarizer_agent(state)

        assert len(result.summaries) == 1
        s = result.summaries[0]
        assert s.arxiv_id == "1234.5678"

    def test_no_papers_returns_empty_summaries(self):
        from agents.summarizer_agent import run_summarizer_agent
        state = make_state()
        result = run_summarizer_agent(state)
        assert result.summaries == []


# ── Writer Agent ───────────────────────────────────────────────────────────────

class TestWriterAgent:
    @patch("agents.writer_agent.get_llm")
    @patch("agents.writer_agent.retrieve", return_value=[])
    def test_fills_all_sections(self, mock_retrieve, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated section content for testing purposes."
        mock_get_llm.return_value = mock_llm

        from agents.writer_agent import run_writer_agent
        state = make_state()
        state.summaries = [
            PaperSummary(
                arxiv_id="1234",
                title="Test Paper",
                key_contributions="Main contribution here.",
                methodology="DNN approach.",
                results="Good results.",
                limitations="Small dataset.",
            )
        ]
        result = run_writer_agent(state)

        assert result.report_sections.introduction != ""
        assert result.report_sections.related_work != ""
        assert result.report_sections.key_findings != ""


# ── Citation Agent ─────────────────────────────────────────────────────────────

class TestCitationAgent:
    @patch("agents.citation_agent.export_report")
    def test_generates_ieee_citations(self, mock_export):
        mock_export.return_value = MagicMock(name="report_1234.pdf")

        from agents.citation_agent import run_citation_agent, _ieee_citation
        paper = make_paper()
        ref = _ieee_citation(paper, 1)

        assert "[1]" in ref
        assert "Test Paper on AI" in ref
        assert "1234.5678" in ref
        assert "2024" in ref

    def test_apa_citation_format(self):
        from agents.citation_agent import _apa_citation
        paper = make_paper()
        ref = _apa_citation(paper)
        assert "2024" in ref
        assert "Test Paper on AI" in ref

    @patch("agents.citation_agent.export_report")
    def test_populates_state_citations(self, mock_export):
        mock_export.return_value = MagicMock()
        mock_export.return_value.__str__ = lambda _: "report.pdf"

        from agents.citation_agent import run_citation_agent
        state = make_state()
        state.papers = [make_paper("1111.2222"), make_paper("3333.4444")]
        state.report_sections = ReportSections(
            introduction="Intro",
            related_work="Related",
            key_findings="Findings",
        )
        result = run_citation_agent(state)

        assert len(result.citations_ieee) == 2
        assert len(result.citations_apa) == 2
        assert "[1]" in result.citations_ieee[0]
        assert "[2]" in result.citations_ieee[1]
