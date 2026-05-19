"""
workflows/states.py — Pydantic data models for the LangGraph workflow state.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETE = "complete"
    ERROR = "error"


class ClaimVerdict(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


# ── Sub-models ─────────────────────────────────────────────────────────────────

class PaperMetadataModel(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    published: str = ""
    pdf_url: str = ""
    entry_url: str = ""
    categories: List[str] = Field(default_factory=list)
    local_pdf_path: Optional[str] = None
    download_ok: bool = False


class TextChunkModel(BaseModel):
    chunk_id: str
    source: str
    page_num: int
    text: str
    char_count: int


class RetrievedChunkModel(BaseModel):
    chunk_id: str
    source: str
    page_num: int
    text: str
    score: float


class PaperSummary(BaseModel):
    arxiv_id: str
    title: str
    key_contributions: str = ""
    methodology: str = ""
    results: str = ""
    limitations: str = ""
    relevance_score: float = 0.0


class VerifiedClaim(BaseModel):
    claim: str
    verdict: ClaimVerdict = ClaimVerdict.UNVERIFIED
    evidence: str = ""
    confidence: float = 0.0


class ReportSections(BaseModel):
    introduction: str = ""
    related_work: str = ""
    method_comparison: str = ""
    key_findings: str = ""
    challenges: str = ""
    future_work: str = ""


class SessionMetrics(BaseModel):
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    total_papers: int = 0
    total_chunks: int = 0
    total_llm_calls: int = 0
    hallucination_rate: float = 0.0
    retrieval_latency_s: float = 0.0
    total_latency_s: float = 0.0

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)


# ── Master workflow state ──────────────────────────────────────────────────────

class WorkflowState(BaseModel):
    """
    Central state object passed between LangGraph nodes.
    Each agent reads from and writes into this shared state.
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    max_papers: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 5
    review_mode: bool = False           # HITL toggle
    output_format: str = "pdf"          # pdf | md | docx

    # ── Agent outputs ──────────────────────────────────────────────────────────
    papers: List[PaperMetadataModel] = Field(default_factory=list)
    chunks: List[TextChunkModel] = Field(default_factory=list)
    retrieved_context: List[RetrievedChunkModel] = Field(default_factory=list)
    summaries: List[PaperSummary] = Field(default_factory=list)
    cross_paper_synthesis: str = ""
    verified_findings: List[VerifiedClaim] = Field(default_factory=list)
    report_sections: ReportSections = Field(default_factory=ReportSections)
    citations_ieee: List[str] = Field(default_factory=list)
    citations_apa: List[str] = Field(default_factory=list)
    final_report_path: Optional[str] = None

    # ── Control ────────────────────────────────────────────────────────────────
    status: WorkflowStatus = WorkflowStatus.IDLE
    current_agent: str = ""
    errors: List[str] = Field(default_factory=list)
    progress_log: List[str] = Field(default_factory=list)
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)
    approved_papers: Optional[List[str]] = None   # set by HITL, None = all approved

    def log(self, message: str) -> None:
        """Append a timestamped message to progress_log."""
        ts = time.strftime("%H:%M:%S")
        self.progress_log.append(f"[{ts}] {message}")

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.log(f"ERROR: {message}")

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact dict for API status responses."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "status": self.status,
            "current_agent": self.current_agent,
            "papers_found": len(self.papers),
            "chunks_indexed": len(self.chunks),
            "summaries_done": len(self.summaries),
            "verified_claims": len(self.verified_findings),
            "report_ready": self.final_report_path is not None,
            "errors": self.errors,
            "elapsed_s": self.metrics.elapsed,
            "progress_log": self.progress_log[-20:],  # last 20 lines
        }
