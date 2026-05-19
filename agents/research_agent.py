"""
agents/research_agent.py — Agent 1: Search arXiv and download PDFs.
"""
from __future__ import annotations

from loguru import logger

from config import settings
from tools.arxiv_tool import download_pdfs, search_papers
from workflows.states import PaperMetadataModel, WorkflowState


def run_research_agent(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: Research Agent.

    Searches arXiv for the topic, downloads PDFs, updates state.papers.
    """
    state.current_agent = "research_agent"
    state.log(f"🔍 Research Agent: Searching '{state.topic}' (max={state.max_papers})")
    logger.info(f"[ResearchAgent] topic='{state.topic}' max={state.max_papers}")

    try:
        # 1. Search arXiv
        papers_raw = search_papers(state.topic, max_results=state.max_papers)
        if not papers_raw:
            state.add_error("arXiv search returned no results.")
            return state

        state.log(f"✅ Found {len(papers_raw)} papers on arXiv.")

        # 2. Convert to Pydantic models
        paper_models = [PaperMetadataModel(**p.to_dict()) for p in papers_raw]
        state.papers = paper_models

        # 3. Download PDFs
        dest_dir = settings.pdfs_dir(state.session_id)
        state.log(f"📥 Downloading {len(papers_raw)} PDFs...")

        updated = download_pdfs(papers_raw, dest_dir=dest_dir, concurrency=3)

        # Merge download results back into paper models
        dl_map = {p.arxiv_id: p.local_pdf_path for p in updated}
        ok_count = 0
        for pm in state.papers:
            local_path = dl_map.get(pm.arxiv_id)
            if local_path:
                pm.local_pdf_path = local_path
                pm.download_ok = True
                ok_count += 1

        state.metrics.total_papers = ok_count
        state.log(f"✅ Downloaded {ok_count}/{len(papers_raw)} PDFs successfully.")
        logger.info(f"[ResearchAgent] {ok_count} PDFs ready.")

    except Exception as exc:
        state.add_error(f"Research Agent failed: {exc}")
        logger.exception("[ResearchAgent] Unexpected error")

    return state
