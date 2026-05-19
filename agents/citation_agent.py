"""
agents/citation_agent.py — Agent 6: Generate IEEE and APA citations.
"""
from __future__ import annotations

import time
from loguru import logger
from workflows.states import WorkflowState


def _ieee_citation(paper, index: int) -> str:
    """
    Format an IEEE-style citation.
    [N] A. Author1, B. Author2, "Title," arXiv:XXXX.XXXXX, Year.
    """
    authors = paper.authors[:6]
    if not authors:
        author_str = "Unknown Author"
    else:
        formatted = []
        for name in authors:
            parts = name.strip().split()
            if len(parts) >= 2:
                # "FirstName LastName" → "F. LastName"
                initials = " ".join(f"{p[0]}." for p in parts[:-1])
                formatted.append(f"{initials} {parts[-1]}")
            else:
                formatted.append(name)
        if len(paper.authors) > 6:
            formatted.append("et al.")
        author_str = ", ".join(formatted)

    year = paper.published[:4] if paper.published else "n.d."
    title = paper.title.strip()
    arxiv_id = paper.arxiv_id

    return (
        f"[{index}] {author_str}, \"{title},\" "
        f"arXiv:{arxiv_id}, {year}."
    )


def _apa_citation(paper) -> str:
    """
    Format an APA 7th edition citation.
    LastName, F., & LastName2, F. (Year). Title. arXiv. https://arxiv.org/abs/XXXX
    """
    authors = paper.authors[:20]
    if not authors:
        author_str = "Unknown Author"
    else:
        formatted = []
        for name in authors[:6]:
            parts = name.strip().split()
            if len(parts) >= 2:
                last = parts[-1]
                initials = ", ".join(f"{p[0]}." for p in parts[:-1])
                formatted.append(f"{last}, {initials}")
            else:
                formatted.append(name)
        if len(paper.authors) > 6:
            formatted.append("et al.")
        if len(formatted) == 1:
            author_str = formatted[0]
        elif len(formatted) == 2:
            author_str = " & ".join(formatted)
        else:
            author_str = ", ".join(formatted[:-1]) + ", & " + formatted[-1]

    year = paper.published[:4] if paper.published else "n.d."
    title = paper.title.strip()
    arxiv_id = paper.arxiv_id
    url = paper.entry_url or f"https://arxiv.org/abs/{arxiv_id}"

    return (
        f"{author_str} ({year}). {title}. "
        f"arXiv. {url}"
    )


def run_citation_agent(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: Citation Agent.

    Generates IEEE and APA references for all papers in state.papers.
    Also injects citation numbers into the related_work section.
    """
    state.current_agent = "citation_agent"
    state.log("📚 Citation Agent: Generating references...")
    logger.info(f"[CitationAgent] {len(state.papers)} papers to cite")

    ieee_refs: list[str] = []
    apa_refs: list[str] = []

    for i, paper in enumerate(state.papers, start=1):
        try:
            ieee_refs.append(_ieee_citation(paper, i))
            apa_refs.append(_apa_citation(paper))
        except Exception as exc:
            logger.error(f"[CitationAgent] Error citing {paper.arxiv_id}: {exc}")
            ieee_refs.append(f"[{i}] Citation generation failed for {paper.arxiv_id}.")
            apa_refs.append(f"Citation error: {paper.arxiv_id}")

    state.citations_ieee = ieee_refs
    state.citations_apa = apa_refs

    state.log(f"✅ Citation Agent: Generated {len(ieee_refs)} IEEE + {len(apa_refs)} APA citations.")

    # ── Export report ─────────────────────────────────────────────────────────
    import time as _time
    from pathlib import Path
    from config import settings
    from tools.report_generator import export_report

    report_data = {
        "topic": state.topic,
        "generated_at": _time.strftime("%Y-%m-%d %H:%M"),
        "session_id": state.session_id,
        "sections": {
            "introduction": state.report_sections.introduction,
            "related_work": state.report_sections.related_work,
            "method_comparison": state.report_sections.method_comparison,
            "key_findings": state.report_sections.key_findings,
            "challenges": state.report_sections.challenges,
            "future_work": state.report_sections.future_work,
        },
        "papers": [p.model_dump() for p in state.papers],
        "citations": state.citations_ieee,
        "metrics": {
            "total_papers": state.metrics.total_papers,
            "total_chunks": state.metrics.total_chunks,
            "llm_calls": state.metrics.total_llm_calls,
            "hallucination_rate": state.metrics.hallucination_rate,
            "elapsed_s": state.metrics.elapsed,
        },
    }

    try:
        out_path = export_report(
            report_data,
            output_dir=settings.reports_dir,
            session_id=state.session_id,
            fmt=state.output_format,
        )
        state.final_report_path = str(out_path)
        state.log(f"✅ Report exported: {out_path.name}")
    except Exception as exc:
        logger.error(f"[CitationAgent] Export error: {exc}")
        state.add_error(f"Report export failed: {exc}")

    state.metrics.end_time = _time.time()
    state.log(f"🎉 Workflow complete in {state.metrics.elapsed:.1f}s")
    return state
