"""
agents/summarizer_agent.py — Agent 4: Summarize each paper + cross-paper synthesis.
"""
from __future__ import annotations

from loguru import logger

from rag.retriever import format_context, retrieve
from tools.llm import get_llm
from workflows.states import PaperSummary, WorkflowState

_SYSTEM = (
    "You are an expert AI research analyst. Produce structured, technically accurate summaries "
    "of academic papers. Be concise, factual, and use academic language."
)

_PAPER_SUMMARY_PROMPT = """
Summarize the following research paper based on its abstract and extracted content.

PAPER TITLE: {title}
AUTHORS: {authors}
PUBLISHED: {published}

ABSTRACT:
{abstract}

EXTRACTED CONTENT (from PDF):
{context}

Provide a structured summary with these exact sections:
KEY CONTRIBUTIONS: (2-3 bullet points of main contributions)
METHODOLOGY: (brief description of methods/algorithms used)
RESULTS: (key quantitative or qualitative results)
LIMITATIONS: (1-2 known limitations or gaps)

Keep each section under 150 words.
"""

_SYNTHESIS_PROMPT = """
You are synthesizing findings across multiple research papers on the topic: "{topic}"

INDIVIDUAL PAPER SUMMARIES:
{summaries}

Write a cross-paper synthesis (300-400 words) covering:
1. Common themes and shared approaches
2. Key disagreements or conflicting findings
3. Overall state of the field
4. Research gaps that remain open

Be analytical and cite paper titles when relevant.
"""


def _parse_paper_summary(response: str, arxiv_id: str, title: str) -> PaperSummary:
    """Parse the structured LLM output into a PaperSummary model."""
    sections = {
        "key_contributions": "",
        "methodology": "",
        "results": "",
        "limitations": "",
    }
    current_key = None
    lines = response.splitlines()
    for line in lines:
        lower = line.lower()
        if "key contribution" in lower:
            current_key = "key_contributions"
        elif "methodolog" in lower:
            current_key = "methodology"
        elif "result" in lower:
            current_key = "results"
        elif "limitation" in lower:
            current_key = "limitations"
        elif current_key and line.strip():
            sections[current_key] += line.strip() + " "

    return PaperSummary(
        arxiv_id=arxiv_id,
        title=title,
        key_contributions=sections["key_contributions"].strip(),
        methodology=sections["methodology"].strip(),
        results=sections["results"].strip(),
        limitations=sections["limitations"].strip(),
    )


def run_summarizer_agent(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: Summarization Agent.

    For each paper:
    1. Retrieves relevant chunks from FAISS (per-paper source filter)
    2. Generates a structured summary with the LLM

    Then generates a cross-paper synthesis.
    """
    state.current_agent = "summarizer_agent"
    state.log("📝 Summarizer Agent: Generating paper summaries...")
    logger.info(f"[SummarizerAgent] {len(state.papers)} papers to summarize")

    llm = get_llm()
    summaries: list[PaperSummary] = []

    # ── Per-paper summaries ───────────────────────────────────────────────────
    valid_papers = [p for p in state.papers if p.download_ok]
    if not valid_papers:
        # Fall back to abstract-only summarization
        valid_papers = state.papers

    for paper in valid_papers:
        state.log(f"  📋 Summarizing: {paper.title[:55]}...")
        try:
            # Retrieve paper-specific context
            chunks = retrieve(
                query=f"{paper.title} methodology results contributions",
                session_id=state.session_id,
                k=4,
                source_filter=paper.arxiv_id if paper.download_ok else None,
            )
            context = format_context(chunks, max_chars=2500)

            prompt = _PAPER_SUMMARY_PROMPT.format(
                title=paper.title,
                authors=", ".join(paper.authors[:3]),
                published=paper.published,
                abstract=paper.abstract[:800],
                context=context or "(No PDF content available — summarizing from abstract only)",
            )

            response = llm.generate(prompt, agent="summarizer_agent", system=_SYSTEM,
                                    temperature=0.3, max_tokens=600)
            state.metrics.total_llm_calls += 1

            summary = _parse_paper_summary(response, paper.arxiv_id, paper.title)
            summaries.append(summary)
            state.log(f"  ✅ Summarized: {paper.arxiv_id}")

        except Exception as exc:
            logger.error(f"[SummarizerAgent] Error on {paper.arxiv_id}: {exc}")
            state.log(f"  ⚠️  Failed to summarize {paper.arxiv_id}: {exc}")
            summaries.append(PaperSummary(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                key_contributions=paper.abstract[:300] if paper.abstract else "N/A",
            ))

    state.summaries = summaries

    # ── Cross-paper synthesis ────────────────────────────────────────────────
    if len(summaries) >= 2:
        state.log("🔗 Generating cross-paper synthesis...")
        try:
            summary_texts = "\n\n".join(
                f"Paper: {s.title}\n"
                f"Contributions: {s.key_contributions}\n"
                f"Methods: {s.methodology}\n"
                f"Results: {s.results}"
                for s in summaries
            )
            prompt = _SYNTHESIS_PROMPT.format(
                topic=state.topic,
                summaries=summary_texts[:4000],
            )
            state.cross_paper_synthesis = llm.generate(
                prompt, agent="summarizer_synthesis",
                system=_SYSTEM, temperature=0.4, max_tokens=600,
            )
            state.metrics.total_llm_calls += 1
            state.log("✅ Cross-paper synthesis complete.")
        except Exception as exc:
            logger.error(f"[SummarizerAgent] Synthesis error: {exc}")
            state.cross_paper_synthesis = "Synthesis generation failed."

    state.log(f"✅ Summarizer done: {len(summaries)} summaries generated.")
    return state
