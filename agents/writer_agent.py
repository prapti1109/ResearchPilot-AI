"""
agents/writer_agent.py — Agent 5: Generate all report sections via LLM.
"""
from __future__ import annotations

from loguru import logger

from rag.retriever import format_context, retrieve
from tools.llm import get_llm
from workflows.states import ReportSections, WorkflowState

_SYSTEM = (
    "You are an expert technical writer specializing in AI and machine learning research. "
    "Write in formal academic English. Be precise, comprehensive, and well-structured."
)

_INTRO_PROMPT = """
Write a professional Introduction section (250-350 words) for a technical survey on: "{topic}"

Use this context from retrieved papers:
{context}

The introduction should cover:
1. Background and motivation for this research area
2. Key challenges in the field
3. Overview of what this report covers
4. Importance to the broader AI/ML community

Do not use bullet points. Write in flowing academic prose.
"""

_RELATED_WORK_PROMPT = """
Write a Related Work section (350-450 words) for a technical survey on: "{topic}"

Based on these paper summaries:
{summaries}

Structure as coherent paragraphs grouping papers by approach/theme.
Reference papers by title. Do not use generic phrases like "many researchers have studied".
"""

_METHOD_COMPARISON_PROMPT = """
Write a Methodology Comparison section (300-400 words) comparing approaches found in research on: "{topic}"

Paper summaries:
{summaries}

Include:
- Compare key algorithmic/methodological approaches
- Note similarities and fundamental differences
- Discuss tradeoffs between methods
- Mention evaluation metrics used

Use academic language. Reference specific papers.
"""

_FINDINGS_PROMPT = """
Write a Key Findings section (250-350 words) synthesizing results from research on: "{topic}"

Synthesis: {synthesis}

Verified findings: {verified}

Highlight the most significant quantitative and qualitative findings.
Note any consensus or disagreement across papers.
"""

_CHALLENGES_PROMPT = """
Write a Challenges and Limitations section (200-300 words) for a survey on: "{topic}"

Based on paper limitations: {limitations}

Cover:
1. Technical challenges in the field
2. Dataset/evaluation limitations
3. Scalability and deployment challenges
4. Ethical or societal considerations if relevant
"""

_FUTURE_WORK_PROMPT = """
Write a Future Work and Open Challenges section (200-300 words) for a survey on: "{topic}"

Based on identified gaps and limitations:
{gaps}

Suggest concrete future research directions. Be specific and forward-looking.
"""


def run_writer_agent(state: WorkflowState) -> WorkflowState:
    """
    LangGraph node: Report Writer Agent.

    Generates all six report sections using the LLM with context from
    summaries, verified findings, and retrieved chunks.
    """
    state.current_agent = "writer_agent"
    state.log("✍️  Writer Agent: Generating report sections...")
    logger.info(f"[WriterAgent] session={state.session_id}")

    llm = get_llm()
    sections = ReportSections()

    # Prepare shared context strings
    summary_text = "\n\n".join(
        f"**{s.title}**\n"
        f"Contributions: {s.key_contributions}\n"
        f"Methods: {s.methodology}\n"
        f"Results: {s.results}"
        for s in state.summaries
    )[:4000]

    limitations_text = "\n".join(
        f"- {s.title}: {s.limitations}" for s in state.summaries if s.limitations
    )[:2000]

    verified_text = "\n".join(
        f"[{v.verdict.value.upper()}] {v.claim}" for v in state.verified_findings
    )[:2000]

    # ── Introduction ──────────────────────────────────────────────────────────
    state.log("  📄 Writing Introduction...")
    try:
        chunks = retrieve(state.topic, state.session_id, k=3)
        context = format_context(chunks, max_chars=2000)
        sections.introduction = llm.generate(
            _INTRO_PROMPT.format(topic=state.topic, context=context),
            agent="writer_intro", system=_SYSTEM, temperature=0.4, max_tokens=512,
        )
        state.metrics.total_llm_calls += 1
    except Exception as exc:
        logger.error(f"[WriterAgent] Intro error: {exc}")
        sections.introduction = f"Introduction generation failed: {exc}"

    # ── Related Work ──────────────────────────────────────────────────────────
    state.log("  📄 Writing Related Work...")
    try:
        sections.related_work = llm.generate(
            _RELATED_WORK_PROMPT.format(topic=state.topic, summaries=summary_text),
            agent="writer_related", system=_SYSTEM, temperature=0.35, max_tokens=600,
        )
        state.metrics.total_llm_calls += 1
    except Exception as exc:
        logger.error(f"[WriterAgent] Related work error: {exc}")
        sections.related_work = "Related work generation failed."

    # ── Method Comparison ─────────────────────────────────────────────────────
    state.log("  📄 Writing Method Comparison...")
    try:
        sections.method_comparison = llm.generate(
            _METHOD_COMPARISON_PROMPT.format(topic=state.topic, summaries=summary_text),
            agent="writer_methods", system=_SYSTEM, temperature=0.3, max_tokens=600,
        )
        state.metrics.total_llm_calls += 1
    except Exception as exc:
        logger.error(f"[WriterAgent] Methods error: {exc}")
        sections.method_comparison = "Methodology comparison generation failed."

    # ── Key Findings ──────────────────────────────────────────────────────────
    state.log("  📄 Writing Key Findings...")
    try:
        sections.key_findings = llm.generate(
            _FINDINGS_PROMPT.format(
                topic=state.topic,
                synthesis=state.cross_paper_synthesis[:1500],
                verified=verified_text,
            ),
            agent="writer_findings", system=_SYSTEM, temperature=0.35, max_tokens=512,
        )
        state.metrics.total_llm_calls += 1
    except Exception as exc:
        logger.error(f"[WriterAgent] Findings error: {exc}")
        sections.key_findings = "Key findings generation failed."

    # ── Challenges ────────────────────────────────────────────────────────────
    state.log("  📄 Writing Challenges...")
    try:
        sections.challenges = llm.generate(
            _CHALLENGES_PROMPT.format(topic=state.topic, limitations=limitations_text),
            agent="writer_challenges", system=_SYSTEM, temperature=0.3, max_tokens=450,
        )
        state.metrics.total_llm_calls += 1
    except Exception as exc:
        logger.error(f"[WriterAgent] Challenges error: {exc}")
        sections.challenges = "Challenges section generation failed."

    # ── Future Work ───────────────────────────────────────────────────────────
    state.log("  📄 Writing Future Work...")
    try:
        gaps = limitations_text + "\n" + state.cross_paper_synthesis[-500:]
        sections.future_work = llm.generate(
            _FUTURE_WORK_PROMPT.format(topic=state.topic, gaps=gaps[:2000]),
            agent="writer_future", system=_SYSTEM, temperature=0.45, max_tokens=450,
        )
        state.metrics.total_llm_calls += 1
    except Exception as exc:
        logger.error(f"[WriterAgent] Future work error: {exc}")
        sections.future_work = "Future work section generation failed."

    state.report_sections = sections
    state.log("✅ Writer Agent: All report sections generated.")
    return state
