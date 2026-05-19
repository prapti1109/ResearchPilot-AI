"""
workflows/graph.py — LangGraph StateGraph orchestrating all 6 agents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from langgraph.graph import END, StateGraph

from agents.citation_agent import run_citation_agent
from agents.research_agent import run_research_agent
from agents.retrieval_agent import run_retrieval_agent
from agents.summarizer_agent import run_summarizer_agent
from agents.verification_agent import run_verification_agent
from agents.writer_agent import run_writer_agent
from config import settings
from workflows.states import WorkflowState, WorkflowStatus


# ── LangGraph requires a plain dict as state, we wrap/unwrap WorkflowState ────

def _to_dict(state: WorkflowState) -> Dict[str, Any]:
    return state.model_dump()


def _from_dict(d: Dict[str, Any]) -> WorkflowState:
    return WorkflowState.model_validate(d)


# ── Node wrappers (LangGraph nodes receive & return dicts) ────────────────────

def node_research(state_dict: Dict) -> Dict:
    state = _from_dict(state_dict)
    state.status = WorkflowStatus.RUNNING
    result = run_research_agent(state)
    return _to_dict(result)


def node_retrieval(state_dict: Dict) -> Dict:
    state = _from_dict(state_dict)
    result = run_retrieval_agent(state)
    return _to_dict(result)


def node_verification(state_dict: Dict) -> Dict:
    state = _from_dict(state_dict)
    result = run_verification_agent(state)
    return _to_dict(result)


def node_summarization(state_dict: Dict) -> Dict:
    state = _from_dict(state_dict)
    result = run_summarizer_agent(state)
    return _to_dict(result)


def node_writing(state_dict: Dict) -> Dict:
    state = _from_dict(state_dict)
    result = run_writer_agent(state)
    return _to_dict(result)


def node_citation(state_dict: Dict) -> Dict:
    state = _from_dict(state_dict)
    result = run_citation_agent(state)
    result_state = _from_dict(_to_dict(result) if isinstance(result, dict) else result.model_dump())
    result_state.status = WorkflowStatus.COMPLETE
    return _to_dict(result_state)


# ── Conditional routing ───────────────────────────────────────────────────────

def _route_after_research(state_dict: Dict) -> str:
    """Skip to END if research found no downloadable papers."""
    state = _from_dict(state_dict)
    if state.errors and not state.papers:
        logger.warning("[Graph] No papers found — aborting workflow.")
        return END
    return "retrieval"


def _route_after_retrieval(state_dict: Dict) -> str:
    """Skip verification/summarization if indexing failed."""
    state = _from_dict(state_dict)
    if not state.chunks:
        logger.warning("[Graph] No chunks indexed — jumping to citation for abstract-only report.")
        return "summarization"
    return "verification"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Compile and return the LangGraph StateGraph."""
    builder = StateGraph(dict)

    # Register nodes
    builder.add_node("research", node_research)
    builder.add_node("retrieval", node_retrieval)
    builder.add_node("verification", node_verification)
    builder.add_node("summarization", node_summarization)
    builder.add_node("writing", node_writing)
    builder.add_node("citation", node_citation)

    # Set entry point
    builder.set_entry_point("research")

    # Edges with conditional routing
    builder.add_conditional_edges("research", _route_after_research)
    builder.add_conditional_edges("retrieval", _route_after_retrieval)
    builder.add_edge("verification", "summarization")
    builder.add_edge("summarization", "writing")
    builder.add_edge("writing", "citation")
    builder.add_edge("citation", END)

    graph = builder.compile()
    logger.info("[Graph] LangGraph StateGraph compiled successfully.")
    return graph


# ── Session state persistence ─────────────────────────────────────────────────

def save_session_state(state: WorkflowState) -> None:
    """Persist WorkflowState as JSON to the session directory."""
    session_dir = settings.session_dir(state.session_id)
    state_path = session_dir / "state.json"
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    logger.debug(f"[Graph] Session state saved: {state_path}")


def load_session_state(session_id: str) -> WorkflowState | None:
    """Load WorkflowState from disk. Returns None if not found."""
    state_path = settings.session_dir(session_id) / "state.json"
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return WorkflowState.model_validate(data)
    except Exception as exc:
        logger.error(f"[Graph] Failed to load session {session_id}: {exc}")
        return None


def list_sessions() -> list[dict]:
    """Return summary of all persisted sessions."""
    sessions_root = settings.data_dir / "sessions"
    results = []
    if not sessions_root.exists():
        return results
    for session_dir in sessions_root.iterdir():
        state_path = session_dir / "state.json"
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                results.append({
                    "session_id": data.get("session_id"),
                    "topic": data.get("topic"),
                    "status": data.get("status"),
                    "papers_found": len(data.get("papers", [])),
                    "report_path": data.get("final_report_path"),
                    "elapsed_s": data.get("metrics", {}).get("total_latency_s", 0),
                })
            except Exception:
                pass
    return sorted(results, key=lambda x: x.get("session_id", ""), reverse=True)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_workflow(
    topic: str,
    max_papers: int = 5,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    retrieval_top_k: int = 5,
    review_mode: bool = False,
    output_format: str = "pdf",
    session_id: str | None = None,
) -> WorkflowState:
    """
    Execute the full ResearchPilot-AI workflow synchronously.

    Returns the final WorkflowState after all agents have run.
    """
    import uuid
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    initial_state = WorkflowState(
        session_id=session_id,
        topic=topic,
        max_papers=max_papers,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_top_k=retrieval_top_k,
        review_mode=review_mode,
        output_format=output_format,
    )
    initial_state.status = WorkflowStatus.RUNNING
    initial_state.log(f"🚀 Starting ResearchPilot-AI workflow for: '{topic}'")

    graph = build_graph()

    logger.info(f"[Graph] Running workflow session={session_id} topic='{topic}'")
    try:
        final_dict = graph.invoke(_to_dict(initial_state))
        final_state = _from_dict(final_dict)
    except Exception as exc:
        logger.exception(f"[Graph] Workflow failed: {exc}")
        initial_state.add_error(str(exc))
        initial_state.status = WorkflowStatus.ERROR
        final_state = initial_state

    save_session_state(final_state)
    return final_state
