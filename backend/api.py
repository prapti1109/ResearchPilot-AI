"""
backend/api.py — FastAPI server for ResearchPilot-AI.

Endpoints:
  GET  /health                              — Ollama + system status
  POST /research/start                      — Start new research session
  GET  /research/{session_id}/stream        — SSE stream of progress events
  GET  /research/{session_id}/status        — Current workflow state
  POST /research/{session_id}/approve       — HITL paper approval
  GET  /research/{session_id}/report        — Get final report as JSON
  GET  /research/{session_id}/download/{fmt} — Download PDF/MD/DOCX
  GET  /sessions                            — List all sessions
  DELETE /sessions/{session_id}             — Delete session data
  GET  /metrics                             — Aggregate system metrics
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from config import settings
from tools.llm import get_llm
from workflows.graph import (
    list_sessions,
    load_session_state,
    run_workflow,
    save_session_state,
)
from workflows.states import WorkflowState, WorkflowStatus

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResearchPilot-AI API",
    description="Fully local multi-agent AI research assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session tracking ─────────────────────────────────────────────────
# Maps session_id → asyncio.Task (background workflow runner)
_active_tasks: Dict[str, asyncio.Task] = {}
# Maps session_id → WorkflowState (latest snapshot for SSE streaming)
_session_states: Dict[str, WorkflowState] = {}


# ── Request / Response models ──────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str
    max_papers: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 5
    review_mode: bool = False
    output_format: str = "pdf"
    session_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    approved_paper_ids: Optional[List[str]] = None   # None = approve all


# ── Background workflow runner ─────────────────────────────────────────────────

def _run_workflow_sync(request: ResearchRequest, session_id: str) -> None:
    """Run workflow in a thread (called via BackgroundTasks)."""
    try:
        state = run_workflow(
            topic=request.topic,
            max_papers=request.max_papers,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            retrieval_top_k=request.retrieval_top_k,
            review_mode=request.review_mode,
            output_format=request.output_format,
            session_id=session_id,
        )
        _session_states[session_id] = state
    except Exception as exc:
        logger.exception(f"[API] Background workflow failed: {exc}")
        # Create error state
        err_state = WorkflowState(
            session_id=session_id,
            topic=request.topic,
            status=WorkflowStatus.ERROR,
        )
        err_state.add_error(str(exc))
        _session_states[session_id] = err_state


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check Ollama connectivity and model availability."""
    llm = get_llm()
    health = llm.health_check()
    return {
        "status": "ok" if health["ollama_reachable"] else "degraded",
        "ollama": health,
        "embedding_model": settings.embedding_model,
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.post("/research/start")
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Start a new research workflow session."""
    session_id = request.session_id or str(uuid.uuid4())[:8]

    # Create initial state placeholder
    initial = WorkflowState(
        session_id=session_id,
        topic=request.topic,
        max_papers=request.max_papers,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        retrieval_top_k=request.retrieval_top_k,
        review_mode=request.review_mode,
        output_format=request.output_format,
        status=WorkflowStatus.RUNNING,
    )
    initial.log(f"🚀 Session {session_id} started for: '{request.topic}'")
    _session_states[session_id] = initial

    # Run workflow in background thread
    background_tasks.add_task(_run_workflow_sync, request, session_id)

    logger.info(f"[API] Started session {session_id} — topic='{request.topic}'")
    return {
        "session_id": session_id,
        "status": "running",
        "message": f"Research started for: {request.topic}",
    }


@app.get("/research/{session_id}/stream")
async def stream_progress(session_id: str) -> StreamingResponse:
    """
    SSE endpoint streaming progress log lines in real time.
    Clients should connect and listen; the stream closes when workflow completes.
    """
    async def event_generator():
        sent_lines = 0
        max_wait = 600   # 10 minutes max
        t0 = time.time()

        while time.time() - t0 < max_wait:
            state = _session_states.get(session_id) or load_session_state(session_id)
            if state is None:
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                break

            # Send any new log lines
            new_lines = state.progress_log[sent_lines:]
            for line in new_lines:
                payload = json.dumps({
                    "type": "log",
                    "message": line,
                    "agent": state.current_agent,
                    "status": state.status,
                })
                yield f"data: {payload}\n\n"
                sent_lines += 1

            # Send a heartbeat with summary
            summary = state.to_summary_dict()
            yield f"data: {json.dumps({'type': 'status', **summary})}\n\n"

            if state.status in (WorkflowStatus.COMPLETE, WorkflowStatus.ERROR):
                yield f"data: {json.dumps({'type': 'done', 'status': state.status})}\n\n"
                break

            await asyncio.sleep(1.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/research/{session_id}/status")
async def get_status(session_id: str) -> Dict[str, Any]:
    """Get the current status of a research session."""
    state = _session_states.get(session_id) or load_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return state.to_summary_dict()


@app.post("/research/{session_id}/approve")
async def approve_papers(session_id: str, body: ApprovalRequest) -> Dict[str, Any]:
    """HITL endpoint — approve selected papers to continue the workflow."""
    state = _session_states.get(session_id) or load_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    state.approved_papers = body.approved_paper_ids
    state.status = WorkflowStatus.RUNNING
    state.log(f"✅ User approved {len(body.approved_paper_ids or [])} papers.")
    _session_states[session_id] = state
    save_session_state(state)
    return {"status": "approved", "approved_count": len(body.approved_paper_ids or [])}


@app.get("/research/{session_id}/report")
async def get_report(session_id: str) -> Dict[str, Any]:
    """Return the final report data as JSON."""
    state = _session_states.get(session_id) or load_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if state.status != WorkflowStatus.COMPLETE:
        raise HTTPException(status_code=202, detail="Report not ready yet")

    return {
        "session_id": session_id,
        "topic": state.topic,
        "sections": state.report_sections.model_dump(),
        "summaries": [s.model_dump() for s in state.summaries],
        "verified_findings": [v.model_dump() for v in state.verified_findings],
        "citations_ieee": state.citations_ieee,
        "citations_apa": state.citations_apa,
        "papers": [p.model_dump() for p in state.papers],
        "metrics": state.metrics.model_dump(),
        "report_path": state.final_report_path,
    }


@app.get("/research/{session_id}/download/{fmt}")
async def download_report(session_id: str, fmt: str) -> FileResponse:
    """Download the generated report in the requested format."""
    state = _session_states.get(session_id) or load_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Try the pre-generated file first
    fmt = fmt.lower().strip(".")
    report_path = settings.reports_dir / f"report_{session_id}.{fmt}"

    # If not the right format, regenerate
    if not report_path.exists():
        if state.status != WorkflowStatus.COMPLETE:
            raise HTTPException(status_code=202, detail="Report not ready yet")
        from tools.report_generator import export_report
        report_data = {
            "topic": state.topic,
            "generated_at": time.strftime("%Y-%m-%d %H:%M"),
            "sections": state.report_sections.model_dump(),
            "papers": [p.model_dump() for p in state.papers],
            "citations": state.citations_ieee,
        }
        report_path = export_report(report_data, settings.reports_dir, session_id, fmt)

    media_types = {
        "pdf": "application/pdf",
        "md": "text/markdown",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    return FileResponse(
        str(report_path),
        media_type=media_types.get(fmt, "application/octet-stream"),
        filename=f"ResearchPilot_{session_id}.{fmt}",
    )


@app.get("/sessions")
async def get_sessions() -> List[Dict[str, Any]]:
    """List all previous research sessions."""
    return list_sessions()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, str]:
    """Delete all data for a session (PDFs, FAISS index, state)."""
    # Remove from memory
    _session_states.pop(session_id, None)
    _active_tasks.pop(session_id, None)

    # Remove from disk
    session_dir = settings.data_dir / "sessions" / session_id
    pdf_dir = settings.data_dir / "pdfs" / session_id
    for path in [session_dir, pdf_dir]:
        if path.exists():
            shutil.rmtree(path)

    # Remove report files
    for fmt in ["pdf", "md", "docx"]:
        rp = settings.reports_dir / f"report_{session_id}.{fmt}"
        if rp.exists():
            rp.unlink()

    logger.info(f"[API] Deleted session {session_id}")
    return {"status": "deleted", "session_id": session_id}


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Return aggregate metrics from the metrics.jsonl log."""
    metrics_path = settings.logs_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {"total_calls": 0, "agents": {}, "avg_latency_s": 0}

    records = []
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        return {"total_calls": 0}

    agents: Dict[str, Dict] = {}
    for r in records:
        agent = r.get("agent", "unknown")
        if agent not in agents:
            agents[agent] = {"calls": 0, "total_latency": 0.0, "total_tokens": 0}
        agents[agent]["calls"] += 1
        agents[agent]["total_latency"] += r.get("latency_s", 0)
        agents[agent]["total_tokens"] += r.get("response_tokens", 0)

    avg_latency = sum(r.get("latency_s", 0) for r in records) / len(records)

    return {
        "total_calls": len(records),
        "avg_latency_s": round(avg_latency, 3),
        "agents": {
            k: {
                "calls": v["calls"],
                "avg_latency_s": round(v["total_latency"] / v["calls"], 3),
                "total_tokens": v["total_tokens"],
            }
            for k, v in agents.items()
        },
    }


# ── Dev entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
