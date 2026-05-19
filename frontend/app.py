"""
frontend/app.py — Streamlit multi-page dashboard for ResearchPilot-AI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchPilot-AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d0d1a; color: #e2e2f0; }
section[data-testid="stSidebar"] { background: #12122a; border-right: 1px solid #2a2a4a; }

.hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub { font-size: 1rem; color: #888aaa; margin-top: 0.2rem; }

.agent-card {
    background: #1a1a3a; border: 1px solid #2e2e5a;
    border-radius: 12px; padding: 1rem 1.2rem; margin: 0.4rem 0;
    display: flex; align-items: center; gap: 0.8rem;
    transition: all 0.3s ease;
}
.agent-card.active { border-color: #6c63ff; background: #1e1b4b; box-shadow: 0 0 16px #6c63ff44; }
.agent-card.done   { border-color: #22c55e; background: #14291e; }
.agent-card.error  { border-color: #ef4444; background: #291414; }

.metric-chip {
    background: #1e1e3f; border: 1px solid #3a3a6a;
    border-radius: 8px; padding: 0.6rem 1rem; text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.75rem; color: #888aaa; text-transform: uppercase; letter-spacing: 0.05em; }

.status-badge {
    display: inline-block; padding: 0.2rem 0.7rem;
    border-radius: 999px; font-size: 0.75rem; font-weight: 600;
}
.badge-running  { background: #1d4ed844; color: #60a5fa; border: 1px solid #3b82f6; }
.badge-complete { background: #15803d44; color: #4ade80; border: 1px solid #22c55e; }
.badge-error    { background: #991b1b44; color: #f87171; border: 1px solid #ef4444; }
.badge-idle     { background: #37415144; color: #9ca3af; border: 1px solid #6b7280; }

.paper-card {
    background: #141430; border: 1px solid #2a2a50;
    border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
}
.paper-title { font-weight: 600; font-size: 0.95rem; color: #c4b5fd; }
.paper-meta  { font-size: 0.78rem; color: #7878a0; margin-top: 0.2rem; }

.report-preview {
    background: #0f0f28; border: 1px solid #2a2a50;
    border-radius: 10px; padding: 1.5rem; line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

AGENT_PIPELINE = [
    ("research_agent",      "🔍", "Research Agent",      "Searches arXiv & downloads PDFs"),
    ("retrieval_agent",     "📚", "Retrieval Agent",     "Embeds chunks into FAISS vector DB"),
    ("verification_agent",  "🔬", "Verification Agent",  "Fact-checks claims against evidence"),
    ("summarizer_agent",    "📝", "Summarizer Agent",    "Generates per-paper summaries"),
    ("writer_agent",        "✍️", "Writer Agent",        "Writes all report sections"),
    ("citation_agent",      "📚", "Citation Agent",      "Generates IEEE/APA citations & exports"),
]


# ── API helpers ────────────────────────────────────────────────────────────────

def api_get(path: str, timeout: float = 10.0) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, body: dict, timeout: float = 15.0) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Session state helpers ──────────────────────────────────────────────────────

def init_session():
    defaults = {
        "session_id": None,
        "workflow_running": False,
        "workflow_done": False,
        "progress_log": [],
        "status_data": {},
        "report_data": None,
        "active_tab": "Research Hub",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def poll_status():
    """Poll API for status and update session state."""
    sid = st.session_state.get("session_id")
    if not sid:
        return
    data = api_get(f"/research/{sid}/status", timeout=5)
    if data and "error" not in data:
        st.session_state.status_data = data
        logs = data.get("progress_log", [])
        st.session_state.progress_log = logs
        status = data.get("status", "")
        if status in ("complete", "error"):
            st.session_state.workflow_running = False
            st.session_state.workflow_done = (status == "complete")


# ── Agent pipeline visual ──────────────────────────────────────────────────────

def render_agent_pipeline(current_agent: str, status: str):
    completed_agents = []
    found_current = False
    for agent_id, *_ in AGENT_PIPELINE:
        if agent_id == current_agent:
            found_current = True
        if not found_current:
            completed_agents.append(agent_id)

    for agent_id, icon, label, desc in AGENT_PIPELINE:
        if status == "complete":
            css_class = "done"
            state_icon = "✅"
        elif agent_id == current_agent and status == "running":
            css_class = "active"
            state_icon = "⏳"
        elif agent_id in completed_agents:
            css_class = "done"
            state_icon = "✅"
        else:
            css_class = ""
            state_icon = "⭕"

        st.markdown(f"""
        <div class="agent-card {css_class}">
            <span style="font-size:1.4rem">{state_icon}</span>
            <span style="font-size:1.3rem">{icon}</span>
            <div>
                <div style="font-weight:600;color:#c4b5fd">{label}</div>
                <div style="font-size:0.78rem;color:#7878a0">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="hero-title">🧠 ResearchPilot</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Local Multi-Agent AI Research System</div>', unsafe_allow_html=True)
        st.divider()

        # System health
        health = api_get("/health", timeout=8)
        if health and "error" not in health:
            ollama_ok = health.get("ollama", {}).get("ollama_reachable", False)
            model_ok = health.get("ollama", {}).get("model_available", False)
            st.markdown("**System Status**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Backend", "🟢 On")
            c2.metric("Ollama", "🟢 On" if ollama_ok else "🔴 Off")
            c3.metric("Model", "🟢 Ready" if model_ok else "🔴 N/A")
            if not ollama_ok:
                st.warning("Start Ollama: `ollama serve` then `ollama pull llama3.1:8b`")
            elif not model_ok:
                st.warning("Pull model: `ollama pull llama3.1:8b`")
        else:
            st.error("⚠️ Backend not reachable. Start it with:\n`uvicorn backend.api:app --port 8000`")

        st.divider()
        page = st.radio("Navigation", ["🔬 Research Hub", "📁 Session History", "📊 System Metrics"],
                        label_visibility="collapsed")
        return page


# ── Page: Research Hub ─────────────────────────────────────────────────────────

def page_research_hub():
    st.markdown('<h1 class="hero-title">Research Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Enter a topic — the AI agents will do the rest.</p>',
                unsafe_allow_html=True)
    st.divider()

    # ── Input panel ───────────────────────────────────────────────────────────
    with st.container():
        col_inp, col_cfg = st.columns([2, 1])
        with col_inp:
            topic = st.text_input(
                "🎯 Research Topic",
                placeholder="e.g. AI-Based Spectrum Sensing in 6G",
                help="Enter any AI/ML research topic to generate a full technical survey.",
            )
        with col_cfg:
            max_papers = st.slider("Papers to fetch", 1, 15, 5)
            top_k = st.slider("Retrieval chunks (k)", 3, 10, 5)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            output_fmt = st.selectbox("Output format", ["pdf", "md", "docx"])
        with col_b:
            review_mode = st.toggle("Review Mode (HITL)", value=False,
                                    help="Pause after paper retrieval for human approval")
        with col_c:
            chunk_size = st.number_input("Chunk size (chars)", 200, 1000, 500, step=50)

        run_btn = st.button("🚀 Generate Report", type="primary",
                            disabled=st.session_state.workflow_running,
                            use_container_width=True)

    if run_btn and topic:
        payload = {
            "topic": topic,
            "max_papers": max_papers,
            "retrieval_top_k": top_k,
            "output_format": output_fmt,
            "review_mode": review_mode,
            "chunk_size": chunk_size,
            "chunk_overlap": 100,
        }
        resp = api_post("/research/start", payload)
        if resp and "error" not in resp:
            st.session_state.session_id = resp["session_id"]
            st.session_state.workflow_running = True
            st.session_state.workflow_done = False
            st.session_state.progress_log = []
            st.session_state.status_data = {}
            st.session_state.report_data = None
            st.rerun()
        else:
            st.error(f"Failed to start: {resp}")

    # ── Live workflow panel ────────────────────────────────────────────────────
    if st.session_state.session_id:
        st.divider()
        sid = st.session_state.session_id
        status_data = st.session_state.status_data
        current_agent = status_data.get("current_agent", "")
        wf_status = status_data.get("status", "idle")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("### 🤖 Agent Pipeline")
            render_agent_pipeline(current_agent, wf_status)

            if st.session_state.workflow_running:
                st.markdown("---")
                st.markdown("**Live Metrics**")
                elapsed = status_data.get("elapsed_s", 0)
                c1, c2 = st.columns(2)
                c1.metric("Papers", status_data.get("papers_found", 0))
                c2.metric("Chunks", status_data.get("chunks_indexed", 0))
                c3, c4 = st.columns(2)
                c3.metric("Summaries", status_data.get("summaries_done", 0))
                c4.metric("Elapsed", f"{elapsed:.0f}s")

        with col_right:
            st.markdown("### 📟 Progress Log")
            log_container = st.empty()
            logs = st.session_state.progress_log
            log_text = "\n".join(logs[-30:]) if logs else "Waiting for workflow to start..."
            log_container.code(log_text, language=None)

            # Auto-poll when running
            if st.session_state.workflow_running:
                poll_status()
                time.sleep(2)
                st.rerun()

        # ── HITL approval panel ───────────────────────────────────────────────
        if (wf_status == "awaiting_approval" and
                st.session_state.status_data.get("papers_found", 0) > 0):
            st.info("🛑 **Review Mode**: Approve papers before continuing.")
            papers_data = api_get(f"/research/{sid}/report")
            if papers_data and "papers" in papers_data:
                selected = []
                for p in papers_data["papers"]:
                    if st.checkbox(p["title"], value=True, key=f"approve_{p['arxiv_id']}"):
                        selected.append(p["arxiv_id"])
                if st.button("✅ Approve & Continue"):
                    api_post(f"/research/{sid}/approve", {"approved_paper_ids": selected})
                    st.rerun()

        # ── Results tabs ──────────────────────────────────────────────────────
        if status_data.get("papers_found", 0) > 0 or st.session_state.workflow_done:
            st.divider()
            st.markdown("### 📊 Results")
            tabs = st.tabs(["📄 Papers", "🔍 Chunks", "✅ Verification",
                            "📝 Summaries", "📑 Report Preview", "📥 Download"])

            report = api_get(f"/research/{sid}/report", timeout=5)
            if report and "error" not in report:
                st.session_state.report_data = report

            rd = st.session_state.report_data or {}

            # Papers tab
            with tabs[0]:
                papers = rd.get("papers", [])
                if papers:
                    for p in papers:
                        st.markdown(f"""
                        <div class="paper-card">
                            <div class="paper-title">{p.get('title','')}</div>
                            <div class="paper-meta">
                                👤 {', '.join(p.get('authors', [])[:3])} &nbsp;|&nbsp;
                                📅 {p.get('published','')} &nbsp;|&nbsp;
                                🆔 {p.get('arxiv_id','')} &nbsp;|&nbsp;
                                {'✅ Downloaded' if p.get('download_ok') else '❌ Failed'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("Abstract"):
                            st.write(p.get("abstract", ""))
                else:
                    st.info("Papers will appear here once the research agent completes.")

            # Chunks tab
            with tabs[1]:
                chunks_count = status_data.get("chunks_indexed", 0)
                st.metric("Total chunks indexed", chunks_count)
                top_k_display = top_k
                st.caption(f"Showing top retrieved chunks for the query (k={top_k_display})")
                # We show them from status only (full chunks not in status API)
                st.info("Chunks are stored in the FAISS index. Use the report sections to see retrieved context.")

            # Verification tab
            with tabs[2]:
                findings = rd.get("verified_findings", [])
                if findings:
                    for f in findings:
                        verdict = f.get("verdict", "unverified")
                        icon = "✅" if verdict == "verified" else ("❌" if verdict == "contradicted" else "⚠️")
                        color = "#22c55e" if verdict == "verified" else ("#ef4444" if verdict == "contradicted" else "#f59e0b")
                        st.markdown(f"""
                        <div style="border-left: 4px solid {color}; padding: 0.6rem 1rem;
                                    background: #1a1a3a; border-radius: 0 8px 8px 0; margin: 0.4rem 0;">
                            <strong>{icon} [{verdict.upper()}]</strong> — conf: {f.get('confidence', 0):.0%}<br/>
                            <span style="font-size:0.85rem">{f.get('claim','')[:200]}</span><br/>
                            <span style="font-size:0.78rem; color:#888">{f.get('evidence','')[:150]}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Verification results will appear here.")

            # Summaries tab
            with tabs[3]:
                summaries = rd.get("summaries", [])
                if summaries:
                    for s in summaries:
                        with st.expander(f"📄 {s.get('title', '')[:70]}"):
                            c1, c2 = st.columns(2)
                            c1.markdown("**Key Contributions**")
                            c1.write(s.get("key_contributions", "N/A"))
                            c2.markdown("**Methodology**")
                            c2.write(s.get("methodology", "N/A"))
                            c1.markdown("**Results**")
                            c1.write(s.get("results", "N/A"))
                            c2.markdown("**Limitations**")
                            c2.write(s.get("limitations", "N/A"))
                else:
                    st.info("Summaries will appear here after the summarizer agent runs.")

            # Report preview tab
            with tabs[4]:
                sections = rd.get("sections", {})
                if any(sections.values()):
                    topic_title = status_data.get("topic", "Research Report")
                    st.markdown(f"# {topic_title}")
                    st.divider()
                    section_labels = [
                        ("introduction", "1. Introduction"),
                        ("related_work", "2. Related Work"),
                        ("method_comparison", "3. Methodology Comparison"),
                        ("key_findings", "4. Key Findings"),
                        ("challenges", "5. Challenges & Limitations"),
                        ("future_work", "6. Future Work"),
                    ]
                    for key, label in section_labels:
                        content = sections.get(key, "")
                        if content:
                            st.markdown(f"## {label}")
                            st.write(content)
                    refs = rd.get("citations_ieee", [])
                    if refs:
                        st.markdown("## References")
                        for ref in refs:
                            st.markdown(f"- {ref}")
                else:
                    st.info("Report preview will appear here after the writer agent completes.")

            # Download tab
            with tabs[5]:
                if st.session_state.workflow_done:
                    st.success("✅ Report is ready for download!")
                    col1, col2, col3 = st.columns(3)
                    for col, fmt, icon in [(col1, "pdf", "📕"), (col2, "md", "📝"), (col3, "docx", "📘")]:
                        with col:
                            try:
                                r = httpx.get(f"{API_BASE}/research/{sid}/download/{fmt}", timeout=30)
                                if r.status_code == 200:
                                    col.download_button(
                                        f"{icon} Download {fmt.upper()}",
                                        data=r.content,
                                        file_name=f"ResearchPilot_{sid}.{fmt}",
                                        use_container_width=True,
                                    )
                            except Exception as e:
                                col.error(str(e))
                else:
                    st.info("Download buttons will appear when the workflow completes.")


# ── Page: Session History ──────────────────────────────────────────────────────

def page_session_history():
    st.markdown('<h1 class="hero-title">Session History</h1>', unsafe_allow_html=True)
    st.divider()

    sessions = api_get("/sessions")
    if not sessions or "error" in (sessions or {}):
        st.info("No previous sessions found.")
        return

    for s in (sessions if isinstance(sessions, list) else []):
        status = s.get("status", "unknown")
        badge_class = {
            "complete": "badge-complete",
            "running": "badge-running",
            "error": "badge-error",
        }.get(status, "badge-idle")

        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.markdown(f"""
            **{s.get('topic', 'Unknown topic')}**  
            <span class="status-badge {badge_class}">{status.upper()}</span>
            &nbsp; Session: `{s.get('session_id', '')}` &nbsp; Papers: {s.get('papers_found', 0)}
            """, unsafe_allow_html=True)

            if col2.button("📂 Load", key=f"load_{s['session_id']}"):
                st.session_state.session_id = s["session_id"]
                st.session_state.workflow_done = (status == "complete")
                st.session_state.workflow_running = False
                poll_status()
                st.rerun()

            if col3.button("🗑️ Delete", key=f"del_{s['session_id']}"):
                api_post(f"/sessions/{s['session_id']}", {})
                st.rerun()

            st.divider()


# ── Page: System Metrics ───────────────────────────────────────────────────────

def page_system_metrics():
    st.markdown('<h1 class="hero-title">System Metrics</h1>', unsafe_allow_html=True)
    st.divider()

    health = api_get("/health")
    metrics = api_get("/metrics")

    if health and "error" not in health:
        st.subheader("🩺 System Health")
        ollama = health.get("ollama", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Ollama", "🟢 Online" if ollama.get("ollama_reachable") else "🔴 Offline")
        c2.metric("Model", ollama.get("model", "N/A"))
        c3.metric("Embedding", health.get("embedding_model", "N/A"))

        if ollama.get("all_models"):
            with st.expander("All loaded Ollama models"):
                for m in ollama["all_models"]:
                    st.code(m)

    if metrics and "error" not in (metrics or {}):
        st.subheader("📈 LLM Call Metrics")
        c1, c2 = st.columns(2)
        c1.metric("Total LLM Calls", metrics.get("total_calls", 0))
        c2.metric("Avg Latency", f"{metrics.get('avg_latency_s', 0):.2f}s")

        agents_data = metrics.get("agents", {})
        if agents_data:
            st.subheader("Per-Agent Breakdown")
            rows = []
            for agent, data in agents_data.items():
                rows.append({
                    "Agent": agent,
                    "Calls": data["calls"],
                    "Avg Latency (s)": data["avg_latency_s"],
                    "Total Tokens": data["total_tokens"],
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No metrics data yet. Run a research workflow to generate metrics.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_session()
    page = render_sidebar()

    if "Research Hub" in page:
        page_research_hub()
    elif "Session History" in page:
        page_session_history()
    elif "Metrics" in page:
        page_system_metrics()


if __name__ == "__main__":
    main()
