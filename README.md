# 🧠 ResearchPilot-AI

> **Fully local, privacy-preserving, multi-agent GenAI research & technical report generation system.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-6c63ff)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

ResearchPilot-AI is a complete agentic AI system that autonomously:

1. **Searches** arXiv for papers on any topic
2. **Downloads** PDFs to your local machine
3. **Parses** and chunks document text (PyMuPDF)
4. **Embeds** chunks into a FAISS vector database (BAAI/bge-small-en)
5. **Retrieves** relevant context via semantic search
6. **Verifies** claims against evidence (hallucination detection)
7. **Summarizes** each paper with key findings & limitations
8. **Generates** a full structured technical report (6 sections)
9. **Formats** IEEE + APA citations
10. **Exports** PDF / Markdown / DOCX — all **100% locally**

---

## Architecture

```
User (Streamlit UI)
        │
        ▼
  FastAPI Backend  ──────────────────────────────────┐
        │                                            │
        ▼                                            ▼
  LangGraph Workflow                          Session Persistence
        │                                    (FAISS + JSON state)
   ┌────┴────────────────────────────────────────┐
   │                                             │
   ▼                                             ▼
Research Agent                          Retrieval Agent
(arXiv search +                        (PDF parse → chunk
 PDF download)                          → embed → FAISS)
   │                                             │
   ▼                                             ▼
Verification Agent ◄─── FAISS ───► Summarizer Agent
(claim fact-check)                  (per-paper + synthesis)
   │                                             │
   └─────────────────┬───────────────────────────┘
                     ▼
               Writer Agent
          (6 report sections)
                     │
                     ▼
             Citation Agent
           (IEEE + APA refs)
                     │
                     ▼
           Final Report (PDF/MD/DOCX)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Local LLM Runtime | [Ollama](https://ollama.ai) |
| Main LLM | `llama3.1:8b` |
| Embedding Model | `BAAI/bge-small-en` |
| Vector Database | FAISS |
| Agent Orchestration | LangGraph |
| Backend API | FastAPI + SSE streaming |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF |
| Report Export | ReportLab (PDF) + python-docx |

---

## Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running:
   ```bash
   # Install from https://ollama.ai
   ollama serve
   ollama pull llama3.1:8b
   ```
3. Internet access for arXiv search + PDF downloads (analysis is fully local)

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourname/ResearchPilot-AI
cd ResearchPilot-AI

# Copy environment config
cp .env.example .env

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the backend

```bash
uvicorn backend.api:app --reload --port 8000
```

### 3. Start the frontend (new terminal)

```bash
streamlit run frontend/app.py --server.port 8501
```

### 4. Open the UI

Navigate to **http://localhost:8501** in your browser.

---

## Docker Deployment

```bash
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501
- Ollama must be running on the **host** machine

---

## Usage

1. Enter a research topic (e.g., *"Transformer models for 6G spectrum sensing"*)
2. Configure: number of papers (1–15), output format (PDF/MD/DOCX)
3. Optional: enable **Review Mode** for human-in-the-loop paper approval
4. Click **🚀 Generate Report**
5. Watch the 6-agent pipeline execute in real time
6. Download your professional technical report

---

## Project Structure

```
ResearchPilot-AI/
├── agents/                    # 6 LangGraph agent modules
│   ├── research_agent.py      # arXiv search + PDF download
│   ├── retrieval_agent.py     # PDF parse → chunk → embed → FAISS
│   ├── verification_agent.py  # Claim fact-checking
│   ├── summarizer_agent.py    # Per-paper summaries + synthesis
│   ├── writer_agent.py        # Report section generation
│   └── citation_agent.py      # IEEE/APA references + export
├── rag/                       # RAG pipeline
│   ├── pdf_loader.py          # PyMuPDF text extraction
│   ├── chunker.py             # Sliding-window text chunker
│   ├── embeddings.py          # BAAI/bge-small-en wrapper
│   ├── vectordb.py            # FAISS index manager
│   └── retriever.py           # Semantic search interface
├── workflows/
│   ├── states.py              # Pydantic state models
│   └── graph.py               # LangGraph StateGraph
├── tools/
│   ├── llm.py                 # Centralized Ollama LLM wrapper
│   ├── arxiv_tool.py          # arXiv search + async downloader
│   └── report_generator.py    # MD / PDF / DOCX exporter
├── backend/
│   └── api.py                 # FastAPI server (REST + SSE)
├── frontend/
│   └── app.py                 # Streamlit dashboard
├── tests/                     # pytest test suite
├── data/                      # PDFs + FAISS indexes (gitignored)
├── reports/                   # Generated reports (gitignored)
├── logs/                      # App logs + metrics.jsonl
├── config.py                  # Central configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Advanced Features

| Feature | Description |
|---|---|
| **Verification Layer** | Every extracted claim is fact-checked against FAISS evidence — hallucination rate tracked |
| **Human-in-the-Loop** | Review Mode pauses after paper retrieval for manual paper approval |
| **Session Persistence** | FAISS index + full state persisted per session; reload any past report |
| **Metrics Dashboard** | Per-agent latency, token count, hallucination rate tracked in `metrics.jsonl` |
| **Async PDF Downloads** | Concurrent PDF fetching (configurable concurrency limit) |
| **Multi-format Export** | PDF (ReportLab), Markdown, DOCX with proper styling |
| **Fully Local** | Zero cloud API calls during analysis — all inference via Ollama |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## API Reference

Interactive docs at **http://localhost:8000/docs** (Swagger UI).

Key endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check Ollama + system status |
| `/research/start` | POST | Start new session |
| `/research/{id}/stream` | GET | SSE progress stream |
| `/research/{id}/status` | GET | Session status |
| `/research/{id}/report` | GET | Full report JSON |
| `/research/{id}/download/{fmt}` | GET | Download PDF/MD/DOCX |
| `/sessions` | GET | List all sessions |
| `/metrics` | GET | Aggregate LLM metrics |

---

## What This Demonstrates

- ✅ Local LLM deployment & inference (Ollama)
- ✅ RAG system (FAISS + sentence-transformers)
- ✅ Multi-agent orchestration (LangGraph)
- ✅ Agentic AI workflow design
- ✅ Vector database engineering
- ✅ Hallucination detection & verification
- ✅ FastAPI production backend (REST + SSE streaming)
- ✅ Streamlit interactive dashboard
- ✅ Docker containerization
- ✅ Full test suite (pytest)

---

## License

MIT License — see [LICENSE](LICENSE).
