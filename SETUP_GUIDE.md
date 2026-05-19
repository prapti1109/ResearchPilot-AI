# 🚀 ResearchPilot-AI — Complete Setup & Run Guide

> Step-by-step instructions to get ResearchPilot-AI fully running on your Windows machine.
> Total setup time: ~15–20 minutes (mostly model download).

---

## 📋 What You Need Before Starting

| Requirement | Minimum | Recommended |
|---|---|---|
| Operating System | Windows 10/11 | Windows 11 |
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk Space | 10 GB free | 15 GB free |
| Internet | Required for setup + paper search | — |
| GPU | Not required | NVIDIA GPU (faster inference) |

---

## STEP 1 — Install Python (skip if already installed)

### Check if Python is installed

Open **PowerShell** (press `Win + X` → select "Windows PowerShell") and run:

```powershell
python --version
```

If you see `Python 3.10+`, skip to Step 2.

### Install Python

1. Go to: https://www.python.org/downloads/
2. Download Python 3.11 or later
3. Run the installer
4. ⚠️ **CHECK** the box that says **"Add Python to PATH"** — this is critical
5. Click "Install Now"
6. Verify by reopening PowerShell and running `python --version`

---

## STEP 2 — Install Ollama (Local LLM Runtime)

Ollama is the engine that runs the AI model locally on your machine.

### 2a. Download Ollama

1. Go to: **https://ollama.ai/download**
2. Click **"Download for Windows"**
3. Run the installer (`OllamaSetup.exe`)
4. Follow the installation wizard (click Next → Install → Finish)

### 2b. Start the Ollama server

Open a **new PowerShell terminal** (this is Terminal 1) and run:

```powershell
ollama serve
```

You should see output like:
```
time=... level=INFO source=... msg="Listening on 127.0.0.1:11434"
```

> ⚠️ **Keep this terminal open** — Ollama must stay running the entire time.

### 2c. Pull the LLM model

Open a **second PowerShell terminal** (Terminal 2) and run:

```powershell
ollama pull llama3.1:8b
```

This downloads the llama3.1:8b model (~4.7 GB). Wait for it to finish.

You will see:
```
pulling manifest
pulling aabd4debf0c8... 100% ████████████████ 4.7 GB
...
success
```

### 2d. Verify Ollama is working

In Terminal 2, run:

```powershell
ollama list
```

You should see `llama3.1:8b` in the list. If yes, Ollama is ready. ✅

---

## STEP 3 — Navigate to the Project

In **Terminal 2**, run:

```powershell
cd "C:\Users\Administrator\.gemini\antigravity\scratch\ResearchPilot-AI"
```

Verify you're in the right folder:

```powershell
dir
```

You should see files like `requirements.txt`, `config.py`, `README.md`, and folders like `agents/`, `rag/`, `frontend/`, etc.

---

## STEP 4 — Install Python Dependencies

In **Terminal 2**, run:

```powershell
pip install -r requirements.txt
```

This installs all required Python packages (~100+ packages including FastAPI, Streamlit, LangGraph, FAISS, PyMuPDF, sentence-transformers, etc.)

Wait for it to complete. You should see:
```
Successfully installed ... (long list of packages)
```

> ⏱️ This takes 3–5 minutes on a good internet connection.
> The embedding model (`BAAI/bge-small-en`, ~130 MB) will auto-download on first use.

---

## STEP 5 — Create the Environment Config

In **Terminal 2**, run:

```powershell
copy .env.example .env
```

This creates the `.env` file with default settings. The defaults work out of the box — no editing needed.

---

## STEP 6 — Start the FastAPI Backend Server

In **Terminal 2**, run:

```powershell
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Verify the backend is running

Open your browser and go to: **http://localhost:8000/docs**

You should see the **ResearchPilot-AI API** Swagger documentation page with all 10 endpoints listed.

> ⚠️ **Keep Terminal 2 open** — the backend must stay running.

---

## STEP 7 — Start the Streamlit Frontend

Open a **third PowerShell terminal** (Terminal 3) and run:

```powershell
cd "C:\Users\Administrator\.gemini\antigravity\scratch\ResearchPilot-AI"
python -m streamlit run frontend/app.py --server.port 8501
```

You should see:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://X.X.X.X:8501
```

> ⚠️ **Keep Terminal 3 open** — the frontend must stay running.

---

## STEP 8 — Open the Application

Open your browser and go to: **http://localhost:8501**

### What you should see:

**Sidebar (left):**
- 🧠 **ResearchPilot** branding
- **System Status** showing:
  - Backend: 🟢 On
  - Ollama: 🟢 On
  - Model: 🟢 Ready
- Navigation: Research Hub / Session History / System Metrics

**Main area:**
- "Research Hub" heading
- Research Topic input field
- Papers to fetch slider (1–15)
- Retrieval chunks slider (3–10)
- Output format dropdown (pdf / md / docx)
- Review Mode (HITL) toggle
- 🚀 **Generate Report** button

If the sidebar shows all green indicators — **you're fully set up!** ✅

---

## STEP 9 — Generate Your First Research Report

1. In the **Research Topic** field, type:
   ```
   AI-Based Spectrum Sensing in 6G
   ```

2. Set **Papers to fetch** to **3** (start small for your first run)

3. Set **Output format** to **pdf**

4. Leave **Review Mode** off (Auto mode)

5. Click **🚀 Generate Report**

### What happens next:

The system runs 6 AI agents sequentially. You'll see real-time progress:

```
[14:30:01] 🚀 Starting ResearchPilot-AI workflow for: 'AI-Based Spectrum Sensing in 6G'
[14:30:02] 🔍 Research Agent: Searching 'AI-Based Spectrum Sensing in 6G' (max=3)
[14:30:05] ✅ Found 3 papers on arXiv.
[14:30:06] 📥 Downloading 3 PDFs...
[14:30:15] ✅ Downloaded 3/3 PDFs successfully.
[14:30:16] 📚 Retrieval Agent: Processing PDFs into vector database...
[14:30:20]   📄 Parsing: paper title...
[14:30:30] ✅ Vector DB ready: 120 chunks from 3 papers.
[14:30:31] 🔬 Verification Agent: Checking claims against evidence...
[14:31:00] ✅ Verification done: 8 claims (1 contradicted)
[14:31:01] 📝 Summarizer Agent: Generating paper summaries...
[14:32:00] ✅ Summarizer done: 3 summaries generated.
[14:32:01] ✍️  Writer Agent: Generating report sections...
[14:34:00] ✅ Writer Agent: All report sections generated.
[14:34:01] 📚 Citation Agent: Generating references...
[14:34:05] ✅ Report exported: report_abc123.pdf
[14:34:05] 🎉 Workflow complete in 245.3s
```

### Time estimate:
- 3 papers → ~4–8 minutes
- 5 papers → ~8–15 minutes
- 10 papers → ~20–30 minutes

> ⏱️ Times depend on your hardware. GPU acceleration makes inference ~3x faster.

---

## STEP 10 — View and Download Results

After the workflow completes, the **Results** section appears with 6 tabs:

| Tab | What It Shows |
|---|---|
| 📄 **Papers** | All retrieved papers with titles, authors, abstracts |
| 🔍 **Chunks** | Number of text chunks indexed in FAISS |
| ✅ **Verification** | Fact-check results: VERIFIED / UNVERIFIED / CONTRADICTED |
| 📝 **Summaries** | Per-paper summaries with contributions, methods, results, limitations |
| 📑 **Report Preview** | Full rendered Markdown report with all 6 sections |
| 📥 **Download** | Download buttons for PDF / Markdown / DOCX |

Click the **📥 Download** tab and download your report in any format!

---

## 📁 Terminal Summary — 3 Terminals Must Stay Open

| Terminal | Command | What It Runs |
|---|---|---|
| **Terminal 1** | `ollama serve` | Local LLM server (port 11434) |
| **Terminal 2** | `python -m uvicorn backend.api:app --port 8000` | FastAPI backend (port 8000) |
| **Terminal 3** | `python -m streamlit run frontend/app.py --server.port 8501` | Streamlit UI (port 8501) |

---

## 🛑 How to Stop Everything

To shut down, press `Ctrl + C` in each terminal:

1. **Terminal 3** → stops Streamlit
2. **Terminal 2** → stops FastAPI
3. **Terminal 1** → stops Ollama

---

## 🔄 How to Restart (After First Setup)

After initial setup, restarting is simple — just 3 commands in 3 terminals:

**Terminal 1:**
```powershell
ollama serve
```

**Terminal 2:**
```powershell
cd "C:\Users\Administrator\.gemini\antigravity\scratch\ResearchPilot-AI"
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

**Terminal 3:**
```powershell
cd "C:\Users\Administrator\.gemini\antigravity\scratch\ResearchPilot-AI"
python -m streamlit run frontend/app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

---

## 🐳 Alternative: Run with Docker (Optional)

If you prefer Docker instead of manual setup:

```powershell
cd "C:\Users\Administrator\.gemini\antigravity\scratch\ResearchPilot-AI"
docker-compose up --build
```

> ⚠️ Ollama must still run on the host machine (not in Docker).

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

---

## 🧪 Running Tests

To verify the code is working correctly:

```powershell
cd "C:\Users\Administrator\.gemini\antigravity\scratch\ResearchPilot-AI"
python -m pytest tests/ -v
```

---

## ❓ Troubleshooting

### "Backend not reachable" in sidebar
→ Make sure Terminal 2 is running `uvicorn`. Check http://localhost:8000/health

### "Ollama 🔴 Off" in sidebar
→ Make sure Terminal 1 is running `ollama serve`. Check http://localhost:11434

### "Model 🔴 N/A" in sidebar
→ Run `ollama pull llama3.1:8b` in a separate terminal and wait for download

### pip install fails
→ Try: `python -m pip install --upgrade pip` then retry

### "ModuleNotFoundError"
→ Make sure you're in the project directory and ran `pip install -r requirements.txt`

### Slow generation
→ Start with 2–3 papers. Use a machine with a GPU for faster inference.
→ Reduce chunk_size to 300 for faster embedding.

### Port already in use
→ Kill the process: `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F`
→ Or use a different port: `--port 8001`

---

## 📊 Useful URLs

| URL | What It Shows |
|---|---|
| http://localhost:8501 | Streamlit Dashboard (main UI) |
| http://localhost:8000/docs | FastAPI Swagger Documentation |
| http://localhost:8000/health | System health check (JSON) |
| http://localhost:8000/metrics | LLM call metrics (JSON) |
| http://localhost:8000/sessions | All past research sessions (JSON) |
| http://localhost:11434 | Ollama server (should show "Ollama is running") |
