"""
config.py — Centralized configuration for ResearchPilot-AI.
All settings are loaded from environment variables (or .env file).
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama / LLM ──────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"

    # ── Embedding Model ───────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en"

    # ── RAG Defaults ──────────────────────────────────────────────────────────
    max_papers: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 5

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")

    # ── Backend ───────────────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    def pdfs_dir(self, session_id: str) -> Path:
        p = self.data_dir / "pdfs" / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def session_dir(self, session_id: str) -> Path:
        p = self.data_dir / "sessions" / session_id
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton instance
settings = Settings()

# Ensure required directories exist on import
for _dir in [settings.data_dir, settings.reports_dir, settings.logs_dir,
             settings.data_dir / "pdfs", settings.data_dir / "sessions"]:
    _dir.mkdir(parents=True, exist_ok=True)
