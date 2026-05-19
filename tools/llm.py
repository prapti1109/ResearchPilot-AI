"""
tools/llm.py — Centralized LLM wrapper for ResearchPilot-AI.

All agents talk to Ollama through this module only. Handles:
  - Connection health checks
  - Retry logic with exponential back-off
  - Token / latency metrics logging
  - Streaming support
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Generator, Optional

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings


# ── Metrics helpers ────────────────────────────────────────────────────────────

_METRICS_FILE = settings.logs_dir / "metrics.jsonl"


def _log_metric(agent: str, prompt_tokens: int, response_tokens: int,
                latency_s: float, model: str) -> None:
    record = {
        "ts": time.time(),
        "agent": agent,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "latency_s": round(latency_s, 3),
    }
    with open(_METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Main LLM class ─────────────────────────────────────────────────────────────

class LocalLLM:
    """
    Thin wrapper around the Ollama /api/generate endpoint.

    Usage::

        llm = LocalLLM()
        response = llm.generate("Summarize this text: ...", agent="summarizer")
    """

    def __init__(
        self,
        model: str = settings.llm_model,
        base_url: str = settings.ollama_base_url,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ── Health ─────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model is loaded."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            # Accept prefix match so "llama3.1:8b" matches "llama3.1:8b-instruct-q4"
            return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False

    def health_check(self) -> dict:
        """Return a dict with availability details for the /health endpoint."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            available = any(self.model.split(":")[0] in m for m in models)
            return {
                "ollama_reachable": True,
                "model_available": available,
                "model": self.model,
                "all_models": models,
            }
        except Exception as exc:
            return {
                "ollama_reachable": False,
                "model_available": False,
                "model": self.model,
                "error": str(exc),
            }

    # ── Core generation ────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        agent: str = "unknown",
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a single completion (non-streaming).

        Args:
            prompt: User / instruction prompt.
            agent: Agent name for metrics logging.
            system: Optional system message.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        t0 = time.perf_counter()
        try:
            r = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            response_text: str = data.get("response", "")
            latency = time.perf_counter() - t0

            # Log metrics
            _log_metric(
                agent=agent,
                prompt_tokens=data.get("prompt_eval_count", len(prompt.split())),
                response_tokens=data.get("eval_count", len(response_text.split())),
                latency_s=latency,
                model=self.model,
            )
            logger.debug(f"[{agent}] LLM response in {latency:.2f}s "
                         f"({data.get('eval_count', '?')} tokens)")
            return response_text.strip()
        except httpx.HTTPStatusError as exc:
            logger.error(f"[{agent}] Ollama HTTP error: {exc}")
            raise

    def stream(
        self,
        prompt: str,
        agent: str = "unknown",
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama one by one.

        Yields individual token strings. Useful for UI streaming.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        t0 = time.perf_counter()
        token_count = 0
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        token_count += 1
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

        latency = time.perf_counter() - t0
        _log_metric(agent, len(prompt.split()), token_count, latency, self.model)

    def chat(
        self,
        messages: list[dict],
        agent: str = "unknown",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Chat-style generation (list of {role, content} dicts).
        Falls back to converting messages → single prompt.
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"[{role}]: {content}")
        prompt = "\n".join(prompt_parts) + "\n[ASSISTANT]:"
        return self.generate(prompt, agent=agent, temperature=temperature,
                             max_tokens=max_tokens)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LocalLLM":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Module-level singleton ─────────────────────────────────────────────────────

_llm_instance: Optional[LocalLLM] = None


def get_llm() -> LocalLLM:
    """Return the shared LLM singleton."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance
