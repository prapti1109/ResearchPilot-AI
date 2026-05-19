"""
tests/test_api.py — Integration tests for the FastAPI backend.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from workflows.states import WorkflowState, WorkflowStatus


@pytest.fixture
def client():
    from backend.api import app
    return TestClient(app)


@pytest.fixture
def mock_health_ok():
    with patch("backend.api.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.health_check.return_value = {
            "ollama_reachable": True,
            "model_available": True,
            "model": "llama3.1:8b",
            "all_models": ["llama3.1:8b"],
        }
        mock_get_llm.return_value = mock_llm
        yield mock_get_llm


class TestHealthEndpoint:
    def test_health_returns_200(self, client, mock_health_ok):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_contains_status(self, client, mock_health_ok):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "ollama" in data


class TestResearchStart:
    @patch("backend.api._run_workflow_sync")
    def test_start_returns_session_id(self, mock_run, client):
        r = client.post("/research/start", json={
            "topic": "AI in Healthcare",
            "max_papers": 3,
        })
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["status"] == "running"

    @patch("backend.api._run_workflow_sync")
    def test_start_uses_custom_session_id(self, mock_run, client):
        r = client.post("/research/start", json={
            "topic": "Test topic",
            "session_id": "test123",
        })
        assert r.status_code == 200
        assert r.json()["session_id"] == "test123"


class TestStatusEndpoint:
    def test_status_404_for_unknown_session(self, client):
        with patch("backend.api.load_session_state", return_value=None):
            r = client.get("/research/nonexistent/status")
            assert r.status_code == 404

    def test_status_returns_summary(self, client):
        state = WorkflowState(
            session_id="abc123",
            topic="Test",
            status=WorkflowStatus.RUNNING,
        )
        with patch("backend.api._session_states", {"abc123": state}):
            with patch("backend.api.load_session_state", return_value=None):
                r = client.get("/research/abc123/status")
                assert r.status_code == 200
                assert r.json()["session_id"] == "abc123"


class TestSessionsEndpoint:
    def test_sessions_returns_list(self, client):
        with patch("backend.api.list_sessions", return_value=[]):
            r = client.get("/sessions")
            assert r.status_code == 200
            assert isinstance(r.json(), list)


class TestMetricsEndpoint:
    def test_metrics_returns_dict(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "total_calls" in r.json()


class TestApprovalEndpoint:
    def test_approve_updates_state(self, client):
        state = WorkflowState(
            session_id="sess1",
            topic="Test",
            status=WorkflowStatus.AWAITING_APPROVAL,
        )
        with patch("backend.api._session_states", {"sess1": state}):
            with patch("backend.api.load_session_state", return_value=None):
                with patch("backend.api.save_session_state"):
                    r = client.post("/research/sess1/approve", json={
                        "approved_paper_ids": ["1234.5678"]
                    })
                    assert r.status_code == 200
                    assert r.json()["status"] == "approved"
