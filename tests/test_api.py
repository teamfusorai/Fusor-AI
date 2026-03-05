"""API integration tests: health, ready, metrics (no Qdrant/OpenAI required for health)."""

import pytest
from fastapi.testclient import TestClient

# Import app after env is loaded so config is set
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "ingest_requests_total" in data
    assert "query_requests_total" in data
    assert isinstance(data["ingest_requests_total"], (int, float))
    assert isinstance(data["query_requests_total"], (int, float))


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data
    assert "health" in data["endpoints"]
    assert "ready" in data["endpoints"]


def test_ready_depends_on_qdrant():
    """Ready may be 200 (Qdrant up) or 503 (Qdrant down). We only assert structure."""
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    data = r.json()
    if r.status_code == 200:
        assert data.get("status") == "ready"
    else:
        assert "error" in data
