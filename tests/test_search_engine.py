"""Unit tests for search_engine query preprocessing."""

import pytest


@pytest.fixture(autouse=True)
def _query_preprocessing_on(monkeypatch):
    monkeypatch.setattr("search_engine.config.ENABLE_QUERY_PREPROCESSING", True)
    monkeypatch.setattr("search_engine.config.NORMALIZE_QUERY", True)
    monkeypatch.setattr("search_engine.config.QUERY_EXPANSION", True)


def test_preprocess_query_normalize():
    from search_engine import preprocess_query
    assert preprocess_query("  What   is   API  ") == "what is api (application programming interface)"
    assert preprocess_query("HTTP and HTTPS") != ""
    assert "http" in preprocess_query("HTTP and HTTPS").lower()


def test_preprocess_query_expansion():
    from search_engine import preprocess_query
    out = preprocess_query("REST API")
    assert "rest" in out.lower()
    assert "representational" in out.lower() or "api" in out.lower()
    out2 = preprocess_query("sql database")
    assert "sql" in out2.lower()
    assert "structured" in out2.lower() or "database" in out2.lower()


def test_preprocess_query_passthrough_when_disabled(monkeypatch):
    monkeypatch.setattr("search_engine.config.ENABLE_QUERY_PREPROCESSING", False)
    from search_engine import preprocess_query
    q = "  What   is   API  "
    assert preprocess_query(q) == q
