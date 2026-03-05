"""
Simple in-memory metrics for request counts and latencies.
Suitable for single-process deployment; use Prometheus/StatsD for multi-process.
"""

import time
from typing import Dict, Any

_metrics: Dict[str, Any] = {
    "ingest_requests_total": 0,
    "ingest_errors_total": 0,
    "query_requests_total": 0,
    "query_errors_total": 0,
}


def increment(name: str, value: int = 1) -> None:
    if name in _metrics and isinstance(_metrics[name], (int, float)):
        _metrics[name] += value


def get_metrics() -> Dict[str, Any]:
    return dict(_metrics)
