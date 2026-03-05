"""
Celery application for background ingestion.
Set CELERY_BROKER_URL (e.g. redis://localhost:6379/0) to enable.
Run worker: celery -A celery_app worker -l info
"""

from celery import Celery
import config

app = Celery(
    "fusor_ai",
    broker=config.CELERY_BROKER_URL or "memory://",
    backend=config.CELERY_RESULT_BACKEND or None,
    include=["tasks.ingest_tasks"],
)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,  # 1 hour
)
