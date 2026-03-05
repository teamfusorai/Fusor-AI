"""
Celery task for background document/URL ingestion.
"""

import os
from celery_app import app
from data_ingestion import run_ingest_sync
from utils.logging_config import get_logger

logger = get_logger(__name__)


@app.task(bind=True, name="tasks.ingest_tasks.run_ingest")
def run_ingest(
    self,
    source_type: str,
    path_or_url: str,
    source_name: str,
    user_id: str,
    bot_id: str,
):
    """
    Run ingestion in the worker.
    source_type: "file" | "url"
    path_or_url: path to uploaded file or URL
    """
    try:
        result = run_ingest_sync(
            source_type=source_type,
            path_or_url=path_or_url,
            source_name=source_name,
            user_id=user_id or "default_user",
            bot_id=bot_id or "default_bot",
        )
        if source_type == "file" and path_or_url and os.path.isfile(path_or_url):
            try:
                os.unlink(path_or_url)
            except OSError:
                pass
        return result
    except Exception as e:
        logger.exception("Ingest task failed")
        if source_type == "file" and path_or_url and os.path.isfile(path_or_url):
            try:
                os.unlink(path_or_url)
            except OSError:
                pass
        raise
