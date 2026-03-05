"""
Structured logging for the Fusor AI application.
Use request_id, user_id, bot_id in extra for traceability.
"""

import logging
import sys
from typing import Any

def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def log_extra(request_id: str = None, user_id: str = None, bot_id: str = None, **kwargs) -> dict:
    """Build extra dict for structured logging."""
    out = {}
    if request_id is not None:
        out["request_id"] = request_id
    if user_id is not None:
        out["user_id"] = user_id
    if bot_id is not None:
        out["bot_id"] = bot_id
    out.update(kwargs)
    return out
