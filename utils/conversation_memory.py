"""
Short-term contextual memory for the RAG pipeline.
Keeps a bounded window of recent user/assistant turns per (user_id, bot_id).
"""

from typing import List, Dict
import threading
import config

# Key: (user_id, bot_id) -> list of {"role": "user"|"assistant", "content": str}
_store: Dict[tuple, List[Dict[str, str]]] = {}
_lock = threading.Lock()


def _key(user_id: str, bot_id: str) -> tuple:
    return (user_id or "default_user", bot_id or "default_bot")


def get_history(user_id: str, bot_id: str) -> List[Dict[str, str]]:
    """Return the last N conversation turns as message dicts for the LLM."""
    with _lock:
        key = _key(user_id, bot_id)
        if key not in _store:
            return []
        return list(_store[key])


def add_turn(user_id: str, bot_id: str, user_content: str, assistant_content: str) -> None:
    """Append one user/assistant turn and trim to MAX_CONVERSATION_HISTORY_TURNS."""
    with _lock:
        key = _key(user_id, bot_id)
        if key not in _store:
            _store[key] = []
        lst = _store[key]
        lst.append({"role": "user", "content": user_content})
        lst.append({"role": "assistant", "content": assistant_content})
        max_messages = config.MAX_CONVERSATION_HISTORY_TURNS * 2  # each turn = user + assistant
        if len(lst) > max_messages:
            _store[key] = lst[-max_messages:]


def clear_history(user_id: str, bot_id: str) -> None:
    """Clear conversation history for a user/bot (e.g. on new session)."""
    with _lock:
        key = _key(user_id, bot_id)
        if key in _store:
            del _store[key]
