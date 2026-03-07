"""
Chatbot Configuration Service
Fetches chatbot configuration from Bubble.io and provides fallback system prompts
"""

import os
import time
import json
import httpx
from typing import Optional, Dict, Tuple, Any
from dotenv import load_dotenv
from utils.logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG_LOG_PATH = os.path.join(_project_root, ".cursor", "debug.log")
DEBUG_LOG_FALLBACK = os.path.join(_project_root, "debug_chatbot_config.ndjson")

def _debug_log(message: str, data: Optional[Dict[str, Any]] = None, hypothesis_id: Optional[str] = None, location: str = ""):
    payload = {"message": message, "timestamp": int(time.time() * 1000), "location": location or "chatbot_config.get_chatbot_config"}
    if data is not None:
        payload["data"] = data
    if hypothesis_id:
        payload["hypothesisId"] = hypothesis_id
    line = json.dumps(payload, default=str) + "\n"
    for path in (DEBUG_LOG_PATH, DEBUG_LOG_FALLBACK):
        try:
            if path == DEBUG_LOG_PATH:
                log_dir = os.path.dirname(path)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            break
        except Exception as e:
            if path == DEBUG_LOG_PATH:
                logger.warning("_debug_log primary failed: path=%s err=%s", path, e)
            continue

import database

# Obsolete: We no longer need Bubble API Configs or caching since we're using SQLite
BUBBLE_API_URL = os.getenv("BUBBLE_API_URL", "")
BUBBLE_API_TOKEN = os.getenv("BUBBLE_API_TOKEN", "")
BUBBLE_DATA_TYPE = os.getenv("BUBBLE_DATA_TYPE", "Chatbot")


def build_smart_system_prompt(
    chatbot_name: Optional[str] = None,
    description: Optional[str] = None,
    industry: Optional[str] = None,
    tone: Optional[str] = None,
    custom_system_prompt: Optional[str] = None
) -> str:
    """
    Build a smart system prompt using all available chatbot details.
    If custom_system_prompt is provided, it will be used as base and enhanced with other details.
    Otherwise, a comprehensive prompt is built from scratch.
    """
    
    # Base prompt structure
    base_prompt_parts = []
    
    # Start with custom prompt if provided, otherwise use default base
    if custom_system_prompt:
        base_prompt = custom_system_prompt
    else:
        base_prompt = """You are a retrieval-augmented assistant. Answer queries using ONLY the retrieved context provided. Follow these rules:

CORE: Use context as single source of truth for factual questions regarding the documents. For general conversational greetings (e.g. "hi", "how are you", "what can you do") or common small-talk, respond politely and naturally using safe general knowledge. Do not mention retrieval, chunks, embeddings, or system mechanics. Present answers naturally. Paraphrase, don't quote verbatim unless requested. Never fabricate details. If context conflicts, state the conflict clearly.

AMBIGUITY: If context is vague, acknowledge it, give closest interpretation,
"""
    
    # Add chatbot identity if name is provided
    if chatbot_name:
        base_prompt += f"\n\nCHATBOT IDENTITY: You are {chatbot_name}."
    
    # Add description context
    if description:
        base_prompt += f"\n\nCHATBOT PURPOSE: {description}"
    
    # Add industry context
    if industry:
        base_prompt += f"\n\nINDUSTRY CONTEXT: This chatbot serves the {industry} industry. When answering questions, consider industry-specific terminology, standards, and best practices relevant to {industry}."
    
    # Add tone instruction
    if tone:
        tone_instructions = {
            "professional": "Maintain a formal, business-appropriate tone. Use professional language and avoid casual expressions.",
            "casual": "Use a friendly, conversational tone. Be approachable and use everyday language.",
            "friendly": "Be warm, welcoming, and personable. Use a friendly tone that makes users feel comfortable.",
            "technical": "Use precise technical terminology. Be detailed and accurate in technical explanations.",
            "formal": "Use formal language and structure. Maintain a respectful, official tone.",
            "conversational": "Engage naturally as if in a conversation. Be personable but informative."
        }
        tone_instruction = tone_instructions.get(tone.lower(), f"Use a {tone} tone.")
        base_prompt += f"\n\nTONE: {tone_instruction}"
    else:
        base_prompt += "\n\nTONE: Use a clear, practical, helpful, and structured tone."
    
    # Final instruction
    base_prompt += "\n\nYour answers come ONLY from: retrieved context, user instructions, safe general knowledge that doesn't contradict context. Never hallucinate or reveal retrieval system workings."
    
    return base_prompt


async def get_chatbot_config(user_id: str, bot_id: str) -> Optional[Dict]:
    """
    Fetch chatbot configuration from the local SQLite Database.
    
    Args:
        user_id: User identifier
        bot_id: Bot identifier
        
    Returns:
        Dictionary with chatbot configuration or None if not found
    """
    return database.get_chatbot_config_from_db(user_id, bot_id)


def clear_config_cache() -> None:
    """Clear the configuration cache (Obsolete)."""
    pass

