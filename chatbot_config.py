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

# Bubble.io API Configuration
BUBBLE_API_URL = os.getenv("BUBBLE_API_URL", "")
BUBBLE_API_TOKEN = os.getenv("BUBBLE_API_TOKEN", "")
BUBBLE_DATA_TYPE = os.getenv("BUBBLE_DATA_TYPE", "Chatbot")  # Your Bubble.io data type name

# Cache for chatbot configs: key -> (config_dict, timestamp)
CONFIG_CACHE_TTL_SECONDS = int(os.getenv("CONFIG_CACHE_TTL_SECONDS", "600"))  # 10 min
CONFIG_CACHE_MAX_SIZE = int(os.getenv("CONFIG_CACHE_MAX_SIZE", "1000"))
_config_cache: Dict[str, Tuple[Dict, float]] = {}


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
    Fetch chatbot configuration from Bubble.io API.
    Uses caching to avoid repeated API calls.
    
    Args:
        user_id: User identifier
        bot_id: Bot identifier
        
    Returns:
        Dictionary with chatbot configuration or None if not found
    """
    # #region agent log
    _debug_log("get_chatbot_config entry", {"user_id": user_id, "bot_id": bot_id, "has_url": bool(BUBBLE_API_URL), "has_token": bool(BUBBLE_API_TOKEN)}, "H1", "chatbot_config.py:entry")
    # #endregion
    cache_key = f"{user_id}_{bot_id}"
    now = time.time()
    # Check cache first (with TTL)
    if cache_key in _config_cache:
        cached_val, ts = _config_cache[cache_key]
        if now - ts < CONFIG_CACHE_TTL_SECONDS:
            _debug_log("cache hit, returning cached config", {"cache_key": cache_key}, "H3", "chatbot_config.py:cache_hit")
            return cached_val
        del _config_cache[cache_key]
    
    # If no Bubble.io API configured, return None silently (it's optional now)
    if not BUBBLE_API_URL or not BUBBLE_API_TOKEN:
        _debug_log("early return: missing BUBBLE_API_URL or BUBBLE_API_TOKEN", {"has_url": bool(BUBBLE_API_URL), "has_token": bool(BUBBLE_API_TOKEN)}, "H1", "chatbot_config.py:early_return")
        return None
    
    try:
        # Construct Bubble.io API request
        # Remove /obj if it's already in the URL
        base_url = BUBBLE_API_URL.rstrip('/obj').rstrip('/')
        url = f"{base_url}/obj/{BUBBLE_DATA_TYPE}"
        
        # Bubble.io requires POST with constraints in body
        # Try multiple possible field name formats
        possible_field_names = [
            ("User ID", "Bot ID"),           # With spaces
            ("user_id", "bot_id"),           # snake_case
            ("userId", "botId"),             # camelCase
            ("UserID", "BotID"),             # PascalCase
            ("User", "Bot"),                 # Short names
        ]
        
        headers = {
            "Authorization": f"Bearer {BUBBLE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try each field name combination
            for user_field, bot_field in possible_field_names:
                post_data = {
                    "constraints": [
                        {
                            "key": user_field,
                            "constraint_type": "equals",
                            "value": user_id
                        },
                        {
                            "key": bot_field,
                            "constraint_type": "equals",
                            "value": bot_id
                        }
                    ]
                }
                
                debug_mode = os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true"
                if debug_mode:
                    logger.debug("Trying Bubble.io API", extra={"fields": (user_field, bot_field), "url": url, "user_id": user_id, "bot_id": bot_id})
                response = await client.post(url, json=post_data, headers=headers)
                if debug_mode:
                    logger.debug("Bubble.io response", extra={"status": response.status_code, "text_preview": response.text[:500]})
                # #region agent log
                _debug_log("Bubble API response", {"status_code": response.status_code, "user_field": user_field, "bot_field": bot_field, "url": url}, "H2", "chatbot_config.py:after_post")
                # #endregion
                if response.status_code == 200:
                    data = response.json()
                    if os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true":
                        logger.debug("Bubble response data keys", extra={"keys": list(data.keys()) if isinstance(data, dict) else None})
                    
                    # Extract chatbot config from response
                    chatbot_data = None
                    results_count = 0
                    first_result_keys = None
                    
                    if isinstance(data, dict):
                        # Bubble.io v1.1 API format: {"response": {"results": [...]}}
                        if "response" in data and "results" in data["response"]:
                            results = data["response"]["results"]
                            results_count = len(results) if results else 0
                            if results and len(results) > 0:
                                chatbot_data = results[0]
                                first_result_keys = list(chatbot_data.keys()) if isinstance(chatbot_data, dict) else None
                        # Direct results array
                        elif "results" in data:
                            results = data["results"]
                            results_count = len(results) if results else 0
                            if results and len(results) > 0:
                                chatbot_data = results[0]
                                first_result_keys = list(chatbot_data.keys()) if isinstance(chatbot_data, dict) else None
                        else:
                            chatbot_data = data
                            first_result_keys = list(data.keys()) if isinstance(data, dict) else None
                    # #region agent log
                    _debug_log("Bubble 200 parsed", {"data_top_keys": list(data.keys()) if isinstance(data, dict) else None, "results_count": results_count, "first_result_keys": first_result_keys, "has_chatbot_data": bool(chatbot_data)}, "H3", "chatbot_config.py:parse_200")
                    # #endregion
                    if chatbot_data:
                        # Extract fields - try multiple possible field name formats
                        config = {
                            "chatbot_name": (
                                chatbot_data.get("Chatbot Name") or
                                chatbot_data.get("Name") or 
                                chatbot_data.get("chatbot_name") or
                                chatbot_data.get("ChatbotName") or
                                chatbot_data.get("name")
                            ),
                            "description": (
                                chatbot_data.get("Description") or 
                                chatbot_data.get("description")
                            ),
                            "industry": (
                                chatbot_data.get("Industry") or 
                                chatbot_data.get("industry")
                            ),
                            "color": (
                                chatbot_data.get("Color") or 
                                chatbot_data.get("color")
                            ),
                            "logo": (
                                chatbot_data.get("Logo") or 
                                chatbot_data.get("logo")
                            ),
                            "welcome_message": (
                                chatbot_data.get("Welcome Message") or 
                                chatbot_data.get("Welcome message") or
                                chatbot_data.get("welcome_message") or
                                chatbot_data.get("welcomeMessage")
                            ),
                            "knowledge_source": (
                                chatbot_data.get("Knowledge source") or 
                                chatbot_data.get("Knowledge Source") or
                                chatbot_data.get("knowledge_source") or
                                chatbot_data.get("knowledgeSource")
                            ),
                            "tone": (
                                chatbot_data.get("Tone") or 
                                chatbot_data.get("tone")
                            ),
                            "system_prompt": (
                                chatbot_data.get("System Prompt") or 
                                chatbot_data.get("System prompt") or
                                chatbot_data.get("system_prompt") or
                                chatbot_data.get("systemPrompt")
                            )
                        }
                        
                        # Cache the config (with max size eviction)
                        if len(_config_cache) >= CONFIG_CACHE_MAX_SIZE:
                            oldest_key = min(_config_cache, key=lambda k: _config_cache[k][1])
                            del _config_cache[oldest_key]
                        _config_cache[cache_key] = (config, time.time())
                        _debug_log("returning config from Bubble", {"config_keys": list(config.keys())}, None, "chatbot_config.py:return_config")
                        return config
                    else:
                        continue
                elif response.status_code == 404:
                    continue
                else:
                    if os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true":
                        logger.debug("Bubble API error", extra={"status": response.status_code, "text": response.text[:500]})
                    continue
            
            # If we get here, none of the field name combinations worked
            # #region agent log
            _debug_log("return None: no field combination returned results", {"user_id": user_id, "bot_id": bot_id}, "H3", "chatbot_config.py:no_combination")
            # #endregion
            return None
                
    except httpx.TimeoutException:
        _debug_log("Bubble timeout", {"error": "TimeoutException"}, "H4", "chatbot_config.py:timeout")
        logger.warning("Bubble.io API request timed out")
        return None
    except Exception as e:
        _debug_log("Bubble exception", {"error_type": type(e).__name__, "error_message": str(e)[:200]}, "H4", "chatbot_config.py:exception")
        logger.warning("Error fetching chatbot config from Bubble.io: %s", e, exc_info=True)
        return None


def clear_config_cache() -> None:
    """Clear the configuration cache."""
    global _config_cache
    _config_cache.clear()

