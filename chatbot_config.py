"""
Chatbot Configuration Service
Fetches chatbot configuration from Bubble.io and provides fallback system prompts
"""

import os
import json
import httpx
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

# Bubble.io API Configuration
BUBBLE_API_URL = os.getenv("BUBBLE_API_URL", "")
BUBBLE_API_TOKEN = os.getenv("BUBBLE_API_TOKEN", "")
BUBBLE_DATA_TYPE = os.getenv("BUBBLE_DATA_TYPE", "Chatbot")  # Your Bubble.io data type name

# Debug: Print environment variable status (without exposing token)
if BUBBLE_API_URL:
    print(f"Bubble.io API URL configured: {BUBBLE_API_URL[:50]}...")
else:
    print("⚠️ BUBBLE_API_URL not found in environment variables")
    
if BUBBLE_API_TOKEN:
    print(f"Bubble.io API Token configured: {'*' * min(len(BUBBLE_API_TOKEN), 20)}...")
else:
    print("⚠️ BUBBLE_API_TOKEN not found in environment variables")
    
print(f"Bubble.io Data Type: {BUBBLE_DATA_TYPE}")

# Cache for chatbot configs (simple in-memory cache)
_config_cache: Dict[str, Dict] = {}


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

CORE: Use context as single source of truth. If context lacks the answer, say so clearly. Never hallucinate. Never mention retrieval, chunks, embeddings, or system mechanics. Present answers naturally. Paraphrase, don't quote verbatim unless requested. Never fabricate details. If context conflicts, state the conflict clearly.

PROCESSING: Identify relevant parts, synthesize information, avoid verbosity. Mask sensitive data (passwords, private numbers) automatically.

ANSWERING: For direct questions, provide clear concise answers with relevant details. For explanations, give clean summaries. For procedures, construct steps using context + safe general knowledge. If information is missing, say "I don't have enough information in the provided documents to answer this directly" and suggest next steps.

SAFETY: Never invent dates, numbers, names, or claims. Refuse harmful/unethical requests politely. Mask sensitive personal data (e.g., "****1234") and explain it requires explicit permission.

AMBIGUITY: If context is vague, acknowledge it, give closest interpretation, ask if user wants to upload more documents.

RESTRICTIONS: Never mention context, chunks, documents, retrieval, confidence scores, citations, evidence, metadata, document IDs, or internal reasoning. No JSON unless requested."""
    
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
    cache_key = f"{user_id}_{bot_id}"
    
    # Check cache first
    if cache_key in _config_cache:
        return _config_cache[cache_key]
    
    # If no Bubble.io API configured, return None silently (it's optional now)
    if not BUBBLE_API_URL or not BUBBLE_API_TOKEN:
        # Don't print error - Bubble.io API is optional, config can come from request
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
                
                # Only log if we're debugging (set DEBUG_BUBBLE_API=true in .env)
                debug_mode = os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true"
                if debug_mode:
                    print(f"Trying Bubble.io API with fields: {user_field}, {bot_field}")
                    print(f"URL: {url}")
                    print(f"User ID: {user_id}, Bot ID: {bot_id}")
                
                response = await client.post(url, json=post_data, headers=headers)
                
                if debug_mode:
                    print(f"Response status: {response.status_code}")
                    print(f"Response text: {response.text[:500]}")  # First 500 chars
                
                if response.status_code == 200:
                    data = response.json()
                    debug_mode = os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true"
                    if debug_mode:
                        print(f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # Extract chatbot config from response
                    chatbot_data = None
                    
                    if isinstance(data, dict):
                        # Bubble.io v1.1 API format: {"response": {"results": [...]}}
                        if "response" in data and "results" in data["response"]:
                            results = data["response"]["results"]
                            if debug_mode:
                                print(f"Found {len(results)} results in response.results")
                            if results and len(results) > 0:
                                chatbot_data = results[0]
                                if debug_mode:
                                    print(f"Using first result. Keys: {list(chatbot_data.keys()) if isinstance(chatbot_data, dict) else 'Not a dict'}")
                        # Direct results array
                        elif "results" in data:
                            results = data["results"]
                            if debug_mode:
                                print(f"Found {len(results)} results in results")
                            if results and len(results) > 0:
                                chatbot_data = results[0]
                        # Direct object (single result)
                        else:
                            if debug_mode:
                                print("Using data as direct object")
                            chatbot_data = data
                    
                    if chatbot_data:
                        if debug_mode:
                            print(f"Chatbot data found! Keys: {list(chatbot_data.keys()) if isinstance(chatbot_data, dict) else 'Not a dict'}")
                        
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
                        
                        # Cache the config
                        _config_cache[cache_key] = config
                        debug_mode = os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true"
                        if debug_mode:
                            print(f"Successfully cached config for {cache_key}")
                        return config
                    else:
                        debug_mode = os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true"
                        if debug_mode:
                            print(f"No chatbot_data extracted. Trying next field name combination...")
                        continue
                elif response.status_code == 404:
                    # 404 is expected if record doesn't exist - don't log unless debugging
                    debug_mode = os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true"
                    if debug_mode:
                        print(f"404 Not Found - Record doesn't exist or field names are wrong")
                    continue
                else:
                    # Only log errors if debugging (401, 500, etc.)
                    debug_mode = os.getenv("DEBUG_BUBBLE_API", "false").lower() == "true"
                    if debug_mode:
                        print(f"Error {response.status_code}: {response.text[:500]}")
                    continue
            
            # If we get here, none of the field name combinations worked
            # Don't log - it's expected if Bubble.io API is not available
            return None
                
    except httpx.TimeoutException:
        print("Bubble.io API request timed out")
        return None
    except Exception as e:
        print(f"Error fetching chatbot config from Bubble.io: {e}")
        import traceback
        traceback.print_exc()
        return None


def clear_config_cache():
    """Clear the configuration cache"""
    global _config_cache
    _config_cache.clear()

