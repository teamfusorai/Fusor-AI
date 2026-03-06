import os
import re
from typing import Optional
from fastapi import FastAPI, Query, Request, UploadFile, File, Form, APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import config
import chatbot_config
from utils.logging_config import get_logger
from utils.metrics import increment
from utils.conversation_memory import get_history, add_turn

# -------------------------
# Setup
# -------------------------
load_dotenv()
router = APIRouter()
logger = get_logger(__name__)

app = FastAPI()

QDRANT_HOST = config.QDRANT_HOST
QDRANT_PORT = config.QDRANT_PORT

# Connect to Qdrant with connection pooling
qdrant_client = QdrantClient(
    host=QDRANT_HOST, 
    port=QDRANT_PORT,
    timeout=config.TIMEOUT_SECONDS,
    prefer_grpc=False  # HTTP is faster for most use cases
)

# Embeddings (same as ingestion) with timeout configuration
EMBEDDING_MODEL = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=config.TIMEOUT_SECONDS,
    max_retries=config.MAX_RETRIES
)

# Chat LLM for final response with timeout configuration
CHAT_MODEL = ChatOpenAI(
    model=config.CHAT_MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=config.TIMEOUT_SECONDS,
    max_retries=config.MAX_RETRIES,
    temperature=0.1  # Lower temperature for faster, more deterministic responses
)


# -------------------------
# Request Schema
# -------------------------
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    bot_id: str = "default_bot"
    top_k: int = 3
    # Optional chatbot configuration - if not provided, will be fetched from Bubble.io
    system_prompt: Optional[str] = None
    tone: Optional[str] = None
    industry: Optional[str] = None
    chatbot_name: Optional[str] = None
    description: Optional[str] = None


# -------------------------
# Query Preprocessing
# -------------------------
def preprocess_query(query: str) -> str:
    """Preprocess query for better retrieval"""
    if not config.ENABLE_QUERY_PREPROCESSING:
        return query
    
    processed_query = query
    
    if config.NORMALIZE_QUERY:
        # Remove extra spaces and normalize
        processed_query = re.sub(r'\s+', ' ', processed_query.strip().lower())
    
    if config.QUERY_EXPANSION:
        # Common acronym expansions
        expansions = {
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'db': 'database',
            'sql': 'structured query language',
            'http': 'hypertext transfer protocol',
            'https': 'hypertext transfer protocol secure',
            'json': 'javascript object notation',
            'xml': 'extensible markup language',
            'rest': 'representational state transfer',
            'soap': 'simple object access protocol'
        }
        
        for acronym, expansion in expansions.items():
            # Replace standalone acronyms (word boundaries)
            pattern = r'\b' + acronym + r'\b'
            processed_query = re.sub(pattern, f"{acronym} ({expansion})", processed_query, flags=re.IGNORECASE)
    
    return processed_query

# -------------------------
# Core Search + Answer
# -------------------------
async def search_and_answer(
    query: str,
    top_k: int = 3,
    user_id: str = "default_user",
    bot_id: str = "default_bot",
    system_prompt: Optional[str] = None,
    tone: Optional[str] = None,
    industry: Optional[str] = None,
    chatbot_name: Optional[str] = None,
    description: Optional[str] = None,
    chat_history: Optional[list] = None,
):
    # 1. Preprocess query
    processed_query = preprocess_query(query)
    
    # 2. Embed query (async for non-blocking operation)
    query_embedding = await EMBEDDING_MODEL.aembed_query(processed_query)
    
    # 3. Setup collection and namespace
    collection_name = f"{config.COLLECTION_PREFIX}{user_id}"
    namespace = f"{config.NAMESPACE_PREFIX}{bot_id}"
    
    # 4. Search in Qdrant (retrieve at least top_k, up to DEFAULT_TOP_K for filtering)
    search_limit = max(config.DEFAULT_TOP_K, min(top_k, config.MAX_TOP_K))
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=search_limit,
        query_filter=Filter(
            must=[FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace))]
        ) if namespace else None
    )

    if not search_result:
        search_result = []

    # 5. Filter by score threshold and adaptive top_k
    relevant_chunks = []
    for hit in search_result:
        if hit.score >= config.SCORE_THRESHOLD:
            relevant_chunks.append(hit)
        
        # Stop if we have enough high-quality chunks
        if len(relevant_chunks) >= min(top_k, config.MAX_TOP_K):
            break
    
    # Ensure we have at least minimum chunks if available
    if len(relevant_chunks) < config.MIN_TOP_K and len(search_result) >= config.MIN_TOP_K:
        relevant_chunks = search_result[:config.MIN_TOP_K]
    
    # If no relevant chunks are found after filtering
    if not relevant_chunks:
        # We allow the LLM to still process the query with empty context to handle greetings/general chatter
        pass

    # 6. Collect retrieved chunks with metadata
    retrieved_chunks = []
    for hit in relevant_chunks:
        chunk_data = {
            "content": hit.payload.get("page_content", ""),
            "source": hit.payload.get("metadata", {}).get("source_name", "unknown"),
            "chunk_index": hit.payload.get("metadata", {}).get("chunk_index", 0),
            "score": hit.score
        }
        retrieved_chunks.append(chunk_data)
    
    #context + citations
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[{i}] {chunk['content']}")
    
    context = "\n\n".join(context_parts)
    
    citations = [f"[{i}] {chunk['source']}" for i, chunk in enumerate(retrieved_chunks, 1)]
    
    # Build smart system prompt using chatbot configuration
    final_system_prompt = chatbot_config.build_smart_system_prompt(
        chatbot_name=chatbot_name,
        description=description,
        industry=industry,
        tone=tone,
        custom_system_prompt=system_prompt
    )

    # Short-term memory: optional chunked conversation history (last N turns)
    current_user_message = f"{context}\n\nQuestion: {query}"
    messages = [{"role": "system", "content": final_system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": current_user_message})

    response = await CHAT_MODEL.ainvoke(messages)
    
    # 10. Return response with metadata
    return {
        "answer": response.content,
        "citations": citations,
        "chunks_used": len(relevant_chunks),
        "confidence_scores": [chunk['score'] for chunk in retrieved_chunks]
    }


# -------------------------
# API Endpoint
# -------------------------
@router.post("/query")
async def query_endpoint(request: QueryRequest):
    increment("query_requests_total")
    # Use config from request if provided, otherwise try to fetch from Bubble.io
    # If any config field is provided, use request values and skip Bubble.io fetch
    has_config = any([
        request.system_prompt,
        request.tone,
        request.industry,
        request.chatbot_name,
        request.description
    ])
    
    if has_config:
        # Use request values directly (Bubble.io sends config in request)
        system_prompt = request.system_prompt
        tone = request.tone
        industry = request.industry
        chatbot_name = request.chatbot_name
        description = request.description
    else:
        # Try to fetch from Bubble.io (optional, will work if API is available)
        chatbot_config_data = await chatbot_config.get_chatbot_config(request.user_id, request.bot_id)
        
        if chatbot_config_data:
            system_prompt = chatbot_config_data.get("system_prompt")
            tone = chatbot_config_data.get("tone")
            industry = chatbot_config_data.get("industry")
            chatbot_name = chatbot_config_data.get("chatbot_name")
            description = chatbot_config_data.get("description")
        else:
            # No config available, use None (will use default prompts)
            system_prompt = None
            tone = None
            industry = None
            chatbot_name = None
            description = None

    chat_history = get_history(request.user_id, request.bot_id)
    result = await search_and_answer(
        query=request.query,
        top_k=request.top_k,
        user_id=request.user_id,
        bot_id=request.bot_id,
        system_prompt=system_prompt,
        tone=tone,
        industry=industry,
        chatbot_name=chatbot_name,
        description=description,
        chat_history=chat_history,
    )

    if isinstance(result, dict):
        add_turn(request.user_id, request.bot_id, request.query, result["answer"])

    # Handle both old string response and new dict response
    if isinstance(result, dict):
        return {
            "query": request.query,
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "chunks_used": result.get("chunks_used", 0),
            "confidence_scores": result.get("confidence_scores", []),
            "user_id": request.user_id,
            "bot_id": request.bot_id,
            "top_k": request.top_k
        }
    else:
        # Fallback for old string response format
        return {
            "query": request.query,
            "answer": result,
            "citations": [],
            "chunks_used": 0,
            "confidence_scores": [],
            "user_id": request.user_id,
            "bot_id": request.bot_id,
            "top_k": request.top_k
        }


