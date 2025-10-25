import os
import re
from fastapi import FastAPI, Query, Request, UploadFile, File, Form, APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import config

# -------------------------
# Setup
# -------------------------
load_dotenv()
router = APIRouter()

app = FastAPI()

QDRANT_HOST = config.QDRANT_HOST
QDRANT_PORT = config.QDRANT_PORT

# Connect to Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Embeddings (same as ingestion)
EMBEDDING_MODEL = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Chat LLM for final response
CHAT_MODEL = ChatOpenAI(
    model=config.CHAT_MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY")
)


# -------------------------
# Request Schema
# -------------------------
class QueryRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    bot_id: str = "default_bot"
    top_k: int = 3


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
def search_and_answer(query: str, top_k: int = 3, user_id: str = "default_user", bot_id: str = "default_bot"):
    # 1. Preprocess query
    processed_query = preprocess_query(query)
    
    # 2. Embed query
    query_embedding = EMBEDDING_MODEL.embed_query(processed_query)
    
    # 3. Setup collection and namespace
    collection_name = f"{config.COLLECTION_PREFIX}{user_id}"
    namespace = f"{config.NAMESPACE_PREFIX}{bot_id}"
    
    # 4. Search in Qdrant (no score threshold here, we'll filter later)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=config.DEFAULT_TOP_K,  # Retrieve more initially
        query_filter=Filter(
            must=[FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace))]
        ) if namespace else None
    )

    if not search_result:
        return "No relevant context found in knowledge base."

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
    
    if not relevant_chunks:
        return "No relevant context found in knowledge base."

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
    
    # 7. Build context with citations
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[{i}] {chunk['content']}")
    
    context = "\n\n".join(context_parts)
    
    # 8. Build citations
    citations = [f"[{i}] {chunk['source']}" for i, chunk in enumerate(retrieved_chunks, 1)]
    
    # 9. Ask LLM with enhanced context
    system_prompt = """You are a helpful assistant. Use the provided context to answer questions accurately. 
When referencing information, use the citation numbers [1], [2], etc. provided in the context.
If the context doesn't contain enough information to answer the question, say so clearly."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = CHAT_MODEL.invoke(messages)
    
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
    result = search_and_answer(
        query=request.query,
        top_k=request.top_k,
        user_id=request.user_id,
        bot_id=request.bot_id
    )
    
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


