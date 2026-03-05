"""
RAG Pipeline Configuration
Centralized configuration for multi-tenant chatbot platform
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ================================
# Qdrant Configuration
# ================================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
if isinstance(QDRANT_PORT, str):
    QDRANT_PORT = int(QDRANT_PORT)

# ================================
# Embedding Configuration
# ================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
EMBEDDING_DISTANCE_METRIC = os.getenv("EMBEDDING_DISTANCE_METRIC", "COSINE")  # COSINE, DOT, EUCLIDEAN

# ================================
# Chunking Configuration
# ================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CHUNK_METHOD = os.getenv("CHUNK_METHOD", "recursive")  # recursive, semantic, fixed

# ================================
# Retrieval Configuration
# ================================
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "10"))
MIN_TOP_K = int(os.getenv("MIN_TOP_K", "1"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

# ================================
# LLM Configuration
# ================================
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))

# ================================
# Multi-Tenancy Configuration
# ================================
COLLECTION_PREFIX = "user_"  # Collections will be named user_{user_id}
NAMESPACE_PREFIX = "bot_"    # Namespaces will be named bot_{bot_id}

# ================================
# Performance Configuration
# ================================
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))

# ================================
# Ingestion Limits
# ================================
MAX_INGEST_FILE_BYTES = int(os.getenv("MAX_INGEST_FILE_BYTES", "20971520"))  # 20 MB default
INGEST_EXTRACTION_TIMEOUT = int(os.getenv("INGEST_EXTRACTION_TIMEOUT", "120"))  # seconds

# ================================
# Short-term conversation memory (chunked window)
# ================================
MAX_CONVERSATION_HISTORY_TURNS = int(os.getenv("MAX_CONVERSATION_HISTORY_TURNS", "5"))  # Last N user+assistant pairs

# ================================
# Query Preprocessing
# ================================
ENABLE_QUERY_PREPROCESSING = os.getenv("ENABLE_QUERY_PREPROCESSING", "true").lower() == "true"
QUERY_EXPANSION = os.getenv("QUERY_EXPANSION", "true").lower() == "true"
NORMALIZE_QUERY = os.getenv("NORMALIZE_QUERY", "true").lower() == "true"

# ================================
# Celery / Background Jobs
# ================================
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "")  # e.g. redis://localhost:6379/0; empty = no background jobs
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "")  # e.g. redis://localhost:6379/0; optional
INGEST_UPLOAD_DIR = os.getenv("INGEST_UPLOAD_DIR", ".ingest_uploads")  # Dir for file payloads before worker picks up

# ================================
# Deployment Configuration
# ================================
BUBBLE_APP_URL = os.getenv("BUBBLE_APP_URL", "https://fyp-fusor-ai.bubbleapps.io/version-test/qr")  # Base URL for Bubble.io app

# Public base URL for this API (used in embed snippet and widget). Must be reachable from end-users' browsers.
# Examples: https://api.yourdomain.com, https://xxxx.ngrok-free.app (no trailing slash).
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", "").rstrip("/")  # Empty = widget will use script origin