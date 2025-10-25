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
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# ================================
# Embedding Configuration
# ================================
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_DISTANCE_METRIC = "COSINE"  # COSINE, DOT, EUCLIDEAN

# ================================
# Chunking Configuration
# ================================
CHUNK_SIZE = 1000  # Increased from 400 for better context preservation
CHUNK_OVERLAP = 150  # Increased from 50 for better boundary handling
CHUNK_METHOD = "recursive"  # recursive, semantic, fixed

# ================================
# Retrieval Configuration
# ================================
DEFAULT_TOP_K = 5  # Retrieve more initially
MAX_TOP_K = 5  # Maximum chunks to use
MIN_TOP_K = 1  # Minimum chunks to use
SCORE_THRESHOLD = 0.3  # Filter out low-relevance chunks (lowered for better recall)

# ================================
# LLM Configuration
# ================================
CHAT_MODEL_NAME = "gpt-4o-mini"
MAX_CONTEXT_TOKENS = 4000  # Leave room for system prompt and response

# ================================
# Multi-Tenancy Configuration
# ================================
COLLECTION_PREFIX = "user_"  # Collections will be named user_{user_id}
NAMESPACE_PREFIX = "bot_"    # Namespaces will be named bot_{bot_id}

# ================================
# Performance Configuration
# ================================
BATCH_SIZE = 100  # For bulk operations
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# ================================
# Query Preprocessing
# ================================
ENABLE_QUERY_PREPROCESSING = True
QUERY_EXPANSION = True  # Expand acronyms, fix common typos
NORMALIZE_QUERY = True  # Lowercase, remove extra spaces
