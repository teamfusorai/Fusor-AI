import getpass
import os
import uvicorn
from fastapi import FastAPI
from qdrant_client import QdrantClient
import config
import data_ingestion
import search_engine

app = FastAPI()
app.include_router(data_ingestion.router)
app.include_router(search_engine.router)

# Initialize Qdrant client for knowledge base listing
qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

@app.get("/knowledge-bases")
async def list_knowledge_bases():
    """List all available knowledge bases"""
    try:
        collections = qdrant_client.get_collections()
        knowledge_bases = []
        
        for collection in collections.collections:
            collection_name = collection.name
            
            # Check if it's a user collection (starts with user_)
            if collection_name.startswith(config.COLLECTION_PREFIX):
                user_id = collection_name.replace(config.COLLECTION_PREFIX, "")
                
                # Get collection info to find namespaces (bots)
                try:
                    collection_info = qdrant_client.get_collection(collection_name)
                    points_count = collection_info.points_count
                    
                    if points_count > 0:
                        # Get sample points to extract bot namespaces
                        sample_points = qdrant_client.scroll(
                            collection_name=collection_name,
                            limit=100,
                            with_payload=True
                        )[0]
                        
                        # Extract unique bot namespaces
                        bot_namespaces = set()
                        for point in sample_points:
                            if point.payload and "metadata" in point.payload:
                                metadata = point.payload["metadata"]
                                if "namespace" in metadata:
                                    namespace = metadata["namespace"]
                                    if namespace.startswith(config.NAMESPACE_PREFIX):
                                        bot_id = namespace.replace(config.NAMESPACE_PREFIX, "")
                                        bot_namespaces.add(bot_id)
                        
                        # Create knowledge base entries for each bot
                        for bot_id in bot_namespaces:
                            knowledge_bases.append({
                                "user_id": user_id,
                                "bot_id": bot_id,
                                "collection_name": collection_name,
                                "points_count": points_count
                            })
                
                except Exception as e:
                    print(f"Error processing collection {collection_name}: {e}")
                    continue
        
        return knowledge_bases
        
    except Exception as e:
        return {"error": f"Failed to list knowledge bases: {str(e)}"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Tenant Chatbot Platform API",
        "version": "1.0.0",
        "endpoints": {
            "ingest": "/ingest",
            "query": "/query", 
            "knowledge-bases": "/knowledge-bases"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
