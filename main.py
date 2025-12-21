import getpass
import os
import uvicorn
import uuid
import base64
from typing import Optional
from io import BytesIO
from urllib.parse import quote
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from qdrant_client import QdrantClient
import qrcode
import json
import config
import data_ingestion
import search_engine
import chatbot_config

app = FastAPI(
    title="Multi-Tenant Chatbot Platform API",
    description="API for document ingestion and chatbot querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                                "chunks_count": points_count,
                                "created_at": collection_info.status  # Use status as created_at placeholder
                            })
                
                except Exception as e:
                    print(f"Error processing collection {collection_name}: {e}")
                    continue
        
        return knowledge_bases
        
    except Exception as e:
        return {"error": f"Failed to list knowledge bases: {str(e)}"}

@app.get("/knowledge-bases/{user_id}/{bot_id}/stats")
async def get_knowledge_base_stats(user_id: str, bot_id: str):
    """Get statistics for a specific knowledge base"""
    try:
        collection_name = f"{config.COLLECTION_PREFIX}{user_id}"
        namespace = f"{config.NAMESPACE_PREFIX}{bot_id}"
        
        if not qdrant_client.collection_exists(collection_name):
            return {"error": "Knowledge base not found"}
        
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        
        # Count points in specific namespace
        namespace_points = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "metadata.namespace",
                        "match": {"value": namespace}
                    }
                ]
            },
            limit=10000,  # Large limit to count all
            with_payload=False
        )[0]
        
        chunks_count = len(namespace_points)
        
        return {
            "chunks_count": chunks_count,
            "last_updated": collection_info.status,
            "user_id": user_id,
            "bot_id": bot_id
        }
        
    except Exception as e:
        return {"error": f"Failed to get knowledge base stats: {str(e)}"}

@app.delete("/knowledge-bases/{user_id}/{bot_id}")
async def delete_knowledge_base(user_id: str, bot_id: str):
    """Delete a specific knowledge base (bot namespace)"""
    try:
        collection_name = f"{config.COLLECTION_PREFIX}{user_id}"
        namespace = f"{config.NAMESPACE_PREFIX}{bot_id}"
        
        if not qdrant_client.collection_exists(collection_name):
            return {"error": "Knowledge base not found"}
        
        # Get all points in the namespace
        points = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "metadata.namespace",
                        "match": {"value": namespace}
                    }
                ]
            },
            limit=10000,
            with_payload=False
        )[0]
        
        if points:
            # Delete all points in the namespace
            point_ids = [point.id for point in points]
            qdrant_client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
        
        return {"status": "success", "message": f"Deleted knowledge base {user_id}/{bot_id}"}
        
    except Exception as e:
        return {"error": f"Failed to delete knowledge base: {str(e)}"}

@app.websocket("/ws/chat/{user_id}/{bot_id}")
async def websocket_chat(websocket: WebSocket, user_id: str, bot_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if "message" not in message_data:
                await websocket.send_text(json.dumps({
                    "error": "Missing 'message' field in request"
                }))
                continue
            
            # Process message using existing search engine
            try:
                # Use config from message if provided, otherwise try Bubble.io
                has_config = any([
                    message_data.get("system_prompt"),
                    message_data.get("tone"),
                    message_data.get("industry"),
                    message_data.get("chatbot_name"),
                    message_data.get("description")
                ])
                
                if has_config:
                    # Use message values directly
                    system_prompt = message_data.get("system_prompt")
                    tone = message_data.get("tone")
                    industry = message_data.get("industry")
                    chatbot_name = message_data.get("chatbot_name")
                    description = message_data.get("description")
                else:
                    # Try Bubble.io (optional)
                    chatbot_config_data = await chatbot_config.get_chatbot_config(user_id, bot_id)
                    system_prompt = chatbot_config_data.get("system_prompt") if chatbot_config_data else None
                    tone = chatbot_config_data.get("tone") if chatbot_config_data else None
                    industry = chatbot_config_data.get("industry") if chatbot_config_data else None
                    chatbot_name = chatbot_config_data.get("chatbot_name") if chatbot_config_data else None
                    description = chatbot_config_data.get("description") if chatbot_config_data else None
                
                response = await search_engine.search_and_answer(
                    query=message_data["message"],
                    user_id=user_id,
                    bot_id=bot_id,
                    top_k=message_data.get("top_k", 3),
                    system_prompt=system_prompt,
                    tone=tone,
                    industry=industry,
                    chatbot_name=chatbot_name,
                    description=description
                )
                
                # Send response back to client
                await websocket.send_text(json.dumps({
                    "answer": response["answer"],
                    "citations": response["citations"],
                    "chunks_used": response["chunks_used"],
                    "confidence_scores": response.get("confidence_scores", [])
                }))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": f"Failed to process query: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        print(f"Client {user_id}/{bot_id} disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "error": f"WebSocket error: {str(e)}"
            }))
        except:
            pass

@app.get("/chatbot-config/{user_id}/{bot_id}")
async def get_chatbot_config_endpoint(user_id: str, bot_id: str):
    """Get chatbot configuration from Bubble.io or return default structure"""
    config_data = await chatbot_config.get_chatbot_config(user_id, bot_id)
    
    if config_data:
        return {
            "chatbot_name": config_data.get("chatbot_name"),
            "description": config_data.get("description"),
            "industry": config_data.get("industry"),
            "color": config_data.get("color"),
            "logo": config_data.get("logo"),
            "welcome_message": config_data.get("welcome_message"),
            "knowledge_source": config_data.get("knowledge_source"),
            "tone": config_data.get("tone"),
            "system_prompt": config_data.get("system_prompt"),
            "user_id": user_id,
            "bot_id": bot_id
        }
    else:
        # Return default structure if config not found
        return {
            "chatbot_name": None,
            "description": None,
            "industry": None,
            "color": None,
            "logo": None,
            "welcome_message": None,
            "knowledge_source": None,
            "tone": None,
            "system_prompt": None,
            "user_id": user_id,
            "bot_id": bot_id,
            "message": "Chatbot configuration not found. Using default settings."
        }

@app.get("/generate-uuid")
async def generate_uuid():
    """Generate a new UUID"""
    return {"uuid": str(uuid.uuid4())}

@app.get("/generate-qr/{user_id}/{bot_id}")
async def generate_qr_code(
    user_id: str,
    bot_id: str,
    bubble_url: Optional[str] = Query(None, description="Base URL of Bubble.io app (optional, uses BUBBLE_APP_URL from config if not provided)"),
    size: int = Query(300, description="QR code size in pixels", ge=100, le=1000)
):
    """
    Generate a QR code image that links to the Bubble.io chatbot page.
    Returns a PNG image that can be scanned to open the chatbot.
    """
    # Use provided bubble_url or fall back to config
    base_url = bubble_url or config.BUBBLE_APP_URL
    
    # Construct the chatbot URL with user_id and bot_id parameters
    chatbot_url = f"{base_url}/chatbot?user_id={user_id}&bot_id={bot_id}"
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(chatbot_url)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Resize if needed
    if size != 300:
        img = img.resize((size, size))
    
    # Convert to bytes
    img_buffer = BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    
    # Return image as response
    return Response(content=img_buffer.getvalue(), media_type="image/png")

@app.get("/qr-code/{user_id}/{bot_id}")
async def get_qr_code_info(
    user_id: str,
    bot_id: str,
    bubble_url: Optional[str] = Query(None, description="Base URL of Bubble.io app (optional, uses BUBBLE_APP_URL from config if not provided)"),
    include_base64: bool = Query(False, description="Include base64 encoded QR code image in response")
):
    """
    Get QR code information including the chatbot URL and QR code image URL.
    Returns JSON with QR code details that can be used for embedding or sharing.
    """
    # Use provided bubble_url or fall back to config
    base_url = bubble_url or config.BUBBLE_APP_URL
    
    # Construct the chatbot URL with user_id and bot_id parameters
    chatbot_url = f"{base_url}/chatbot?user_id={user_id}&bot_id={bot_id}"
    
    # Generate QR code URL (points to the image endpoint)
    # URL encode the bubble_url parameter properly
    encoded_bubble_url = quote(base_url, safe='')
    qr_code_url = f"/generate-qr/{user_id}/{bot_id}?bubble_url={encoded_bubble_url}"
    
    response_data = {
        "qr_code_url": qr_code_url,
        "chatbot_url": chatbot_url,
        "user_id": user_id,
        "bot_id": bot_id
    }
    
    # Optionally include base64 encoded image
    if include_base64:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(chatbot_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        response_data["qr_code_base64"] = f"data:image/png;base64,{img_base64}"
    
    return response_data

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Tenant Chatbot Platform API",
        "version": "1.0.0",
        "endpoints": {
            "ingest": "/ingest",
            "query": "/query", 
            "knowledge-bases": "/knowledge-bases",
            "chatbot-config": "/chatbot-config/{user_id}/{bot_id}",
            "generate-uuid": "/generate-uuid",
            "generate-qr": "/generate-qr/{user_id}/{bot_id}",
            "qr-code": "/qr-code/{user_id}/{bot_id}"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
