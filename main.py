import os
import uvicorn
import uuid
import base64
from typing import Optional
from io import BytesIO
from urllib.parse import quote
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import qrcode
import json
import config
import data_ingestion
import search_engine
import chatbot_config
import database
from pydantic import BaseModel
from utils.logging_config import get_logger
from utils.metrics import get_metrics
from utils.conversation_memory import get_history, add_turn, clear_history as clear_conversation_history

logger = get_logger(__name__)

app = FastAPI(
    title="Fusor AI API",
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

from fastapi.responses import Response, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

# ...

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Explicitly intercept widget.js to ensure it is NEVER cached by the browser
@app.get("/static/widget.js")
async def serve_widget_js():
    file_path = os.path.join(_static_dir, "widget.js")
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return _error_response("Widget script not found", status_code=404)

if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Initialize Qdrant client for knowledge base listing
qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

# Initialize local SQLite DB for chatbot configs
database.init_db()


def _error_response(message: str, code: str = "error", status_code: int = 400) -> JSONResponse:
    """Standard error response: { \"error\": { \"code\": ..., \"message\": ... } }."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


@app.get("/health")
async def health():
    """Liveness: is the process up."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: can the app serve traffic (e.g. Qdrant reachable)."""
    try:
        qdrant_client.get_collections()
        return {"status": "ready"}
    except Exception as e:
        logger.warning("Ready check failed: Qdrant unreachable", extra={"error": str(e)})
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "service_unavailable", "message": "Qdrant unavailable"}},
        )


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
                    logger.warning("Error processing collection", extra={"collection": collection_name, "error": str(e)})
                    continue
        return knowledge_bases
    except Exception as e:
        logger.exception("Failed to list knowledge bases")
        return _error_response(f"Failed to list knowledge bases: {str(e)}", "list_failed", 500)

@app.get("/knowledge-bases/{user_id}/{bot_id}/stats")
async def get_knowledge_base_stats(user_id: str, bot_id: str):
    """Get statistics for a specific knowledge base"""
    try:
        collection_name = f"{config.COLLECTION_PREFIX}{user_id}"
        namespace = f"{config.NAMESPACE_PREFIX}{bot_id}"
        
        if not qdrant_client.collection_exists(collection_name):
            return _error_response("Knowledge base not found", "not_found", 404)
        collection_info = qdrant_client.get_collection(collection_name)
        namespace_filter = Filter(
            must=[FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace))]
        )
        chunks_count = 0
        offset = None
        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=namespace_filter,
                limit=1000,
                offset=offset,
                with_payload=False,
            )
            chunks_count += len(points)
            if next_offset is None:
                break
            offset = next_offset
        
        return {
            "chunks_count": chunks_count,
            "last_updated": collection_info.status,
            "user_id": user_id,
            "bot_id": bot_id
        }
        
    except Exception as e:
        logger.exception("Failed to get knowledge base stats")
        return _error_response(f"Failed to get knowledge base stats: {str(e)}", "stats_failed", 500)


@app.delete("/knowledge-bases/{user_id}/{bot_id}")
async def delete_knowledge_base(user_id: str, bot_id: str):
    """Delete a specific knowledge base (bot namespace). Uses filter delete so no scroll limit."""
    try:
        collection_name = f"{config.COLLECTION_PREFIX}{user_id}"
        namespace = f"{config.NAMESPACE_PREFIX}{bot_id}"
        if not qdrant_client.collection_exists(collection_name):
            return _error_response("Knowledge base not found", "not_found", 404)
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace))]
            ),
        )
        logger.info("Deleted knowledge base", extra={"user_id": user_id, "bot_id": bot_id})
        return {"status": "success", "message": f"Deleted knowledge base {user_id}/{bot_id}"}
    except Exception as e:
        logger.exception("Failed to delete knowledge base")
        return _error_response(f"Failed to delete knowledge base: {str(e)}", "delete_failed", 500)

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
                
                chat_history = get_history(user_id, bot_id)
                response = await search_engine.search_and_answer(
                    query=message_data["message"],
                    user_id=user_id,
                    bot_id=bot_id,
                    top_k=message_data.get("top_k", 3),
                    system_prompt=system_prompt,
                    tone=tone,
                    industry=industry,
                    chatbot_name=chatbot_name,
                    description=description,
                    chat_history=chat_history,
                )
                if isinstance(response, dict):
                    add_turn(user_id, bot_id, message_data["message"], response["answer"])
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

@app.post("/chat/clear-history/{user_id}/{bot_id}")
async def clear_chat_history(user_id: str, bot_id: str):
    """Clear short-term conversation memory for this user/bot."""
    clear_conversation_history(user_id, bot_id)
    return {"status": "ok", "message": "Conversation history cleared"}


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
            "tone": config_data.get("tone"),
            "system_prompt": config_data.get("system_prompt"),
            "temperature": config_data.get("temperature"),
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
            "tone": None,
            "system_prompt": None,
            "temperature": None,
            "user_id": user_id,
            "bot_id": bot_id,
            "message": "Chatbot configuration not found. Using default settings."
        }

class ChatbotConfigRequest(BaseModel):
    chatbot_name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    color: Optional[str] = None
    logo: Optional[str] = None
    welcome_message: Optional[str] = None
    tone: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None

@app.post("/chatbot-config/{user_id}/{bot_id}")
async def save_chatbot_config_endpoint(
    user_id: str, 
    bot_id: str, 
    chatbot_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    logo: Optional[str] = Form(None),
    welcome_message: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None)
):
    """Save chatbot configuration to the local SQLite database using Form data."""
    
    parsed_temp = None
    if temperature is not None and str(temperature).strip():
        try:
            parsed_temp = float(temperature)
        except ValueError:
            pass
            
    config_dict = {
        "chatbot_name": chatbot_name,
        "description": description,
        "industry": industry,
        "color": color,
        "logo": logo,
        "welcome_message": welcome_message,
        "tone": tone,
        "system_prompt": system_prompt,
        "temperature": parsed_temp
    }
    success = database.save_chatbot_config(user_id, bot_id, config_dict)
    if success:
        return {"status": "success", "message": "Configuration saved successfully"}
    return _error_response("Failed to save configuration", status_code=500)

@app.delete("/chatbot-config/{user_id}/{bot_id}")
async def delete_chatbot_config_endpoint(user_id: str, bot_id: str):
    """Delete chatbot configuration from the local SQLite database."""
    success = database.delete_chatbot_config(user_id, bot_id)
    if success:
        return {"status": "success", "message": "Configuration deleted successfully"}
    return _error_response("Failed to delete configuration", status_code=500)

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


# -------------------------
# Embeddable widget & domain
# -------------------------

@app.get("/embed/config")
async def embed_config(request: Request):
    """
    Return the public API base URL (domain) and widget info for embedding.
    Used by Bubble or other clients to get the correct domain for the embed snippet.
    """
    base = (config.API_PUBLIC_URL or str(request.base_url).rstrip("/"))
    return {
        "api_public_url": base,
        "widget_script_url": f"{base}/static/widget.js",
        "placement_options": ["bottom-corner", "floating", "inline"],
        "snippet_docs": "Add script with data-api-url, data-user-id, data-bot-id, data-placement (optional), data-target (for inline).",
    }


@app.get("/embed/snippet")
async def embed_snippet(
    request: Request,
    user_id: str = Query(..., description="Creator user ID"),
    bot_id: str = Query(..., description="Chatbot bot ID"),
    placement: str = Query("bottom-corner", description="Widget placement: bottom-corner, floating, or inline"),
    api_url: Optional[str] = Query(None, description="Override API base URL (default: API_PUBLIC_URL or request base)"),
):
    """
    Return the HTML snippet and domain to embed this chatbot as a website widget.
    """
    base = (api_url or config.API_PUBLIC_URL or str(request.base_url).rstrip("/")).rstrip("/")
    script_src = f"{base}/static/widget.js?v=2"
    placement = placement if placement in ("bottom-corner", "floating", "inline") else "bottom-corner"
    attrs = [
        f'src="{script_src}"',
        f'data-api-url="{base}"',
        f'data-user-id="{user_id}"',
        f'data-bot-id="{bot_id}"',
        f'data-placement="{placement}"',
    ]
    snippet = "<script " + " ".join(attrs) + "></script>"
    return {
        "snippet": snippet,
        "api_base_url": base,
        "user_id": user_id,
        "bot_id": bot_id,
        "placement": placement,
    }


@app.get("/embed", response_class=HTMLResponse)
async def embed_iframe(
    request: Request,
    user_id: str = Query(..., description="Creator user ID"),
    bot_id: str = Query(..., description="Chatbot bot ID"),
    placement: str = Query("bottom-corner", description="Widget placement: bottom-corner, floating, or inline"),
):
    """
    Minimal HTML page that loads the widget script with query params.
    Use as iframe src for iframe-based embed: <iframe src="https://api/embed?user_id=...&bot_id=..."></iframe>
    """
    base = (config.API_PUBLIC_URL or str(request.base_url).rstrip("/")).rstrip("/")
    placement = placement if placement in ("bottom-corner", "floating", "inline") else "bottom-corner"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chat</title>
  <style>body{{margin:0;min-height:100vh;}}#fusor-embed-root{{min-height:400px;}}</style>
</head>
<body>
  <div id="fusor-embed-root"></div>
  <script>
(function(){{
  var params = new URLSearchParams(window.location.search);
  var uid = params.get('user_id') || '';
  var bid = params.get('bot_id') || '';
  var pl = params.get('placement') || 'bottom-corner';
  var apiUrl = '{base}';
  var script = document.createElement('script');
  script.src = apiUrl + '/static/widget.js?v=' + Date.now();
  script.setAttribute('data-api-url', apiUrl);
  script.setAttribute('data-user-id', uid);
  script.setAttribute('data-bot-id', bid);
  script.setAttribute('data-placement', pl);
  if (pl === 'inline') script.setAttribute('data-target', '#fusor-embed-root');
  document.body.appendChild(script);
}})();
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/metrics")
async def metrics():
    """Basic request metrics (single-process in-memory)."""
    return get_metrics()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Tenant Chatbot Platform API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "ingest": "/ingest",
            "ingest-status": "/ingest/status/{job_id}",
            "query": "/query",
            "chat-clear-history": "/chat/clear-history/{user_id}/{bot_id}",
            "knowledge-bases": "/knowledge-bases",
            "chatbot-config-get": "GET /chatbot-config/{user_id}/{bot_id}",
            "chatbot-config-post": "POST /chatbot-config/{user_id}/{bot_id}",
            "chatbot-config-delete": "DELETE /chatbot-config/{user_id}/{bot_id}",
            "generate-uuid": "/generate-uuid",
            "generate-qr": "/generate-qr/{user_id}/{bot_id}",
            "qr-code": "/qr-code/{user_id}/{bot_id}",
            "embed-config": "/embed/config",
            "embed-snippet": "/embed/snippet",
            "embed-iframe": "/embed",
            "metrics": "/metrics",
        },
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
