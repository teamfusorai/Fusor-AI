from fastapi import FastAPI, Request, UploadFile, File, Form, APIRouter
from pydantic import BaseModel
from typing import Optional
from docling.document_converter import DocumentConverter
import PyPDF2
import docx
from utils.sitemap import get_sitemap_urls
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
import tempfile
import config

# Set environment variable to disable symlinks on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -------------------------
# Setup
# -------------------------
router = APIRouter()
load_dotenv()

# Initialize docling for URL processing, fallback for documents
try:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    DOCLING_AVAILABLE = True
    print("Docling available for URL processing")
except ImportError:
    converter = None
    DOCLING_AVAILABLE = False
    print("Docling not available, using fallback for all processing")

def extract_text_fallback(file_path: str) -> str:
    """Fallback text extraction for when Docling fails"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        elif file_ext in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        elif file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        else:
            return f"Unsupported file type: {file_ext}"
            
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_from_url(url: str) -> str:
    """Extract text from URL using requests and BeautifulSoup"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text.strip()
        
    except Exception as e:
        return f"Error extracting text from URL: {str(e)}"

QDRANT_HOST = config.QDRANT_HOST
QDRANT_PORT = config.QDRANT_PORT

# ✅ LangChain-compatible embeddings
EMBEDDING_MODEL = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ✅ Native Qdrant client (only to create collection if missing)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

class IngestRequest(BaseModel):
    url: Optional[str] = None
    user_id: Optional[str] = None
    bot_id: Optional[str] = None

# -------------------------
# Ingestion Endpoint
# -------------------------
@router.post("/ingest")
async def ingest(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    bot_id: Optional[str] = Form(None)
):
    sources = []

    # -------------------------
    # 1. Determine input source
    # -------------------------
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Use fast fallback extraction directly
        text = extract_text_fallback(tmp_path)
        if text and not text.startswith("Error"):
            # Create a simple result object
            class SimpleResult:
                def __init__(self, text):
                    self.document = SimpleDocument(text)
            
            class SimpleDocument:
                def __init__(self, text):
                    self.text = text
                
                def export_to_markdown(self):
                    return self.text
            
            sources.append(SimpleResult(text))
        else:
            return {"error": f"Failed to extract text: {text}"}
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

    elif url:
        if DOCLING_AVAILABLE and converter:
            # Use docling for URL processing
            if "sitemap.xml" in url:
                try:
                    sitemap_urls = get_sitemap_urls(url)
                    sources = list(converter.convert_all(sitemap_urls))
                except Exception as e:
                    # Fallback to single page if sitemap fails
                    print(f"Sitemap processing failed: {e}")
                    print(f"Falling back to single page processing: {url}")
                    result = converter.convert(url)
                    sources.append(result)
            else:
                result = converter.convert(url)
                sources.append(result)
        else:
            # Fallback to simple URL processing if docling not available
            print("Docling not available, using simple URL processing")
            if "sitemap.xml" in url:
                try:
                    sitemap_urls = get_sitemap_urls(url)
                    for sitemap_url in sitemap_urls:
                        text = extract_text_from_url(sitemap_url)
                        if text and not text.startswith("Error"):
                            class SimpleResult:
                                def __init__(self, text, url):
                                    self.document = SimpleDocument(text)
                                    self.url = url
                            
                            class SimpleDocument:
                                def __init__(self, text):
                                    self.text = text
                                
                                def export_to_markdown(self):
                                    return self.text
                            
                            sources.append(SimpleResult(text, sitemap_url))
                except Exception as e:
                    print(f"Sitemap processing failed: {e}")
                    print(f"Falling back to single page processing: {url}")
                    text = extract_text_from_url(url)
                    if text and not text.startswith("Error"):
                        class SimpleResult:
                            def __init__(self, text, url):
                                self.document = SimpleDocument(text)
                                self.url = url
                        
                        class SimpleDocument:
                            def __init__(self, text):
                                self.text = text
                            
                            def export_to_markdown(self):
                                return self.text
                        
                        sources.append(SimpleResult(text, url))
            else:
                text = extract_text_from_url(url)
                if text and not text.startswith("Error"):
                    class SimpleResult:
                        def __init__(self, text, url):
                            self.document = SimpleDocument(text)
                            self.url = url
                    
                    class SimpleDocument:
                        def __init__(self, text):
                            self.text = text
                        
                        def export_to_markdown(self):
                            return self.text
                    
                    sources.append(SimpleResult(text, url))
    else:
        return {"error": "Either 'file' or 'url' must be provided."}

    all_chunks = []

    # -------------------------
    # 2. Extract & chunk
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # Multi-tenant setup
    user_id = user_id if user_id else "default_user"
    bot_id = bot_id if bot_id else "default_bot"
    collection_name = f"{config.COLLECTION_PREFIX}{user_id}"
    namespace = f"{config.NAMESPACE_PREFIX}{bot_id}"
    
    # Create collection if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSIONS, 
                distance=getattr(Distance, config.EMBEDDING_DISTANCE_METRIC)
            )
        )
    
    for result in sources:
        if not result.document:
            continue

        doc = result.document
        text = doc.export_to_markdown()
        
        # Extract source information
        source_name = "unknown"
        if file:
            source_name = file.filename
        elif url:
            source_name = url
        
        chunks = splitter.create_documents([text])
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Enhanced metadata for better retrieval and citation
            chunk.metadata.update({
                "namespace": namespace,
                "source_name": source_name,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "user_id": user_id,
                "bot_id": bot_id
            })

        all_chunks.extend(chunks)


    # -------------------------
    # 3. Store in Qdrant (LangChain wrapper)
    # -------------------------
    if not all_chunks:
        return {"error": "No content to embed."}

    Qdrant.from_documents(
        documents=all_chunks,
        embedding=EMBEDDING_MODEL,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        collection_name=collection_name,
    )

    return {
        "status": "success",
        "chunks_stored": len(all_chunks),
        "qdrant_collection": collection_name,
        "namespace": namespace,
        "user_id": user_id,
        "bot_id": bot_id,
        "source_name": source_name if 'source_name' in locals() else "unknown"
    }

# -------------------------
# Run as main
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("your_filename:router", host="0.0.0.0", port=8000, reload=True)
