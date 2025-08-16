from fastapi import FastAPI, Request, UploadFile, File, Form, APIRouter
from pydantic import BaseModel
from typing import Optional
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeStore
from pinecone import Pinecone

import os
import tempfile

router = APIRouter()
converter = DocumentConverter()
load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")  # Example
INDEX_NAME = "my-bot-index"  # You can make this dynamic per bot

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # bge-base-en-v1.5 output size
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )


# -------------------------
# Embedding Model (bge-base-en-v1.5)
# -------------------------
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},  # Change to "cuda" if you have GPU
    encode_kwargs={"normalize_embeddings": True}
)


class IngestRequest(BaseModel):
    url: Optional[str] = None


@router.post("/ingest")
async def ingest(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    bot_id: Optional[str] = Form(None)  # Optional: Namespace per bot
):
    sources = []

    # -------------------------
    # 1. Determine input source
    # -------------------------
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = converter.convert(tmp_path)
        sources.append(result)

    elif url:
        if "sitemap.xml" in url:
            sitemap_urls = get_sitemap_urls(url)
            sources = list(converter.convert_all(sitemap_urls))
        else:
            result = converter.convert(url)
            sources.append(result)
    else:
        return {"error": "Either 'file' or 'url' must be provided."}

    all_chunks = []

    # -------------------------
    # 2. Extract & chunk
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

    for result in sources:
        if not result.document:
            continue

        doc = result.document
        text = doc.export_to_markdown()

        chunks = splitter.create_documents([text])
        all_chunks.extend(chunks)

    # -------------------------
    # 3. Store in Pinecone
    # -------------------------
    if not all_chunks:
        return {"error": "No content to embed."}

    namespace = bot_id if bot_id else "default"

    PineconeStore.from_documents(
        documents=all_chunks,
        embedding=EMBEDDING_MODEL,
        index_name=INDEX_NAME,
        namespace=namespace
    )

    return {
        "status": "success",
        "chunks_stored": len(all_chunks),
        "pinecone_index": INDEX_NAME,
        "namespace": namespace
    }


# -------------------------
# Run as main
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("your_filename:router", host="0.0.0.0", port=8000, reload=True)
