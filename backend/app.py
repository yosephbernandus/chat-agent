"""
Simple RAG Chat Agent - Backend
FastAPI server with Ollama + ChromaDB
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import chromadb
import uuid

# ============== Configuration ==============
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"


# ============== Request Models ==============
class ChatRequest(BaseModel):
    message: str


class DocumentRequest(BaseModel):
    content: str
    source: str


# ============== App Setup ==============
app = FastAPI(title="Simple RAG Chat")

# CORS - allows frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.Client()

try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection(
        name="documents", metadata={"description": "Knowledge base documents"}
    )


def create_embedding(text: str) -> list[float]:
    """
    Convert text into a vector (list of numbers)
    Similiar texts have similiar vectors
    """
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


# ============== Health Check ==============
@app.get("/health")
async def health_check():
    """Check if server is running"""
    return {"status": "ok"}


@app.post("/chat")
async def chat_direct(request: ChatRequest):
    """
    Direct chat with LLM
    Just sends the user's message to Ollama and returns the response
    """
    # Call Ollama's chat API
    response = ollama.chat(
        model=CHAT_MODEL, messages=[{"role": "user", "content": request.message}]
    )

    # Extract the answer from the response
    answer = response["message"]["content"]

    return {"answer": answer, "mode": "direct"}
