"""
Simple RAG Chat Agent - Backend
FastAPI server with Ollama + ChromaDB
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

import json
import ollama
import chromadb
import uuid
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

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
    Similiar texts have similar vectors
    """
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def search_documents(query: str, n_results: int = 3):
    """
    Search for documents similiar to the query
    Steps:
    1. Convert query to vector
    2. Find similar documents in ChromaDB
    3. Return top N results with similarity scores
    """
    # Embed the query
    query_embedding = create_embedding(query)

    # Search ChromaDB for similar documents
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    # Format results
    documents = []
    if results["ids"] and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            documents.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "score": results["distances"][0][i]
                    if "distances" in results
                    else 0,
                }
            )

    return documents


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


@app.post("/chat/rag")
async def chat_rag(request: ChatRequest):
    """
    RAG Chat - Search knowledge base first, then generate answer
    """
    # Step 1: Search for relevant documents
    relevant_docs = search_documents(request.message, n_results=3)

    # Step 2: Build context from retrieved documents
    context = "\n\n".join([doc["content"] for doc in relevant_docs])

    # Step 3: Create prompt with context
    prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.

    Context:
    {context}

    Question: {request.message}

    Answer:"""

    # Step 4: Generate answer with LLM
    response = ollama.chat(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # Step 5: Return answer with sources
    return {
        "answer": answer,
        "sources": [
            {
                "content": doc["content"][:100] + "...",  # Preview
                "source": doc["source"],
                "score": doc["score"],
            }
            for doc in relevant_docs
        ],
        "mode": "rag",
    }


@app.post("/chat/rag/stream")
async def chat_rag_stream(request: ChatRequest):
    """
    RAG Chat with streaming (Server-Sent Events)
    Sends tokens as they're generated
    """

    async def generate():
        # Step 1: Search for relevant documents
        relevant_docs = search_documents(request.message, n_results=3)

        # Step 2: Send source first
        source_data = {
            "type": "sources",
            "content": [
                {
                    "content": doc["content"][:100] + "...",
                    "source": doc["source"],
                    "score": doc["score"],
                }
                for doc in relevant_docs
            ],
        }

        yield f"data: {json.dumps(source_data)}\n\n"

        # Step 3: Build Context and prompt
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.

        Context:
        {context}

        Question: {request.message}

        Answer:"""

        # Step 4: Stream tokens from LLM
        for chunk in ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            token = chunk["message"]["content"]
            if token:
                token_data = {"type": "token", "content": token}
                yield f"data: {json.dumps(token_data)}\n\n"

        # Step 5: Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/documents")
async def add_document(request: DocumentRequest):
    """
    Add a document to the knowledge base
    Steps:
    1. Generate embedding for the document content
    2. Store in ChromaDB with metadata
    """
    # Generate unique ID for this document
    doc_id = str(uuid.uuid4())

    # Convert text to vector
    embedding = create_embedding(request.content)

    # Store in ChromaDB
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[request.content],
        metadatas=[{"source": request.source}],
    )

    return {"success": True, "id": doc_id}


@app.get("/documents")
async def list_documents():
    """
    Get all documents in the knowledge base
    """
    # Get all documents from collection
    data = collection.get()

    return {
        "count": len(data["ids"]),
        "documents": [
            {
                "id": data["ids"][i],
                "content": data["documents"][i],
                "source": data["metadatas"][i]["source"],
            }
            for i in range(len(data["ids"]))
        ],
    }


@app.post("/documents/seed")
async def seed_documents():
    """
    Add 5 sample documents about tech topics
    """
    SAMPLE_DOCS = [
        {
            "content": "FastAPI is a modern Python web framework for building APIs. Key features: automatic docs, type hints, async support, dependency injection, very fast performance.",
            "source": "fastapi-guide.md",
        },
        {
            "content": "React Hooks let you use state in functional components. useState for state, useEffect for side effects, useCallback for memoized functions, useMemo for cached values.",
            "source": "react-hooks.md",
        },
        {
            "content": "RAG (Retrieval-Augmented Generation) improves AI answers by first searching a knowledge base, then using found documents as context for the LLM to generate grounded responses.",
            "source": "rag-guide.md",
        },
        {
            "content": "PostgreSQL optimization: use EXPLAIN ANALYZE, create proper indexes, avoid N+1 queries, use connection pooling, partition large tables.",
            "source": "postgres-tips.md",
        },
        {
            "content": "Redis data structures: Strings for caching, Hashes for objects, Lists for queues, Sets for unique items, Sorted Sets for leaderboards, Pub/Sub for messaging.",
            "source": "redis-guide.md",
        },
    ]

    # Add each sample document
    for doc in SAMPLE_DOCS:
        doc_id = str(uuid.uuid4())
        embedding = create_embedding(doc["content"])

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc["content"]],
            metadatas=[{"source": doc["source"]}],
        )

    return {"success": True, "added": len(SAMPLE_DOCS)}


@app.delete("/documents")
async def clear_documents():
    """
    Delete all documents from the knowledge base
    """
    # Get all document IDs
    data = collection.get()

    if len(data["ids"]) > 0:
        # Delete all documents
        collection.delete(ids=data["ids"])

    return {"success": True, "deleted": len(data["ids"])}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional chat
    Supports both direct and RAG modes with conversation history
    """

    # Step 1: Accept the websocket connection
    await websocket.accept()
    print("Client connected to WebSocket")

    # Keep conversation history for this session
    history = []

    try:
        # Step 2: Loop forever until client disconnects
        while True:
            # Step 3: Wait for message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            mode = data.get("mode", "rag")  # direct or rag

            print(f"Received: {message} (mode: {mode})")

            # Step 4: Process based on mode
            if mode == "rag":
                # Search knowledge base
                relevant_docs = search_documents(message, n_results=3)

                # Send source first
                await websocket.send_json(
                    {
                        "type": "sources",
                        "data": [
                            {
                                "content": doc["content"][:100] + "...",
                                "source": doc["source"],
                                "score": doc["score"],
                            }
                            for doc in relevant_docs
                        ],
                    }
                )

                # Build context
                context = "\n\n".join([doc["content"] for doc in relevant_docs])
                prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.

        Context:
        {context}

        Question: {message}
        Answer:"""
            else:
                # Direct mode no RAG
                prompt = message
                context = None

            # Step 5: Build messages with history
            messages = []

            # add conversation history
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})

            # Add current message
            messages.append({"role": "user", "content": prompt})

            # Step 6: Stream LLM response token by token
            full_response = ""
            for chunk in ollama.chat(model=CHAT_MODEL, messages=messages, stream=True):
                token = chunk["message"]["content"]
                if token:
                    full_response += token
                    await websocket.send_json({"type": "token", "data": token})

            # Step 7: Send done signal
            await websocket.send_json({"type": "done"})

            # Step 8: Update history for multi-turn conversation
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": full_response})

            print(f"Sent response: {full_response[:50]}...")
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
