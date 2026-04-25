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
import asyncio
import time

from observability import span, shutdown

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ============== Configuration ==============
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.1.253:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:1b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

client = ollama.Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = """You are Sya, Yoseph Bernandus's personal AI assistant. \
When someone asks who you are, introduce yourself as: \
"Hi! I'm Sya, Yoseph's personal assistant. You can ask me anything related to Yoseph!" \
Always answer questions about Yoseph based on the provided context. \
Be friendly and helpful."""


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

@app.on_event("shutdown")
async def on_shutdown():
    shutdown()

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
    response = client.embed(model=EMBED_MODEL, input=text)
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
    """Check if server and Ollama are reachable"""
    try:
        client.list()
        return {"status": "ok"}
    except Exception:
        return {"status": "degraded", "detail": "Ollama unreachable"}


@app.post("/chat")
async def chat_direct(request: ChatRequest):
    """
    Direct chat with LLM
    Just sends the user's message to Ollama and returns the response
    """
    with span("workflow", "chat-direct") as root:
        if root:
            root.set_input({"message": request.message})

        with span("llm", "ollama-chat", model=CHAT_MODEL, provider="ollama") as s:
            if s:
                s.set_input(request.message)

            response = client.chat(
                model=CHAT_MODEL, messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request.message},
                ]
            )
            answer = response["message"]["content"]

            if s:
                s.set_output(answer)

        if root:
            root.set_output({"answer": answer})

        return {"answer": answer, "mode": "direct"}


@app.post("/chat/rag")
async def chat_rag(request: ChatRequest):
    """
    RAG Chat - Search knowledge base first, then generate answer
    """
    with span("workflow", "rag-chat") as root:
        if root:
            root.set_input({"message": request.message})

        with span("retrieval", "vector-search") as ret:
            if ret:
                ret.set_input(request.message)
            relevant_docs = search_documents(request.message, n_results=3)
            if ret:
                ret.set_output(relevant_docs)

        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.

    Context:
    {context}

    Question: {request.message}

    Answer:"""

        with span("llm", "ollama-chat", model=CHAT_MODEL, provider="ollama") as s:
            if s:
                s.set_input(prompt)

            response = client.chat(
                model=CHAT_MODEL, messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            answer = response["message"]["content"]

            if s:
                s.set_output(answer)

        result = {
            "answer": answer,
            "sources": [
                {
                    "content": doc["content"][:100] + "...",
                    "source": doc["source"],
                    "score": doc["score"],
                }
                for doc in relevant_docs
            ],
            "mode": "rag",
        }

        if root:
            root.set_output(result)

        return result


@app.post("/chat/rag/stream")
async def chat_rag_stream(request: ChatRequest):
    """
    RAG Chat with streaming (Server-Sent Events)
    Sends tokens as they're generated, with keepalive pings
    """

    async def generate():
        with span("workflow", "rag-chat-stream") as root:
            if root:
                root.set_input({"message": request.message})

            with span("retrieval", "vector-search") as ret:
                if ret:
                    ret.set_input(request.message)
                relevant_docs = search_documents(request.message, n_results=3)
                if ret:
                    ret.set_output(relevant_docs)

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

            context = "\n\n".join([doc["content"] for doc in relevant_docs])
            prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.

        Context:
        {context}

        Question: {request.message}

        Answer:"""

            with span("llm", "ollama-chat-stream", model=CHAT_MODEL, provider="ollama") as s:
                if s:
                    s.set_input(prompt)

                full_response = ""
                last_token_time = time.monotonic()
                for chunk in client.chat(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                ):
                    token = chunk["message"]["content"]
                    now = time.monotonic()
                    if now - last_token_time > 15:
                        yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                    if token:
                        full_response += token
                        token_data = {"type": "token", "content": token}
                        yield f"data: {json.dumps(token_data)}\n\n"
                        last_token_time = now

                if s:
                    s.set_output(full_response)

            if root:
                root.set_output({"answer_length": len(full_response)})

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


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
            "content": "Yoseph Bernandus is a Software Engineer based in Jakarta, Indonesia. Contact: yosephbernandus@gmail.com, +62 877 8154 0796. LinkedIn: linkedin.com/in/yosephbernandus, GitHub: github.com/yosephbernandus, Personal site: yoseph.my.id, Blog: blog.yosephbernandus.com. He is an experienced Software Engineer with expertise in building software systems. His work has enabled products serving 30,000+ monthly user registrations, processing 800,000 transactions per month, and over $5M in monthly disbursements. He has successfully led multiple projects in the finance industry, delivering significant improvements in system performance, security, and product flow.",
            "source": "yoseph-profile.md",
        },
        {
            "content": "Yoseph Bernandus works at PT JULO TEKNOLOGI FINANSIAL in Jakarta, Indonesia. As Lead Software Engineer (July 2025 - Present): Developed Python library optimizing Celery Worker utilization with dynamic worker addition without redeployment. Took ownership of government reporting project, improving success rate from 5% to 90%. Mentored team member to accelerate CI/CD static analysis using Ruff, reducing pipeline time from 7 minutes to 8 seconds. Guided UV package manager implementation for legacy core service, achieving 50% faster dependency builds. Co-initiated reconciliation service to automate Finance team's manual operations. As Senior Software Engineer (April 2024 - July 2025): Initiated complete redesign of Leadgen Core Product with plug-and-play architecture supporting 5 partners. Led modernization of Merchant Financing Core Product with Python/Django, UV, Ruff, and mypy. Redesigned product flow for government regulation compliance. Spearheaded SLIK credit information mechanism handling 20,000+ records daily. Partnered on cost optimization strategies for infrastructure.",
            "source": "yoseph-julo-experience.md",
        },
        {
            "content": "Yoseph Bernandus at JULO as Software Engineer (April 2022 - March 2024): Implemented Halt & Restart feature for Grab integration, improving repayment rates by 70%. Led Leadgen product development from scratch, now contributing 10% of company traffic with 30,000+ monthly users. Fixed critical IDOR security vulnerabilities, reducing PII data exposure by 40%. Eliminated brute force attack vectors on third-party services, cutting security costs by 20%. Key contributor in designing major partner product with SNAP BI standardization handling 30,000+ monthly users and driving 40% of company traffic. Designed automated Settlement Flow for disbursements, repayments and refunds. Optimized two critical API endpoints: reduced latency from 4s to 600ms and from 2s to 200ms. Technologies: Go, Python, Django, Ansible, Terraform, Jenkins, Github Actions, PostgreSQL, Redis, Docker, Google Cloud, Bash Script, Nix Shell, Locust, Goose, HTML, Javascript, CSS, NSQ, Celery, RabbitMQ.",
            "source": "yoseph-julo-engineer.md",
        },
        {
            "content": "Yoseph Bernandus previous experience: At PT DANABAGUS INDONESIA (December 2019 - March 2022) as Software Developer, developed end-to-end processes handling IDR 1 billion per month. Served as PIC for audits including ISO 90001 certification and Indonesian government agencies (FDC, Pusdafil, Silaras) with no findings. Optimized database N+1 queries reducing response time from 1s to 50ms with index strategy. Created caching implementation reducing database traffic. Built CSV automation reconciliation for finance team. Technologies: Python, Django, PostgreSQL, Redis, Alicloud, Kotlin, Android Development, Firebase. At PT INFOTEK GLOBAL NETWORK (June 2014 - September 2019) as NOC Engineer, monitored ISP network infrastructure, configured Mikrotik routers and wireless access points, created internal ticketing system using PHP 5 and Javascript. Technologies: Mikrotik, Cisco, Proxmox, Ubiquity.",
            "source": "yoseph-previous-experience.md",
        },
        {
            "content": "Yoseph Bernandus education and skills: Bachelor's Degree in Computer Science from Nusantara Manado University (July 2014 - February 2018), graduated with 3.67 GPA focusing on hardware IoT. Final project: smart home prototype using Wemos D1 ESP 32 microcontroller with real-time MQTT protocol control and web app interface. Skills: Web Development, Android Development, Backend Engineering, System Design, Linux Server Configuration, Network Configuration, Project Management, Deployment Pipeline Configuration, Software Architecture. Notable projects: Academic System (Codeigniter 3), Smart Home IoT thesis project, Mikrolet real-time GPS tracking (Node.js/React), Uangku financial management app (Kotlin), Baevicca e-commerce platform, Siaptes CASN exam preparation platform.",
            "source": "yoseph-education-skills.md",
        },
        {
            "content": "I am Sya, Yoseph Bernandus's personal AI assistant. If someone asks who I am, I should introduce myself as Sya and let them know they can ask me anything related to Yoseph — his career, skills, experience, projects, education, or anything else about him. I'm here to help people learn about Yoseph Bernandus.",
            "source": "sya-identity.md",
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
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            mode = data.get("mode", "rag")

            print(f"Received: {message} (mode: {mode})")

            with span("workflow", f"ws-chat-{mode}") as root:
                if root:
                    root.set_input({"message": message, "mode": mode})

                if mode == "rag":
                    with span("retrieval", "vector-search") as ret:
                        if ret:
                            ret.set_input(message)
                        relevant_docs = search_documents(message, n_results=3)
                        if ret:
                            ret.set_output(relevant_docs)

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

                    context = "\n\n".join([doc["content"] for doc in relevant_docs])
                    prompt = f"""Answer the question based on the following context. If the context doesn't contain relevant information, say so.

        Context:
        {context}

        Question: {message}
        Answer:"""
                else:
                    prompt = message

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for h in history:
                    messages.append({"role": h["role"], "content": h["content"]})
                messages.append({"role": "user", "content": prompt})

                with span("llm", "ollama-chat-ws", model=CHAT_MODEL, provider="ollama") as s:
                    if s:
                        s.set_input(prompt)

                    full_response = ""
                    for chunk in client.chat(model=CHAT_MODEL, messages=messages, stream=True):
                        token = chunk["message"]["content"]
                        if token:
                            full_response += token
                            await websocket.send_json({"type": "token", "data": token})

                    if s:
                        s.set_output(full_response)

                await websocket.send_json({"type": "done"})

                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": full_response})

                if root:
                    root.set_output({"answer_length": len(full_response)})

                print(f"Sent response: {full_response[:50]}...")
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
