# Chat Agent

A simple RAG (Retrieval-Augmented Generation) chat application with real-time streaming responses.

## Features

- Direct chat with LLM
- RAG-enhanced chat with document retrieval
- Three communication modes: HTTP, Server-Sent Events (SSE), and WebSocket
- Real-time streaming responses
- ChromaDB vector storage for document embeddings
- React frontend with TypeScript

## Tech Stack

**Backend**
- FastAPI
- Ollama (llama3.2:3b for chat, nomic-embed-text for embeddings)
- ChromaDB (in-memory vector database)
- Python 3.11+

**Frontend**
- React 19
- TypeScript
- Vite
- Bun (package manager)
- Tailwind CSS

## Prerequisites

You need to have these installed:

```bash
# Ollama with models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama serve

# Python 3.11 or higher
# Bun (https://bun.sh)
```

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd chat-agent

# Install dependencies
make install

# Seed sample documents (optional but recommended)
make setup

# Start development servers
make dev
```

The backend will run on http://localhost:8000 and frontend on http://localhost:5173.

## Development

```bash
# Start backend only
make backend

# Start frontend only
make frontend

# Start both
make dev

# Check backend health
make health

# Add sample documents
make seed

# Clear all documents
make clear-docs
```

## API Endpoints

### Chat Endpoints

- `POST /chat` - Direct chat without RAG
- `POST /chat/rag` - RAG-enhanced chat
- `POST /chat/rag/stream` - Streaming RAG chat (SSE)
- `WS /ws/chat` - WebSocket real-time chat

### Document Management

- `GET /documents` - List all documents
- `POST /documents` - Add a new document
- `POST /documents/seed` - Add sample documents
- `DELETE /documents` - Clear all documents

### Health

- `GET /health` - Check backend status

## Testing

```bash
# Run backend tests
make test

# Test WebSocket connection
make test-ws

# Manual testing with curl
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "Hello"}'
```

## Project Structure

```
chat-agent/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── test_websocket.py   # WebSocket test script
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React component
│   │   ├── App.css         # Styles
│   │   └── main.tsx        # Entry point
│   ├── package.json        # Bun dependencies
│   └── vite.config.ts      # Vite configuration
└── Makefile                # Development commands
```

## How RAG Works

1. User asks a question
2. Question is converted to an embedding vector
3. ChromaDB searches for similar documents using vector similarity
4. Top matching documents are retrieved
5. Documents are used as context for the LLM
6. LLM generates an answer based on the context
7. Response is streamed back to the user

## Communication Modes

**HTTP**: Simple request-response for single queries

**SSE (Server-Sent Events)**: Server pushes data to client for streaming responses

**WebSocket**: Full-duplex communication for real-time bidirectional chat with conversation history

## License

MIT
