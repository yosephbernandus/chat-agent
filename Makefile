.PHONY: help install setup backend frontend dev health seed clear-docs lint build test test-ws clean

help:
	@echo "Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install dependencies"
	@echo "  make setup        Full setup with sample data"
	@echo ""
	@echo "Development:"
	@echo "  make backend      Start backend server"
	@echo "  make frontend     Start frontend dev server"
	@echo "  make dev          Start both backend and frontend"
	@echo ""
	@echo "Utils:"
	@echo "  make health       Check backend health"
	@echo "  make seed         Add sample documents"
	@echo "  make clear-docs   Clear all documents"
	@echo "  make lint         Lint frontend code"
	@echo "  make build        Build frontend for production"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run tests"
	@echo "  make test-ws      Test WebSocket connection"
	@echo ""
	@echo "  make clean        Remove dependencies and cache"

install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && bun install
	@echo "Done"

setup: install
	@echo "Starting backend..."
	cd backend && python -m uvicorn app:app --port 8000 > /tmp/backend.log 2>&1 &
	@sleep 3
	@echo "Seeding sample documents..."
	@curl -s -X POST http://localhost:8000/documents/seed > /dev/null
	@pkill -f "python -m uvicorn"
	@echo "Setup complete. Run 'make dev' to start"

backend:
	cd backend && python -m uvicorn app:app --reload --port 8000

frontend:
	cd frontend && bun run dev

dev:
	@echo "Starting servers..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	@echo ""
	@(cd backend && python -m uvicorn app:app --reload --port 8000) & \
	(cd frontend && bun run dev) & \
	wait

health:
	@curl -s http://localhost:8000/health | jq .

seed:
	curl -X POST http://localhost:8000/documents/seed | jq .

clear-docs:
	curl -X DELETE http://localhost:8000/documents | jq .

lint:
	cd frontend && bun run lint

build:
	cd frontend && bun run build

test:
	cd backend && python -m pytest

test-ws:
	cd backend && python test_websocket.py

clean:
	@echo "Cleaning up..."
	rm -rf backend/__pycache__ backend/.ruff_cache
	rm -rf frontend/node_modules frontend/dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Done"
