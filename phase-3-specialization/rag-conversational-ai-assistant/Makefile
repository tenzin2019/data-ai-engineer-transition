# RAG Conversational AI Assistant - Makefile
.PHONY: help setup install test clean run run-dev docker-up docker-down lint format

# Default target
help:
	@echo "RAG Conversational AI Assistant - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup     - Run automated setup script"
	@echo "  install   - Install Python dependencies"
	@echo "  test      - Run test setup verification"
	@echo ""
	@echo "Development Commands:"
	@echo "  run       - Start the full application (API + Frontend)"
	@echo "  run-dev   - Start in development mode"
	@echo "  run-api   - Start only the API server"
	@echo "  run-frontend - Start only the frontend"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-up    - Start all services with Docker Compose"
	@echo "  docker-down  - Stop all Docker services"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-logs  - View Docker logs"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint      - Run linting (flake8, black check)"
	@echo "  format    - Format code (black, isort)"
	@echo "  test-unit - Run unit tests"
	@echo "  test-api  - Run API tests"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-azure - Deploy to Azure App Service"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean     - Clean up temporary files"
	@echo "  reset     - Reset vector database"

# Setup commands
setup:
	@echo "Running automated setup..."
	./scripts/setup.sh

install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

test:
	@echo "🧪 Running setup test..."
	python test_setup.py

# Development commands
run:
	@echo "Starting RAG Conversational AI Assistant..."
	@echo "Starting API server in background..."
	python src/api/main.py &
	@sleep 5
	@echo "Starting frontend..."
	streamlit run src/frontend/streamlit_app.py

run-dev:
	@echo "Starting in development mode..."
	@echo "Make sure to set DEBUG=true in your .env file"
	python src/api/main.py &
	streamlit run src/frontend/streamlit_app.py --server.runOnSave=true

run-api:
	@echo "Starting API server..."
	python src/api/main.py

run-frontend:
	@echo "Starting frontend..."
	streamlit run src/frontend/streamlit_app.py

# Docker commands
docker-up:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

# Code quality
lint:
	@echo "Running linting..."
	flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/ -v -m "not slow and not integration"

test-api:
	@echo "🧪 Running API tests..."
	pytest tests/test_api.py -v

test-all:
	@echo "🧪 Running all tests..."
	pytest tests/ -v

# Deployment
deploy-azure:
	@echo "Deploying to Azure App Service..."
	./scripts/deploy-azure.sh

# Maintenance
clean:
	@echo "🧹 Cleaning up temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf temp/*
	rm -rf logs/*

reset:
	@echo "Resetting vector database..."
	rm -rf chroma_db/*
	@echo "Vector database reset complete"

# Environment setup
env:
	@echo "Setting up environment file..."
	cp env.example .env
	@echo "Please edit .env file with your configuration"

# Requirements management
freeze:
	@echo "Freezing requirements..."
	pip freeze > requirements-freeze.txt

# Health check
health:
	@echo "Checking system health..."
	curl -f http://localhost:8000/health || echo "API not running"
	curl -f http://localhost:8501/_stcore/health || echo "Frontend not running"

# Documentation
docs:
	@echo "Opening documentation..."
	@echo "API Documentation: http://localhost:8000/docs"
	@echo "Frontend: http://localhost:8501"
	@echo "Project Documentation: see README.md and docs/ folder"
