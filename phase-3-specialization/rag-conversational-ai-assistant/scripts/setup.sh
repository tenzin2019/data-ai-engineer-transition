#!/bin/bash

# RAG Conversational AI Assistant - Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up RAG Conversational AI Assistant..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.11+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]]; then
            print_status "Python $PYTHON_VERSION found ✓"
        else
            print_error "Python 3.11+ is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Check if Node.js 18+ is installed
check_node() {
    print_status "Checking Node.js version..."
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v | cut -d'v' -f2)
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
        if [[ $NODE_MAJOR -ge 18 ]]; then
            print_status "Node.js $NODE_VERSION found ✓"
        else
            print_error "Node.js 18+ is required. Found: $NODE_VERSION"
            exit 1
        fi
    else
        print_error "Node.js is not installed"
        exit 1
    fi
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker..."
    if command -v docker &> /dev/null; then
        print_status "Docker found ✓"
    else
        print_warning "Docker not found. Some features may not work."
    fi
}
# Create virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "rag-venv" ]; then
        python3 -m venv rag-venv
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    source rag-venv/bin/activate
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install development dependencies
    print_status "Installing development dependencies..."
    pip install -e ".[dev]"
    
    print_status "Python environment setup complete ✓"
}

# Setup Node.js environment
setup_node_env() {
    print_status "Setting up Node.js environment..."
    
    if [ ! -d "node_modules" ]; then
        npm install
        print_status "Node.js dependencies installed"
    else
        print_status "Node.js dependencies already installed"
    fi
    
    print_status "Node.js environment setup complete ✓"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p logs
    mkdir -p data
    mkdir -p chroma_db
    mkdir -p config/nginx/ssl
    mkdir -p config/grafana/dashboards
    mkdir -p config/grafana/datasources
    
    print_status "Directories created ✓"
}

# Setup environment file
setup_env_file() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Please update .env file with your actual configuration values"
        else
            print_error ".env.example file not found"
            exit 1
        fi
    else
        print_status ".env file already exists"
    fi
}

# Setup pre-commit hooks
setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."
    
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_status "Pre-commit hooks installed ✓"
    else
        print_warning "Pre-commit not installed. Skipping hook setup."
    fi
}

# Setup database
setup_database() {
    print_status "Setting up database..."
    
    # Check if PostgreSQL is running
    if command -v pg_isready &> /dev/null; then
        if pg_isready -h localhost -p 5432 &> /dev/null; then
            print_status "PostgreSQL is running"
        else
            print_warning "PostgreSQL is not running. Please start it before running the application."
        fi
    else
        print_warning "PostgreSQL client not found. Please ensure PostgreSQL is installed and running."
    fi
}

# Main setup function
main() {
    echo "=========================================="
    echo "RAG Conversational AI Assistant Setup"
    echo "=========================================="
    
    check_python
    check_node
    check_docker
    
    create_directories
    setup_env_file
    setup_python_env
    setup_node_env
    setup_pre_commit
    setup_database
    
    echo ""
    echo "=========================================="
    echo "✅ Setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Update .env file with your configuration"
    echo "2. Start the database services: docker-compose up -d postgres redis"
    echo "3. Run database migrations: python scripts/migrate.py"
    echo "4. Start the development server: python src/api/main.py"
    echo "5. Start the frontend: npm run dev"
    echo ""
    echo "For more information, see the README.md file."
}

# Run main function
main "$@"
