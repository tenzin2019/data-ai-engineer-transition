#!/bin/bash

# RAG Conversational AI Assistant - Local Setup Script
set -e

echo "🤖 RAG Conversational AI Assistant - Local Setup"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed.${NC}"
    echo "Please install Python 3.11 or later from https://python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}ERROR: Python 3.11 or later is required. Current version: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Python $PYTHON_VERSION found${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker is not installed. Some features may not work.${NC}"
    echo "Install Docker from https://www.docker.com/products/docker-desktop"
else
    echo -e "${GREEN}Docker found${NC}"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker Compose is not installed. Some features may not work.${NC}"
    echo "Install Docker Compose from https://docs.docker.com/compose/install/"
else
    echo -e "${GREEN}Docker Compose found${NC}"
fi

echo ""
echo -e "${BLUE}Setting up the development environment...${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv rag-venv

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source rag-venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p uploads
mkdir -p chroma_db
mkdir -p logs
mkdir -p temp

# Copy environment file
echo -e "${YELLOW}Setting up environment variables...${NC}"
if [ ! -f .env ]; then
    cp env.example .env
    echo -e "${GREEN}Created .env file from template${NC}"
    echo -e "${YELLOW}Please edit .env file with your API keys and configuration${NC}"
else
    echo -e "${GREEN}.env file already exists${NC}"
fi

# Set permissions
chmod +x scripts/*.sh

echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Edit the .env file with your configuration:"
echo "   nano .env"
echo ""
echo "2. Activate the virtual environment:"
echo "   source rag-venv/bin/activate"
echo ""
echo "3. Start the development server:"
echo "   python src/api/main.py"
echo ""
echo "4. In another terminal, start the frontend:"
echo "   streamlit run src/frontend/streamlit_app.py"
echo ""
echo "5. Or use Docker Compose for full stack:"
echo "   docker-compose up -d"
echo ""
echo -e "${BLUE}Important configuration:${NC}"
echo "- Set your OpenAI API key: OPENAI_API_KEY"
echo "- Set your Anthropic API key: ANTHROPIC_API_KEY"
echo "- Configure Azure OpenAI if using: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"
echo ""
echo -e "${GREEN}Happy coding!${NC}"