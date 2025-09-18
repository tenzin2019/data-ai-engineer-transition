#!/bin/bash

# Intelligent Document Analysis System - Startup Script

set -e

echo "Starting Intelligent Document Analysis System..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "WARNING: .env file not found. Creating from template..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "SUCCESS: Created .env file from template. Please update with your Azure credentials."
    else
        echo "ERROR: env.example file not found. Please create .env file manually."
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads temp logs data/sample_documents

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Docker is available. You can use docker-compose for full deployment."
    echo "   Run: docker-compose up -d"
else
    echo "WARNING: Docker not found. Running in development mode..."
fi

# Check Azure configuration
echo "Checking Azure configuration..."
if grep -q "your-azure-openai-api-key" .env; then
    echo "WARNING: Please update your Azure OpenAI API key in .env file"
fi

if grep -q "your-document-intelligence-api-key" .env; then
    echo "WARNING: Please update your Azure Document Intelligence API key in .env file"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To start the application:"
echo "1. Update your Azure credentials in .env file"
echo "2. Run: python -m streamlit run src/web/app.py"
echo ""
echo "Or use Docker:"
echo "1. Run: docker-compose up -d"
echo "2. Access the app at: http://localhost:8501"
echo ""
echo "📖 For more information, see README.md"
