#!/bin/bash

# Azure App Service Startup Script
# This script optimizes the application for Azure App Service deployment

set -e

echo "Starting Intelligent Document Analysis on Azure App Service..."

# Set Azure-specific environment variables
export PYTHONPATH="/app"
export STREAMLIT_SERVER_PORT="${PORT:-8000}"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_SERVER_ENABLE_CORS="false"
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="false"
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE="200"

# Create necessary directories
mkdir -p /app/temp /app/uploads /app/logs

# Set proper permissions
chmod -R 755 /app/temp /app/uploads /app/logs

# Download essential NLTK data if not present
if [ ! -d "/app/nltk_data" ]; then
    echo "📚 Downloading essential NLTK data..."
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
fi

# Health check will be handled by Streamlit app's built-in health endpoint

# Start the main application
echo "Starting Streamlit application..."
exec python -m streamlit run src/web/app.py \
    --server.port="${STREAMLIT_SERVER_PORT}" \
    --server.address="${STREAMLIT_SERVER_ADDRESS}" \
    --server.headless="${STREAMLIT_SERVER_HEADLESS}" \
    --server.enableCORS="${STREAMLIT_SERVER_ENABLE_CORS}" \
    --server.enableXsrfProtection="${STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION}" \
    --server.maxUploadSize="${STREAMLIT_SERVER_MAX_UPLOAD_SIZE}"
