#!/bin/bash

# Azure App Service Startup Script
# This script optimizes the application for Azure App Service deployment

set -e

echo "🚀 Starting Intelligent Document Analysis on Azure App Service..."

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

# Health check endpoint
create_health_check() {
    cat > /app/health.py << 'EOF'
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'port': os.environ.get('PORT', '8000'),
        'environment': os.environ.get('ENVIRONMENT', 'production')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
EOF
}

# Start health check in background
echo "🏥 Starting health check endpoint..."
create_health_check
python /app/health.py &
HEALTH_PID=$!

# Wait for health check to be ready
sleep 5

# Start the main application
echo "🌐 Starting Streamlit application..."
exec python -m streamlit run src/web/app.py \
    --server.port="${STREAMLIT_SERVER_PORT}" \
    --server.address="${STREAMLIT_SERVER_ADDRESS}" \
    --server.headless="${STREAMLIT_SERVER_HEADLESS}" \
    --server.enableCORS="${STREAMLIT_SERVER_ENABLE_CORS}" \
    --server.enableXsrfProtection="${STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION}" \
    --server.maxUploadSize="${STREAMLIT_SERVER_MAX_UPLOAD_SIZE}"
