#!/bin/bash

# Azure App Service Optimization Script
# This script optimizes the application for Azure App Service deployment

set -e

echo "Optimizing application for Azure App Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists pip; then
    echo -e "${RED}ERROR: pip is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites check passed${NC}"

# Create optimized requirements file
echo -e "${YELLOW}Creating optimized requirements file...${NC}"
if [ -f "requirements-azure.txt" ]; then
    echo "Using existing requirements-azure.txt"
else
    echo -e "${RED}ERROR: requirements-azure.txt not found${NC}"
    exit 1
fi

# Install Azure-optimized dependencies
echo -e "${YELLOW}Installing Azure-optimized dependencies...${NC}"
pip install -r requirements-azure.txt --no-cache-dir

# Optimize Python bytecode
echo -e "${YELLOW}Optimizing Python bytecode...${NC}"
find . -name "*.py" -exec python -m py_compile {} \;

# Create optimized startup script
echo -e "${YELLOW}Creating optimized startup script...${NC}"
cat > start-azure-optimized.sh << 'EOF'
#!/bin/bash

# Azure App Service Optimized Startup Script
set -e

# Set Azure-specific environment variables
export PYTHONPATH="/app"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Memory optimization
export MALLOC_ARENA_MAX=2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072
export MALLOC_MMAP_MAX_=65536

# Create necessary directories
mkdir -p /app/temp /app/uploads /app/logs

# Set proper permissions
chmod -R 755 /app/temp /app/uploads /app/logs

# Download essential NLTK data if not present
if [ ! -d "/app/nltk_data" ]; then
    echo "📚 Downloading essential NLTK data..."
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
fi

# Start the application
exec python -m streamlit run src/web/app.py \
    --server.port="${PORT:-8000}" \
    --server.address="0.0.0.0" \
    --server.headless="true" \
    --server.enableCORS="false" \
    --server.enableXsrfProtection="false" \
    --server.maxUploadSize="200"
EOF

chmod +x start-azure-optimized.sh

# Create memory optimization script
echo -e "${YELLOW}Creating memory optimization script...${NC}"
cat > src/utils/memory_optimizer.py << 'EOF'
"""
Memory optimization utilities for Azure App Service.
"""

import gc
import psutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.initial_memory = psutil.virtual_memory().percent
    
    def check_memory_usage(self) -> float:
        """Check current memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is high."""
        return self.check_memory_usage() > self.max_memory_percent
    
    def optimize_memory(self) -> None:
        """Optimize memory usage."""
        if self.is_memory_high():
            logger.warning("High memory usage detected, optimizing...")
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            memory_usage = self.check_memory_usage()
            logger.info(f"Memory usage after optimization: {memory_usage:.2f}%")
    
    def get_memory_info(self) -> dict:
        """Get detailed memory information."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "free": memory.free
        }

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()

def optimize_memory_if_needed():
    """Optimize memory if usage is high."""
    memory_optimizer.optimize_memory()

def get_memory_status() -> dict:
    """Get current memory status."""
    return memory_optimizer.get_memory_info()
EOF

# Create Azure-specific configuration
echo -e "${YELLOW}Creating Azure-specific configuration...${NC}"
cat > src/config/azure_config.py << 'EOF'
"""
Azure-specific configuration for the application.
"""

import os
from typing import Dict, Any

class AzureConfig:
    """Azure-specific configuration."""
    
    @staticmethod
    def get_azure_settings() -> Dict[str, Any]:
        """Get Azure-specific settings."""
        return {
            "app_service": {
                "port": os.getenv("PORT", "8000"),
                "region": os.getenv("WEBSITE_SITE_NAME", "unknown"),
                "instance_id": os.getenv("WEBSITE_INSTANCE_ID", "unknown"),
                "resource_group": os.getenv("WEBSITE_RESOURCE_GROUP", "unknown")
            },
            "storage": {
                "account_name": os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
                "account_key": os.getenv("AZURE_STORAGE_ACCOUNT_KEY"),
                "container_name": os.getenv("AZURE_STORAGE_CONTAINER_NAME", "documents")
            },
            "openai": {
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
            },
            "monitoring": {
                "application_insights": os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
                "log_level": os.getenv("LOG_LEVEL", "INFO")
            }
        }
    
    @staticmethod
    def is_azure_environment() -> bool:
        """Check if running in Azure environment."""
        return os.getenv("WEBSITE_SITE_NAME") is not None
    
    @staticmethod
    def get_optimized_settings() -> Dict[str, Any]:
        """Get optimized settings for Azure."""
        return {
            "max_workers": 1,  # Azure App Service single instance
            "max_connections": 10,
            "timeout": 30,
            "memory_limit": "1G",
            "cpu_limit": "1000m"
        }
EOF

# Create performance monitoring script
echo -e "${YELLOW}Creating performance monitoring script...${NC}"
cat > src/utils/performance_monitor.py << 'EOF'
"""
Performance monitoring utilities for Azure App Service.
"""

import time
import psutil
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    @staticmethod
    def monitor_performance(func: Callable) -> Callable:
        """Decorator to monitor function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_memory = psutil.virtual_memory().used
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                logger.info(f"Function {func.__name__} executed in {execution_time:.2f}s, memory delta: {memory_delta / 1024 / 1024:.2f}MB")
        
        return wrapper
    
    @staticmethod
    def get_system_metrics() -> dict:
        """Get current system metrics."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "used": disk.used,
                "percent": (disk.used / disk.total) * 100
            }
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    return performance_monitor.monitor_performance(func)
EOF

# Create Azure-specific startup optimization
echo -e "${YELLOW}Creating Azure startup optimization...${NC}"
cat > src/utils/azure_startup.py << 'EOF'
"""
Azure App Service startup optimization.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AzureStartupOptimizer:
    """Azure startup optimization utilities."""
    
    @staticmethod
    def optimize_startup() -> None:
        """Optimize application startup for Azure."""
        logger.info("Optimizing startup for Azure App Service...")
        
        # Set optimal environment variables
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        os.environ.setdefault("MALLOC_ARENA_MAX", "2")
        
        # Log Azure environment info
        logger.info(f"Azure App Service: {os.getenv('WEBSITE_SITE_NAME', 'Unknown')}")
        logger.info(f"Instance ID: {os.getenv('WEBSITE_INSTANCE_ID', 'Unknown')}")
        logger.info(f"Port: {os.getenv('PORT', '8000')}")
    
    @staticmethod
    def get_azure_info() -> Dict[str, Any]:
        """Get Azure environment information."""
        return {
            "site_name": os.getenv("WEBSITE_SITE_NAME"),
            "instance_id": os.getenv("WEBSITE_INSTANCE_ID"),
            "resource_group": os.getenv("WEBSITE_RESOURCE_GROUP"),
            "port": os.getenv("PORT", "8000"),
            "region": os.getenv("WEBSITE_SITE_NAME", "unknown").split("-")[0] if os.getenv("WEBSITE_SITE_NAME") else "unknown"
        }

# Initialize startup optimization
azure_startup = AzureStartupOptimizer()
azure_startup.optimize_startup()
EOF

# Create .dockerignore for Azure optimization
echo -e "${YELLOW}🐳 Creating optimized .dockerignore...${NC}"
cat > .dockerignore << 'EOF'
# Git
.git
.gitignore
README.md
*.md

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Development files
tests/
docs/
*.test.py
test_*.py
*_test.py

# Temporary files
temp/
tmp/
*.tmp
*.temp

# Logs
logs/
*.log

# Database
*.db
*.sqlite
*.sqlite3

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Azure specific
.azure/
azure-pipelines.yml
EOF

# Create Azure-specific requirements optimization
echo -e "${YELLOW}Optimizing requirements for Azure...${NC}"
cat > requirements-azure-optimized.txt << 'EOF'
# Azure App Service Optimized Requirements (Minimal)
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database (Azure PostgreSQL)
sqlalchemy==2.0.23
psycopg2-binary==2.9.9

# AI/ML (Essential only)
openai==1.3.7
azure-ai-documentintelligence==1.0.0b4
spacy==3.7.2
transformers==4.36.2
torch==2.1.1
scikit-learn==1.3.2
nltk==3.8.1

# Document Processing (Essential)
PyPDF2==3.0.1
pdfplumber==0.10.3
python-docx==1.1.0
openpyxl==3.1.2
python-magic==0.4.27
pytesseract==0.3.10
Pillow==10.1.0

# Web Interface
streamlit==1.28.2
plotly==5.17.0
pandas==2.1.4
numpy==1.25.2

# HTTP
httpx==0.25.2
requests==2.31.0

# Configuration
python-dotenv==1.0.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Monitoring
structlog==23.2.0
psutil==5.9.6

# Azure
azure-storage-blob==12.19.0
azure-identity==1.15.0

# Production
gunicorn==21.2.0
EOF

# Create Azure deployment checklist
echo -e "${YELLOW}Creating Azure deployment checklist...${NC}"
cat > AZURE_DEPLOYMENT_CHECKLIST.md << 'EOF'
# Azure Deployment Checklist

## Pre-deployment
- [ ] Azure CLI installed and configured
- [ ] Docker installed and running
- [ ] Azure subscription active
- [ ] Required permissions (Owner/Contributor)
- [ ] Environment variables configured
- [ ] Azure resources created (Database, Storage, etc.)

## Deployment
- [ ] Resource group created
- [ ] App Service plan created
- [ ] Container Registry created
- [ ] Docker image built and pushed
- [ ] Web app created
- [ ] App settings configured
- [ ] Health check configured
- [ ] Logging enabled

## Post-deployment
- [ ] Application accessible via URL
- [ ] Health check endpoint working
- [ ] Azure OpenAI integration working
- [ ] Azure Storage integration working
- [ ] Database connection working
- [ ] Monitoring configured
- [ ] Alerts configured

## Testing
- [ ] File upload functionality
- [ ] Document analysis functionality
- [ ] Error handling
- [ ] Performance under load
- [ ] Memory usage monitoring
- [ ] Log analysis

## Security
- [ ] HTTPS enabled
- [ ] Secrets stored in Azure Key Vault
- [ ] Network security configured
- [ ] Access controls implemented
- [ ] Data encryption enabled

## Monitoring
- [ ] Application Insights configured
- [ ] Custom metrics implemented
- [ ] Alerts configured
- [ ] Log aggregation working
- [ ] Performance monitoring active
EOF

echo -e "${GREEN}Azure optimization completed successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Review the optimized configuration files"
echo "2. Test the application locally with Azure settings"
echo "3. Run the deployment script: ./scripts/deploy-azure.sh"
echo "4. Monitor the application using the health check endpoint"
echo ""
echo -e "${GREEN}Optimization complete!${NC}"
