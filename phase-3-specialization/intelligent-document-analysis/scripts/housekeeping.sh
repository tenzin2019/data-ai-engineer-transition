#!/bin/bash

# Housekeeping Script for Intelligent Document Analysis
# This script cleans up unused files and optimizes the project structure

set -e

echo "🧹 Starting housekeeping process..."

# Change to project directory
cd "$(dirname "$0")/.."

echo "📁 Current directory: $(pwd)"

# 1. Remove log files and temporary data
echo "🗑️ Cleaning up log files and temporary data..."
rm -rf LogFiles/
rm -rf logs/
rm -rf temp/*
rm -rf uploads/*
rm -f *.log
rm -f workflow.log

# 2. Remove zip files (app logs)
echo "🗑️ Removing app log zip files..."
rm -f app-logs*.zip

# 3. Remove Python cache files
echo "🗑️ Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# 4. Remove virtual environment (we're using Azure deployment)
echo "🗑️ Removing local virtual environment..."
rm -rf venv/

# 5. Remove old deployment scripts that are no longer needed
echo "🗑️ Removing obsolete deployment scripts..."
rm -f scripts/azure-deployment-fix.sh
rm -f scripts/fix-azure-deployment.sh
rm -f scripts/fix-database-connection.sh
rm -f scripts/fix-database-sql.sh
rm -f scripts/quick-fix-no-db.sh
rm -f scripts/redeploy-from-scratch.sh
rm -f scripts/redeploy-automated.sh

# 6. Remove old Docker files (keep only the Azure one)
echo "🗑️ Removing old Docker files..."
rm -f Dockerfile
rm -f docker-compose.yml
rm -f docker-compose.azure.yml

# 7. Remove old environment files
echo "🗑️ Removing old environment files..."
rm -f env.azure
rm -f env.example

# 8. Remove old database files
echo "🗑️ Removing old database files..."
rm -f mlflow.db
rm -f create_tables.sql
rm -f init_database.py

# 9. Remove old test files
echo "🗑️ Removing old test files..."
rm -f run_azure_tests.py
rm -f debug_analysis.py

# 10. Remove old documentation files (consolidate)
echo "🗑️ Removing redundant documentation..."
rm -f AZURE_DEPLOYMENT_CHECKLIST.md
rm -f AZURE_DEPLOYMENT_FIXES_SUMMARY.md
rm -f AZURE_DEPLOYMENT_FIXES.md
rm -f AZURE_DEPLOYMENT_SUMMARY.md
rm -f HOUSEKEEPING_SUMMARY.md
rm -f IMPLEMENTATION_GUIDE.md
rm -f PROJECT_SUMMARY.md

# 11. Clean up requirements files (keep only the essential ones)
echo "🗑️ Cleaning up requirements files..."
rm -f requirements-test.txt

# 12. Remove empty directories
echo "🗑️ Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true

# 13. Create a clean .gitignore
echo "📝 Creating clean .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

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

# Logs
*.log
logs/
LogFiles/

# Temporary files
temp/
uploads/
*.tmp
*.temp

# Database
*.db
*.sqlite
*.sqlite3

# MLflow
mlruns/
mlflow.db

# Azure
.azure/

# Environment variables
.env
.env.local
.env.*.local

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
EOF

echo "✅ Housekeeping completed successfully!"
echo ""
echo "📊 Summary of cleaned up items:"
echo "  - Log files and temporary data"
echo "  - Python cache files"
echo "  - Virtual environment"
echo "  - Obsolete deployment scripts"
echo "  - Old Docker files"
echo "  - Redundant documentation"
echo "  - Empty directories"
echo "  - Created clean .gitignore"
echo ""
echo "🎯 Project is now clean and optimized for production!"
