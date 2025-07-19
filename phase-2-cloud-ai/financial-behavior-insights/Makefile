# MLOps Workflow Makefile
# Provides convenient commands for running the ML workflow

.PHONY: help install train test deploy workflow clean

# Default target
help:
	@echo "MLOps Workflow Commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make train           - Train model only"
	@echo "  make test            - Test existing model locally"
	@echo "  make deploy          - Deploy to Azure (with approval)"
	@echo "  make workflow        - Run complete workflow"
	@echo "  make workflow-dev    - Run development workflow (no Azure)"
	@echo "  make workflow-ci     - Run CI workflow (auto-approve)"
	@echo "  make clean           - Clean artifacts and logs"
	@echo "  make check-env       - Check environment setup"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install azure-ai-ml azure-identity pyyaml

# Train model only
train:
	python3 workflows/ml_workflow.py --skip-local-testing --skip-azure-deployment

# Test existing model locally
test:
	python3 workflows/ml_workflow.py --skip-training --skip-azure-deployment

# Deploy to Azure with user approval
deploy:
	python3 workflows/ml_workflow.py --skip-training --skip-local-testing

# Run complete workflow with user approval
workflow:
	python3 workflows/ml_workflow.py --config workflows/config.yaml

# Development workflow (no Azure deployment)
workflow-dev:
	python3 workflows/ml_workflow.py --skip-azure-deployment

# CI/CD workflow (auto-approve Azure deployment)
workflow-ci:
	python3 workflows/ml_workflow.py --auto-approve

# Blue-green deployment
blue-green-deploy:
	python3 workflows/blue_green_deploy.py --model-uri models:/financial-behavior-model@production

# Clean artifacts and logs
clean:
	rm -rf workflow_artifacts/
	rm -f workflow.log
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Check environment setup
check-env:
	@echo "Checking Python environment..."
	@python3 --version
	@echo "Checking required packages..."
	@python3 -c "import mlflow, sklearn, pandas, numpy; print('✓ Core packages available')"
	@echo "Checking Azure CLI..."
	@az --version > /dev/null 2>&1 && echo "✓ Azure CLI available" || echo "✗ Azure CLI not found"
	@echo "Checking environment variables..."
	@python3 -c "import os; print('✓ Azure config:' if all([os.getenv('AZURE_SUBSCRIPTION_ID'), os.getenv('AZURE_RESOURCE_GROUP'), os.getenv('AZURE_WORKSPACE_NAME')]) else '✗ Missing Azure environment variables')"
	@echo "Checking data files..."
	@test -f data/processed/Comprehensive_Banking_Database_processed.csv && echo "✓ Data file available" || echo "✗ Data file not found"

# Quick validation
validate:
	python3 src/serving/test_local.py --skip-performance

# MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Show workflow status
status:
	@echo "Recent workflow artifacts:"
	@ls -la workflow_artifacts/ 2>/dev/null || echo "No workflow artifacts found"
	@echo "\nMLflow experiments:"
	@ls -la mlruns/ 2>/dev/null || echo "No MLflow runs found"

# Example workflows
example-local:
	@echo "Running example local workflow..."
	python3 workflows/ml_workflow.py --skip-azure-deployment --input-data data/processed/Comprehensive_Banking_Database_processed.csv

example-azure:
	@echo "Running example Azure workflow..."
	python3 workflows/ml_workflow.py --config workflows/config.yaml

# Development commands
dev-setup:
	make install
	make check-env
	@echo "Development environment ready!"

# Production deployment
prod-deploy:
	@echo "Deploying to production..."
	@read -p "Are you sure you want to deploy to production? (y/N): " confirm && [ "$$confirm" = "y" ]
	python3 workflows/ml_workflow.py --config workflows/config.yaml --auto-approve 