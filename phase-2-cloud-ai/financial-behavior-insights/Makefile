# MLOps Workflow Makefile
# Provides convenient commands for running the ML workflow

.PHONY: help install train test deploy workflow clean

# Default target
help:
	@echo "MLOps Workflow Commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make create-env       - Create conda environment"
	@echo "  make update-env       - Update conda environment"
	@echo "  make train           - Train model only"
	@echo "  make test            - Test all deployments"
	@echo "  make deploy          - Deploy model to Azure ML"
	@echo "  make register        - Register model in MLflow and Azure ML"
	@echo "  make status          - Check deployment status"
	@echo "  make clean           - Clean artifacts and logs"
	@echo "  make check-env       - Check environment setup"

# Install dependencies (updated with specific versions)
install:
	pip install -r requirements.txt

# Create conda environment with consistent versions
create-env:
	conda env create -f environment.yml

# Update current environment to match requirements
update-env:
	conda env update -f environment.yml

# Train model only
train:
	python3 src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv --output-dir outputs/

# Test all deployments
test:
	python3 test_deployments.py

# Deploy model to Azure ML
deploy:
	python3 src/serving/deploy_manager.py --action deploy --model-name financial-behavior-model-optimized

# Register model in MLflow and Azure ML
register:
	python3 src/utils/model_registry.py --action register --model-path ./outputs --model-name financial-behavior-model

# Check deployment status
status:
	python3 src/serving/deploy_manager.py --action status

# Complete workflow (train, register, deploy, test)
workflow:
	make train
	make register
	make deploy
	make test

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
	python3 test_deployments.py

# MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Show workflow status
workflow-status:
	@echo "Recent workflow artifacts:"
	@ls -la workflow_artifacts/ 2>/dev/null || echo "No workflow artifacts found"
	@echo "\nMLflow experiments:"
	@ls -la mlruns/ 2>/dev/null || echo "No MLflow runs found"

# Development commands
dev-setup:
	make install
	make check-env
	@echo "Development environment ready!"

# Production deployment
prod-deploy:
	@echo "Deploying to production..."
	@read -p "Are you sure you want to deploy to production? (y/N): " confirm && [ "$$confirm" = "y" ]
	make workflow 