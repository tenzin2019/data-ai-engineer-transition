# MLOps Workflow Makefile
# Provides convenient commands for running the ML workflow

.PHONY: help install train retrain test deploy workflow clean data-prep full-pipeline

# Default target
help:
	@echo "MLOps Workflow Commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make create-env       - Create conda environment"
	@echo "  make update-env       - Update conda environment"
	@echo "  make data-prep        - Prepare and preprocess data"
	@echo "  make train           - Train model with original sklearn version"
	@echo "  make retrain         - Retrain model with compatible sklearn version"
	@echo "  make test            - Test all deployments"
	@echo "  make deploy          - Deploy model to Azure ML"
	@echo "  make register        - Register model in MLflow and Azure ML"
	@echo "  make status          - Check deployment status"
	@echo "  make logs            - Get deployment logs"
	@echo "  make clean           - Clean artifacts and logs"
	@echo "  make check-env       - Check environment setup"
	@echo "  make full-pipeline   - Complete end-to-end workflow"
	@echo "  make validate        - Validate deployment"

# Install dependencies (updated with specific versions)
install:
	pip install -r requirements.txt

# Create conda environment with consistent versions
create-env:
	conda env create -f environment.yml

# Update current environment to match requirements
update-env:
	conda env update -f environment.yml

# Prepare and preprocess data
data-prep:
	@echo "Preparing and preprocessing data..."
	python3 src/data/preprocess_banking.py --input data/Banking-Dataset/Comprehensive_Banking_Database.csv --output data/processed/

# Train model with original sklearn version
train:
	@echo "Training model with original sklearn version..."
	python3 src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv --output-dir outputs/

# Retrain model with compatible sklearn version for Azure ML
retrain:
	@echo "Retraining model with compatible sklearn version..."
	python3 retrain_compatible_model.py

# Test all deployments
test:
	@echo "Testing deployments..."
	python3 test_deployments.py

# Deploy model to Azure ML
deploy:
	@echo "Deploying model to Azure ML..."
	python3 src/serving/deploy_manager.py --action deploy --model-name financial-behavior-model-fixed

# Register model in MLflow and Azure ML
register:
	@echo "Registering model..."
	python3 src/utils/model_registry.py --action register --model-path ./outputs --model-name financial-behavior-model

# Check deployment status
status:
	@echo "Checking deployment status..."
	python3 src/serving/deploy_manager.py --action status

# Get deployment logs
logs:
	@echo "Getting deployment logs..."
	az ml online-deployment get-logs --endpoint-name fin-behavior-ep-fixed --name blue --resource-group rg-data-ai-eng-con --workspace-name mlw-finance-phase-2 --lines 50

# Complete end-to-end workflow (data prep, train, retrain, deploy, test)
full-pipeline:
	@echo "Starting complete end-to-end workflow..."
	python3 workflow_runner.py --full-pipeline

# Run workflow with Python runner (more robust)
workflow-runner:
	@echo "Running workflow with Python runner..."
	python3 workflow_runner.py --full-pipeline

# Legacy workflow (train, register, deploy, test)
workflow:
	@echo "Running legacy workflow..."
	make train
	make register
	make deploy
	make test

# Clean artifacts and logs
clean:
	@echo "Cleaning artifacts and logs..."
	rm -rf workflow_artifacts/
	rm -f workflow.log
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf outputs/
	rm -rf simple_model/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Check environment setup
check-env:
	@echo "Checking Python environment..."
	@python3 --version
	@echo "Checking required packages..."
	@python3 -c "import mlflow, sklearn, pandas, numpy; print('✓ Core packages available')"
	@echo "Checking scikit-learn version..."
	@python3 -c "import sklearn; print(f'✓ scikit-learn version: {sklearn.__version__}')"
	@echo "Checking Azure CLI..."
	@az --version > /dev/null 2>&1 && echo "✓ Azure CLI available" || echo "✗ Azure CLI not found"
	@echo "Checking environment variables..."
	@python3 -c "import os; print('✓ Azure config:' if all([os.getenv('AZURE_SUBSCRIPTION_ID'), os.getenv('AZURE_RESOURCE_GROUP'), os.getenv('AZURE_WORKSPACE_NAME')]) else '✗ Missing Azure environment variables')"
	@echo "Checking data files..."
	@test -f data/processed/Comprehensive_Banking_Database_processed.csv && echo "✓ Data file available" || echo "✗ Data file not found"

# Quick validation
validate:
	@echo "Validating deployment..."
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
	@echo "\nModel outputs:"
	@ls -la outputs/ 2>/dev/null || echo "No model outputs found"

# Development commands
dev-setup:
	@echo "Setting up development environment..."
	make install
	make check-env
	@echo "Development environment ready!"

# Production deployment
prod-deploy:
	@echo "Deploying to production..."
	@read -p "Are you sure you want to deploy to production? (y/N): " confirm && [ "$$confirm" = "y" ]
	make full-pipeline

# Quick deployment (assumes model is already trained)
quick-deploy:
	@echo "Quick deployment (assumes model is ready)..."
	make deploy
	make test

# Troubleshooting commands
troubleshoot:
	@echo "Running troubleshooting checks..."
	@echo "1. Environment check..."
	make check-env
	@echo "2. Model files check..."
	@ls -la outputs/ 2>/dev/null || echo "No outputs directory found"
	@echo "3. Azure ML connection test..."
	python3 -c "from azure.ai.ml import MLClient; from azure.identity import DefaultAzureCredential; print('✓ Azure ML connection successful')" 2>/dev/null || echo "✗ Azure ML connection failed"

# Reset everything and start fresh
reset:
	@echo "Resetting everything..."
	make clean
	@echo "Reset complete. Run 'make full-pipeline' to start fresh." 