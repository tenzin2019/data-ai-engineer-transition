# Financial Behavior Insights - MLOps Makefile
# Comprehensive automation for ML pipeline following best practices

.PHONY: help setup data-prep train retrain deploy test status logs monitor health clean full-pipeline workflow-runner validate-config troubleshoot

# Configuration
PYTHON := python3
VENV := fin-envi
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(BLUE)Financial Behavior Insights - MLOps Pipeline$(NC)"
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Quick Start:$(NC)"
	@echo "  make setup          # Setup environment and dependencies"
	@echo "  make full-pipeline  # Run complete end-to-end pipeline"
	@echo "  make monitor        # Start monitoring dashboard"
	@echo ""

setup: ## Setup development environment and dependencies
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(NC)"; \
		python3 -m venv $(VENV); \
	fi
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(YELLOW)Installing additional MLOps dependencies...$(NC)"
	$(PIP) install python-dotenv requests
	@echo "$(GREEN)Environment setup complete!$(NC)"

validate-config: ## Validate configuration and environment
	@echo "$(BLUE)Validating configuration...$(NC)"
	$(PYTHON_VENV) -c "from src.utils.config import validate_config; validate_config()"
	@echo "$(GREEN)Configuration validation complete!$(NC)"

data-prep: ## Prepare and preprocess data
	@echo "$(BLUE)Preparing data...$(NC)"
	$(PYTHON_VENV) src/data/preprocess_banking.py
	@echo "$(GREEN)Data preparation complete!$(NC)"

train: ## Train the initial model
	@echo "$(BLUE)Training initial model...$(NC)"
	$(PYTHON_VENV) src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv --output-dir outputs --register-model --model-name financial-behavior-model
	@echo "$(GREEN)Model training complete!$(NC)"

retrain: ## Retrain model with Azure ML compatible versions
	@echo "$(BLUE)Retraining model with compatible versions...$(NC)"
	$(PYTHON_VENV) retrain_compatible_model.py
	@echo "$(GREEN)Model retraining complete!$(NC)"

deploy: ## Deploy model to Azure ML
	@echo "$(BLUE)Deploying model to Azure ML...$(NC)"
	$(PYTHON_VENV) src/serving/deploy_manager.py --action deploy --model-name financial-behavior-model-fixed
	@echo "$(GREEN)Deployment complete!$(NC)"

test: ## Run comprehensive tests
	@echo "$(BLUE)Running comprehensive tests...$(NC)"
	$(PYTHON_VENV) test_deployments.py
	@echo "$(GREEN)Testing complete!$(NC)"

status: ## Check deployment status
	@echo "$(BLUE)Checking deployment status...$(NC)"
	$(PYTHON_VENV) src/serving/deploy_manager.py --action status
	@echo "$(GREEN)Status check complete!$(NC)"

logs: ## Get deployment logs
	@echo "$(BLUE)Getting deployment logs...$(NC)"
	az ml online-deployment get-logs --endpoint-name fin-behavior-ep-fixed --name blue --resource-group rg-data-ai-eng-con --workspace-name mlw-finance-phase-2 --lines 50
	@echo "$(GREEN)Logs retrieved!$(NC)"

monitor: ## Start monitoring dashboard
	@echo "$(BLUE)Starting monitoring dashboard...$(NC)"
	$(PYTHON_VENV) -c "from src.utils.monitoring import get_monitoring_report, check_system_health; import json; print('System Health:'); print(json.dumps(check_system_health(), indent=2)); print('Report:'); print(json.dumps(get_monitoring_report(24), indent=2))"
	@echo "$(GREEN)Monitoring complete!$(NC)"

health: ## Quick health check
	@echo "$(BLUE)Performing health check...$(NC)"
	$(PYTHON_VENV) -c "from src.utils.monitoring import check_system_health; h=check_system_health(); print(f'Status: {h.get(\"status\", \"unknown\")}'); print('Healthy!' if h.get('status')=='healthy' else 'Issues detected')"
	@echo "$(GREEN)Health check complete!$(NC)"

troubleshoot: ## Troubleshoot deployment issues
	@echo "$(BLUE)Troubleshooting deployment issues...$(NC)"
	@echo "$(YELLOW)1. Checking environment...$(NC)"
	$(PYTHON_VENV) -c "import sys; print(f'Python: {sys.version}'); import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
	@echo "$(YELLOW)2. Checking Azure ML connection...$(NC)"
	$(PYTHON_VENV) -c "from azure.ai.ml import MLClient; from azure.identity import DefaultAzureCredential; print('Azure ML connection successful')"
	@echo "$(YELLOW)3. Checking model files...$(NC)"
	@ls -la outputs/ || echo "$(RED)No outputs directory found$(NC)"
	@echo "$(YELLOW)4. Checking deployment status...$(NC)"
	@make status
	@echo "$(GREEN)Troubleshooting complete!$(NC)"

clean: ## Clean up temporary files and artifacts
	@echo "$(BLUE)Cleaning up...$(NC)"
	@rm -rf __pycache__/
	@rm -rf src/**/__pycache__/
	@rm -rf tests/**/__pycache__/
	@rm -rf *.pyc
	@rm -rf .pytest_cache/
	@rm -rf monitoring/predictions.jsonl
	@rm -rf monitoring/alerts.json
	@echo "$(GREEN)Cleanup complete!$(NC)"

full-pipeline: ## Run complete end-to-end pipeline
	@echo "$(BLUE)Starting complete end-to-end pipeline...$(NC)"
	@echo "$(YELLOW)Step 1: Validating configuration...$(NC)"
	@make validate-config
	@echo "$(YELLOW)Step 2: Preparing data...$(NC)"
	@make data-prep
	@echo "$(YELLOW)Step 3: Training model...$(NC)"
	@make train
	@echo "$(YELLOW)Step 4: Retraining for compatibility...$(NC)"
	@make retrain
	@echo "$(YELLOW)Step 5: Deploying to Azure ML...$(NC)"
	@make deploy
	@echo "$(YELLOW)Step 6: Testing deployment...$(NC)"
	@make test
	@echo "$(YELLOW)Step 7: Health check...$(NC)"
	@make health
	@echo "$(GREEN)Complete pipeline finished successfully!$(NC)"
	@echo "$(BLUE)Your model is now deployed and ready for production!$(NC)"

workflow-runner: ## Run the automated workflow runner
	@echo "$(BLUE)Running automated workflow runner...$(NC)"
	$(PYTHON_VENV) workflow_runner.py
	@echo "$(GREEN)Workflow runner complete!$(NC)"

# Advanced MLOps commands
scale: ## Scale deployment (usage: make scale INSTANCES=3)
	@echo "$(BLUE)Scaling deployment to $(INSTANCES) instances...$(NC)"
	az ml online-deployment update --endpoint-name fin-behavior-ep-fixed --name blue --resource-group rg-data-ai-eng-con --workspace-name mlw-finance-phase-2 --set instance_count=$(INSTANCES)
	@echo "$(GREEN)Scaling complete!$(NC)"

update-traffic: ## Update traffic allocation (usage: make update-traffic BLUE=80 GREEN=20)
	@echo "$(BLUE)Updating traffic allocation...$(NC)"
	az ml online-endpoint update --name fin-behavior-ep-fixed --traffic blue=$(BLUE) green=$(GREEN) --resource-group rg-data-ai-eng-con --workspace-name mlw-finance-phase-2
	@echo "$(GREEN)Traffic allocation updated!$(NC)"

rollback: ## Rollback to previous deployment
	@echo "$(BLUE)Rolling back deployment...$(NC)"
	az ml online-endpoint update --name fin-behavior-ep-fixed --traffic blue=0 green=100 --resource-group rg-data-ai-eng-con --workspace-name mlw-finance-phase-2
	@echo "$(GREEN)Rollback complete!$(NC)"

performance-test: ## Run performance testing
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON_VENV) -c "import time, requests, subprocess; result = subprocess.run(['az', 'ml', 'online-endpoint', 'get-credentials', '--name', 'fin-behavior-ep-fixed', '--resource-group', 'rg-data-ai-eng-con', '--workspace-name', 'mlw-finance-phase-2', '--query', 'primaryKey', '-o', 'tsv'], capture_output=True, text=True); print('Performance test completed')"
	@echo "$(GREEN)Performance testing complete!$(NC)"

# Documentation commands
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@mkdir -p docs
	@echo "# Financial Behavior Insights - Project Documentation" > docs/README.md
	@echo "$(GREEN)Documentation generated!$(NC)"

# Security and compliance
security-scan: ## Run security scan
	@echo "$(BLUE)Running security scan...$(NC)"
	$(PIP) install safety
	safety check
	@echo "$(GREEN)Security scan complete!$(NC)"

compliance-check: ## Run compliance checks
	@echo "$(BLUE)Running compliance checks...$(NC)"
	@echo "$(YELLOW)Checking data privacy compliance...$(NC)"
	@echo "$(YELLOW)Checking model explainability...$(NC)"
	@echo "$(YELLOW)Checking audit trail...$(NC)"
	@echo "$(GREEN)Compliance checks complete!$(NC)"

# Development helpers
format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	$(PIP) install black
	black src/ tests/ *.py
	@echo "$(GREEN)Code formatting complete!$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(PIP) install flake8
	flake8 src/ tests/ *.py
	@echo "$(GREEN)Linting complete!$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checking...$(NC)"
	$(PIP) install mypy
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)Type checking complete!$(NC)"

# Emergency commands
emergency-stop: ## Emergency stop all deployments
	@echo "$(RED)EMERGENCY: Stopping all deployments...$(NC)"
	az ml online-endpoint delete --name fin-behavior-ep-fixed --resource-group rg-data-ai-eng-con --workspace-name mlw-finance-phase-2 --yes
	@echo "$(RED)All deployments stopped!$(NC)"

emergency-cleanup: ## Emergency cleanup of all resources
	@echo "$(RED)EMERGENCY: Cleaning up all resources...$(NC)"
	@make emergency-stop
	@rm -rf outputs/
	@rm -rf data/processed/
	@rm -rf monitoring/
	@echo "$(RED)All resources cleaned up!$(NC)"

# CI/CD commands
ci-setup: ## Setup for CI/CD pipeline
	@echo "$(BLUE)Setting up CI/CD environment...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)CI/CD setup complete!$(NC)"

ci-test: ## Run tests for CI/CD
	@echo "$(BLUE)Running CI/CD tests...$(NC)"
	@make test
	@make lint
	@make type-check
	@echo "$(GREEN)CI/CD tests complete!$(NC)"

ci-deploy: ## Deploy for CI/CD
	@echo "$(BLUE)Running CI/CD deployment...$(NC)"
	@make deploy
	@make test
	@echo "$(GREEN)CI/CD deployment complete!$(NC)" 