#!/bin/bash

# RAG Conversational AI Assistant - Azure App Service Deployment Script
# Based on successful patterns from intelligent-document-analysis deployment
# This script deploys the RAG system to Azure App Service

set -e

echo "🚀 Starting Azure App Service Deployment for RAG Conversational AI Assistant..."

# Configuration (using proven settings from successful deployment)
APP_NAME="rag-conversational-ai-assistant"
RESOURCE_GROUP="rg-data-ai-eng-con"  # Use same resource group as successful deployment
LOCATION="australiaeast"  # Use same location as successful deployment
PLAN_NAME="plan-rag-assistant"
REGISTRY_NAME="1a27253794c8488f83ef31437e7d1248"  # Use same registry as successful deployment
SKU="P1V2"  # Change to FREE for development

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Deployment Configuration:${NC}"
echo "App Name: $APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Plan: $PLAN_NAME"
echo "SKU: $SKU"
echo "Registry: $REGISTRY_NAME"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}ERROR: Azure CLI is not installed. Please install it first:${NC}"
    echo "https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed. Please install it first:${NC}"
    echo "https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Please log in to Azure:${NC}"
    az login
fi

echo -e "${GREEN}Azure CLI is ready${NC}"

# Check if resource group exists
echo -e "${YELLOW}Checking resource group...${NC}"
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}Resource group $RESOURCE_GROUP already exists${NC}"
else
    echo -e "${YELLOW}Creating resource group...${NC}"
    az group create --name $RESOURCE_GROUP --location $LOCATION
fi

# Create App Service plan
echo -e "${YELLOW}Creating App Service plan...${NC}"
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku $SKU \
    --is-linux

# Check if Container Registry exists (using existing registry from successful deployment)
echo -e "${YELLOW}🐳 Checking Container Registry...${NC}"
if az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}Container Registry $REGISTRY_NAME already exists${NC}"
else
    echo -e "${YELLOW}Creating Container Registry...${NC}"
    az acr create \
        --name $REGISTRY_NAME \
        --resource-group $RESOURCE_GROUP \
        --sku Basic \
        --admin-enabled true
fi

# Login to ACR
echo -e "${YELLOW}Logging into Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

# Build and push Docker image with correct platform (using Azure-optimized Dockerfile)
echo -e "${YELLOW}Building Docker image for linux/amd64...${NC}"
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.azure \
    -t $REGISTRY_NAME.azurecr.io/$APP_NAME:latest \
    --push .

# Create web app
echo -e "${YELLOW}Creating web app...${NC}"
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --deployment-container-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME:latest

# Get ACR credentials
echo -e "${YELLOW}Getting Container Registry credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value --output tsv)

# Configure app settings (using proven patterns from successful deployment)
echo -e "${YELLOW}Configuring app settings...${NC}"
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DOCKER_REGISTRY_SERVER_URL=https://$REGISTRY_NAME.azurecr.io \
        DOCKER_REGISTRY_SERVER_USERNAME=$ACR_USERNAME \
        DOCKER_REGISTRY_SERVER_PASSWORD=$ACR_PASSWORD \
        WEBSITES_PORT=8000 \
        WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
        WEBSITES_CONTAINER_START_TIME_LIMIT=1800 \
        PYTHONPATH=/app \
        API_HOST=0.0.0.0 \
        API_PORT=8000 \
        ENVIRONMENT=production \
        DEBUG=false \
        LOG_LEVEL=INFO \
        DATABASE_URL="sqlite:///./rag_assistant.db" \
        CHROMA_PERSIST_DIRECTORY=./chroma_db

# Configure continuous deployment
echo -e "${YELLOW}Configuring continuous deployment...${NC}"
az webapp config container set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME:latest

# Enable logging
echo -e "${YELLOW}Enabling application logging...${NC}"
az webapp log config \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --application-logging filesystem \
    --level information

# Configure health check (using same pattern as successful deployment)
echo -e "${YELLOW}Configuring health check...${NC}"
az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --startup-file "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1"

# Get app URL
APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

echo -e "${GREEN}Azure App Service created successfully!${NC}"
echo ""
echo -e "${GREEN}Your app is available at:${NC}"
echo "https://$APP_URL"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Set your AI service API keys in the App Service configuration:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings OPENAI_API_KEY=your-key"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings ANTHROPIC_API_KEY=your-key"
echo ""
echo "2. Set your Azure OpenAI configuration:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_OPENAI_ENDPOINT=your-endpoint AZURE_OPENAI_API_KEY=your-key"
echo ""
echo "3. Monitor your application:"
echo "   az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "4. View application health:"
echo "   https://$APP_URL/health"
echo ""
echo -e "${GREEN}Deployment script completed!${NC}"
echo ""
echo -e "${YELLOW}🔍 Running deployment verification...${NC}"
if [ -f "./scripts/verify-deployment.sh" ]; then
    ./scripts/verify-deployment.sh
else
    echo -e "${YELLOW}⚠️ Verification script not found. Please run manually:${NC}"
    echo "curl -f https://$APP_URL/health"
fi
