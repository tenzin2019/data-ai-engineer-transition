#!/bin/bash

# Azure App Service Deployment Script for RAG Conversational AI Assistant
set -e

echo "Starting Azure App Service Deployment..."

# Configuration
APP_NAME="rag-conversational-ai-assistant"
RESOURCE_GROUP="rg-rag-assistant"
LOCATION="eastus"
PLAN_NAME="plan-rag-assistant"
REGISTRY_NAME="acrragassistant"
SKU="P2V2"  # Premium for production workloads

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

# Create resource group
echo -e "${YELLOW}Creating resource group...${NC}"
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# Create App Service plan
echo -e "${YELLOW}Creating App Service plan...${NC}"
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku $SKU \
    --is-linux

# Create Container Registry
echo -e "${YELLOW}Creating Container Registry...${NC}"
az acr create \
    --name $REGISTRY_NAME \
    --resource-group $RESOURCE_GROUP \
    --sku Standard \
    --admin-enabled true

# Login to ACR
echo -e "${YELLOW}Logging into Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

# Build and push Docker images
echo -e "${YELLOW}Building and pushing Docker images...${NC}"

# Backend API
echo "Building backend API image..."
docker build \
    --platform linux/amd64 \
    -f Dockerfile \
    -t $REGISTRY_NAME.azurecr.io/$APP_NAME-api:latest \
    .

docker push $REGISTRY_NAME.azurecr.io/$APP_NAME-api:latest

# Frontend
echo "Building frontend image..."
docker build \
    --platform linux/amd64 \
    -f Dockerfile.frontend \
    -t $REGISTRY_NAME.azurecr.io/$APP_NAME-frontend:latest \
    .

docker push $REGISTRY_NAME.azurecr.io/$APP_NAME-frontend:latest

# Get ACR credentials
echo -e "${YELLOW}Getting Container Registry credentials...${NC}"
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value --output tsv)

# Create web app for API
echo -e "${YELLOW}Creating web app for API...${NC}"
az webapp create \
    --name $APP_NAME-api \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --deployment-container-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME-api:latest

# Configure API app settings
echo -e "${YELLOW}Configuring API app settings...${NC}"
az webapp config appsettings set \
    --name $APP_NAME-api \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DOCKER_REGISTRY_SERVER_URL=https://$REGISTRY_NAME.azurecr.io \
        DOCKER_REGISTRY_SERVER_USERNAME=$ACR_USERNAME \
        DOCKER_REGISTRY_SERVER_PASSWORD=$ACR_PASSWORD \
        WEBSITES_PORT=8000 \
        WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
        WEBSITES_CONTAINER_START_TIME_LIMIT=1800 \
        PYTHONPATH=/app \
        DEBUG=false \
        ENVIRONMENT=production \
        API_HOST=0.0.0.0 \
        API_PORT=8000

# Create web app for Frontend
echo -e "${YELLOW}Creating web app for Frontend...${NC}"
az webapp create \
    --name $APP_NAME-frontend \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --deployment-container-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME-frontend:latest

# Configure Frontend app settings
echo -e "${YELLOW}Configuring Frontend app settings...${NC}"
az webapp config appsettings set \
    --name $APP_NAME-frontend \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DOCKER_REGISTRY_SERVER_URL=https://$REGISTRY_NAME.azurecr.io \
        DOCKER_REGISTRY_SERVER_USERNAME=$ACR_USERNAME \
        DOCKER_REGISTRY_SERVER_PASSWORD=$ACR_PASSWORD \
        WEBSITES_PORT=8501 \
        WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
        WEBSITES_CONTAINER_START_TIME_LIMIT=1800 \
        API_BASE_URL=https://$APP_NAME-api.azurewebsites.net

# Create PostgreSQL database
echo -e "${YELLOW}Creating PostgreSQL database...${NC}"
az postgres flexible-server create \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME-db \
    --location $LOCATION \
    --admin-user dbadmin \
    --admin-password "RagAssistant2024!" \
    --sku-name Standard_B2s \
    --tier Burstable \
    --public-access 0.0.0.0 \
    --storage-size 32

# Create database
az postgres flexible-server db create \
    --resource-group $RESOURCE_GROUP \
    --server-name $APP_NAME-db \
    --database-name rag_assistant

# Create Redis cache
echo -e "${YELLOW}Creating Redis cache...${NC}"
az redis create \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME-redis \
    --location $LOCATION \
    --sku Standard \
    --vm-size c1

# Update API app with database connection
echo -e "${YELLOW}Updating API app with database connection...${NC}"
az webapp config appsettings set \
    --name $APP_NAME-api \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DATABASE_URL="postgresql://dbadmin:RagAssistant2024!@$APP_NAME-db.postgres.database.azure.com:5432/rag_assistant?sslmode=require" \
        REDIS_URL="rediss://$APP_NAME-redis.redis.cache.windows.net:6380/0?ssl_cert_reqs=required"

# Enable logging
echo -e "${YELLOW}Enabling application logging...${NC}"
az webapp log config \
    --name $APP_NAME-api \
    --resource-group $RESOURCE_GROUP \
    --application-logging filesystem \
    --level information

az webapp log config \
    --name $APP_NAME-frontend \
    --resource-group $RESOURCE_GROUP \
    --application-logging filesystem \
    --level information

# Get app URLs
API_URL=$(az webapp show --name $APP_NAME-api --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)
FRONTEND_URL=$(az webapp show --name $APP_NAME-frontend --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

echo -e "${GREEN}Azure App Service deployment completed successfully!${NC}"
echo ""
echo -e "${GREEN}Your applications are available at:${NC}"
echo "API: https://$API_URL"
echo "Frontend: https://$FRONTEND_URL"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Set your AI service API keys in the API App Service configuration:"
echo "   az webapp config appsettings set --name $APP_NAME-api --resource-group $RESOURCE_GROUP --settings OPENAI_API_KEY=your-key"
echo ""
echo "2. Monitor your applications:"
echo "   az webapp log tail --name $APP_NAME-api --resource-group $RESOURCE_GROUP"
echo "   az webapp log tail --name $APP_NAME-frontend --resource-group $RESOURCE_GROUP"
echo ""
echo "3. View application health:"
echo "   https://$API_URL/health"
echo ""
echo -e "${GREEN}Deployment script completed!${NC}"