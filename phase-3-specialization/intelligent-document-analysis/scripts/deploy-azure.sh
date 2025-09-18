#!/bin/bash

# Azure App Service Deployment Script
# This script helps deploy the intelligent-document-analysis app to Azure App Service

set -e

echo "Starting Azure App Service Deployment..."

# Configuration
APP_NAME="intelligent-document-analysis"
RESOURCE_GROUP="rg-document-analysis"
LOCATION="eastus"
PLAN_NAME="plan-document-analysis"
REGISTRY_NAME="acrdocumentanalysis"
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

# Create resource group
echo -e "${YELLOW}Creating resource group...${NC}"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service plan
echo -e "${YELLOW}Creating App Service plan...${NC}"
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku $SKU \
    --is-linux

# Create Container Registry
echo -e "${YELLOW}🐳 Creating Container Registry...${NC}"
az acr create \
    --name $REGISTRY_NAME \
    --resource-group $RESOURCE_GROUP \
    --sku Basic \
    --admin-enabled true

# Login to ACR
echo -e "${YELLOW}Logging into Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

# Build and push Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f Dockerfile.azure -t $REGISTRY_NAME.azurecr.io/$APP_NAME:latest .

echo -e "${YELLOW}Pushing Docker image...${NC}"
docker push $REGISTRY_NAME.azurecr.io/$APP_NAME:latest

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

# Configure app settings
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
        STREAMLIT_SERVER_PORT=8000 \
        STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        STREAMLIT_SERVER_HEADLESS=true \
        STREAMLIT_SERVER_ENABLE_CORS=false \
        STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
        STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200 \
        ENVIRONMENT=production \
        DEBUG=false \
        LOG_LEVEL=INFO

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

# Configure health check
echo -e "${YELLOW}Configuring health check...${NC}"
az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --startup-file "python -m streamlit run src/web/app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"

# Get app URL
APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

echo -e "${GREEN}Azure App Service created successfully!${NC}"
echo ""
echo -e "${GREEN}Your app is available at:${NC}"
echo "https://$APP_URL"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Set your Azure OpenAI API keys in the App Service configuration:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_OPENAI_ENDPOINT=your-endpoint AZURE_OPENAI_API_KEY=your-key"
echo ""
echo "2. Set your Azure Storage configuration:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_STORAGE_ACCOUNT_NAME=your-account AZURE_STORAGE_ACCOUNT_KEY=your-key"
echo ""
echo "3. Monitor your application:"
echo "   az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "4. View application health:"
echo "   https://$APP_URL/health"
echo ""
echo -e "${GREEN}Deployment script completed!${NC}"
