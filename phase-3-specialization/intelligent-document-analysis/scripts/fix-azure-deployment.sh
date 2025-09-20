#!/bin/bash

# Azure Deployment Fix Script
# This script fixes the identified Azure deployment issues

set -e

echo "🔧 Starting Azure Deployment Fix..."

# Configuration
APP_NAME="intelligent-document-analysis"
RESOURCE_GROUP="rg-data-ai-eng-con"
REGISTRY_NAME="1a27253794c8488f83ef31437e7d1248"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔍 Diagnosing Azure deployment issues...${NC}"

# Check if Azure CLI is installed and logged in
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ ERROR: Azure CLI is not installed${NC}"
    exit 1
fi

if ! az account show &> /dev/null; then
    echo -e "${YELLOW}🔐 Please log in to Azure:${NC}"
    az login
fi

echo -e "${GREEN}✅ Azure CLI is ready${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ ERROR: Docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is ready${NC}"

# Fix 1: Get actual ACR credentials and update app settings
echo -e "${YELLOW}🔑 Fixing Container Registry authentication...${NC}"

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --query passwords[0].value --output tsv)

if [ -z "$ACR_USERNAME" ] || [ -z "$ACR_PASSWORD" ]; then
    echo -e "${RED}❌ ERROR: Could not retrieve ACR credentials${NC}"
    echo "Make sure the Container Registry exists and you have access to it."
    exit 1
fi

echo -e "${GREEN}✅ Retrieved ACR credentials${NC}"

# Update app settings with actual credentials
echo -e "${YELLOW}📝 Updating app settings with actual credentials...${NC}"
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DOCKER_REGISTRY_SERVER_URL=https://$REGISTRY_NAME.azurecr.io \
        DOCKER_REGISTRY_SERVER_USERNAME=$ACR_USERNAME \
        DOCKER_REGISTRY_SERVER_PASSWORD=$ACR_PASSWORD

echo -e "${GREEN}✅ App settings updated${NC}"

# Fix 2: Build and push Docker image with correct architecture
echo -e "${YELLOW}🐳 Building Docker image with correct architecture...${NC}"

# Login to ACR
az acr login --name $REGISTRY_NAME

# Build image with explicit platform for linux/amd64
echo -e "${YELLOW}Building image for linux/amd64 platform...${NC}"
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.azure \
    -t $REGISTRY_NAME.azurecr.io/$APP_NAME:latest \
    --push .

echo -e "${GREEN}✅ Docker image built and pushed successfully${NC}"

# Fix 3: Update web app configuration
echo -e "${YELLOW}⚙️ Updating web app configuration...${NC}"

# Configure container settings
az webapp config container set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME:latest \
    --docker-registry-server-url https://$REGISTRY_NAME.azurecr.io \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

echo -e "${GREEN}✅ Web app configuration updated${NC}"

# Fix 4: Restart the web app
echo -e "${YELLOW}🔄 Restarting web app...${NC}"
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

echo -e "${GREEN}✅ Web app restarted${NC}"

# Fix 5: Wait for deployment and check status
echo -e "${YELLOW}⏳ Waiting for deployment to complete...${NC}"
sleep 30

# Check app status
echo -e "${YELLOW}📊 Checking application status...${NC}"
APP_STATUS=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query state --output tsv)
echo "App Status: $APP_STATUS"

# Get app URL
APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

echo -e "${GREEN}🎉 Azure deployment fix completed!${NC}"
echo ""
echo -e "${GREEN}Your app is available at:${NC}"
echo "https://$APP_URL"
echo ""
echo -e "${BLUE}Health check URL:${NC}"
echo "https://$APP_URL/health"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Wait 2-3 minutes for the container to fully start"
echo "2. Check the health endpoint: https://$APP_URL/health"
echo "3. Monitor logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo "4. Set your environment variables (OpenAI, Storage, etc.)"
echo ""
echo -e "${GREEN}🔧 Fixes applied:${NC}"
echo "✅ Container Registry authentication fixed"
echo "✅ Docker image rebuilt for linux/amd64"
echo "✅ App settings updated with actual credentials"
echo "✅ Web app configuration updated"
echo "✅ Application restarted"
echo ""
echo -e "${BLUE}If you still encounter issues, check the logs:${NC}"
echo "az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
