#!/bin/bash

# Azure Fresh Deployment Script
# This script completely removes existing resources and deploys from scratch

set -e

# Configuration
APP_NAME="intelligent-document-analysis"
RESOURCE_GROUP="rg-data-ai-eng-con"
LOCATION="australiaeast"
PLAN_NAME="plan-document-analysis"
REGISTRY_NAME="1a27253794c8488f83ef31437e7d1248"
SKU="P1V2"  # Change to FREE for development

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Fresh Azure Deployment...${NC}"
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will DELETE all existing resources and deploy from scratch!${NC}"
echo "App Name: $APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Plan: $PLAN_NAME"
echo "Registry: $REGISTRY_NAME"
echo "SKU: $SKU"
echo ""

# Confirmation prompt
read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${RED}❌ Deployment cancelled by user${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Proceeding with fresh deployment...${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ ERROR: Azure CLI is not installed${NC}"
    echo "Please install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ ERROR: Docker is not installed${NC}"
    echo "Please install Docker: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}🔐 Please log in to Azure:${NC}"
    az login
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Step 1: Clean up existing resources
echo -e "${YELLOW}🧹 Step 1: Cleaning up existing resources...${NC}"

# Delete web app if exists
if az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${YELLOW}Deleting existing web app...${NC}"
    az webapp delete --name $APP_NAME --resource-group $RESOURCE_GROUP --yes
    echo -e "${GREEN}✅ Web app deleted${NC}"
else
    echo -e "${GREEN}✅ No existing web app found${NC}"
fi

# Delete app service plan if exists
if az appservice plan show --name $PLAN_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${YELLOW}Deleting existing app service plan...${NC}"
    az appservice plan delete --name $PLAN_NAME --resource-group $RESOURCE_GROUP --yes
    echo -e "${GREEN}✅ App service plan deleted${NC}"
else
    echo -e "${GREEN}✅ No existing app service plan found${NC}"
fi

# Delete container registry if exists
if az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${YELLOW}Deleting existing container registry...${NC}"
    az acr delete --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP --yes
    echo -e "${GREEN}✅ Container registry deleted${NC}"
else
    echo -e "${GREEN}✅ No existing container registry found${NC}"
fi

# Delete resource group if exists (optional - uncomment if you want to delete everything)
# echo -e "${YELLOW}Deleting resource group...${NC}"
# az group delete --name $RESOURCE_GROUP --yes --no-wait
# echo -e "${GREEN}✅ Resource group deletion initiated${NC}"

echo -e "${GREEN}✅ Cleanup completed${NC}"
echo ""

# Step 2: Create fresh resources
echo -e "${YELLOW}🏗️  Step 2: Creating fresh resources...${NC}"

# Create resource group
echo -e "${YELLOW}Creating resource group...${NC}"
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}✅ Resource group already exists${NC}"
else
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo -e "${GREEN}✅ Resource group created${NC}"
fi

# Create App Service plan
echo -e "${YELLOW}Creating App Service plan...${NC}"
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku $SKU \
    --is-linux
echo -e "${GREEN}✅ App Service plan created${NC}"

# Create Container Registry
echo -e "${YELLOW}Creating Container Registry...${NC}"
az acr create \
    --name $REGISTRY_NAME \
    --resource-group $RESOURCE_GROUP \
    --sku Basic \
    --admin-enabled true
echo -e "${GREEN}✅ Container Registry created${NC}"

echo -e "${GREEN}✅ Fresh resources created${NC}"
echo ""

# Step 3: Build and push Docker image
echo -e "${YELLOW}🐳 Step 3: Building and pushing Docker image...${NC}"

# Login to ACR
echo -e "${YELLOW}Logging into Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

# Build and push image with correct platform
echo -e "${YELLOW}Building Docker image for linux/amd64...${NC}"
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.azure \
    -t $REGISTRY_NAME.azurecr.io/$APP_NAME:latest \
    --push .

echo -e "${GREEN}✅ Docker image built and pushed${NC}"
echo ""

# Step 4: Create web app
echo -e "${YELLOW}🌐 Step 4: Creating web app...${NC}"

# Create web app
az webapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --deployment-container-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME:latest

echo -e "${GREEN}✅ Web app created${NC}"

# Step 5: Configure app settings
echo -e "${YELLOW}⚙️  Step 5: Configuring app settings...${NC}"

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value --output tsv)

# Configure app settings
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

echo -e "${GREEN}✅ App settings configured${NC}"

# Step 6: Configure continuous deployment
echo -e "${YELLOW}🔄 Step 6: Configuring continuous deployment...${NC}"

az webapp config container set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME:latest

echo -e "${GREEN}✅ Continuous deployment configured${NC}"

# Step 7: Enable logging
echo -e "${YELLOW}📋 Step 7: Enabling application logging...${NC}"

az webapp log config \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --application-logging filesystem \
    --level information

echo -e "${GREEN}✅ Logging enabled${NC}"

# Step 8: Configure health check
echo -e "${YELLOW}🏥 Step 8: Configuring health check...${NC}"

az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --startup-file "python -m streamlit run src/web/app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"

echo -e "${GREEN}✅ Health check configured${NC}"

# Step 9: Get app URL and wait for deployment
echo -e "${YELLOW}⏳ Step 9: Waiting for deployment to complete...${NC}"

APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

echo -e "${GREEN}✅ Fresh deployment completed!${NC}"
echo ""
echo -e "${GREEN}🎉 Your application is being deployed to:${NC}"
echo "https://$APP_URL"
echo ""
echo -e "${YELLOW}⏳ Please wait 3-5 minutes for the container to fully start...${NC}"
echo ""

# Step 10: Run verification
echo -e "${YELLOW}🔍 Step 10: Running deployment verification...${NC}"

# Wait a bit for the app to start
echo "Waiting 60 seconds for app to initialize..."
sleep 60

# Run verification script if it exists
if [ -f "./scripts/verify-deployment.sh" ]; then
    echo -e "${YELLOW}Running verification script...${NC}"
    ./scripts/verify-deployment.sh
else
    echo -e "${YELLOW}Manual verification:${NC}"
    echo "1. Check health endpoint: curl -f https://$APP_URL/health"
    echo "2. Check main app: curl -f https://$APP_URL/"
    echo "3. Monitor logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
fi

echo ""
echo -e "${GREEN}🎉 Fresh Azure deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Set your Azure OpenAI API keys:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_OPENAI_ENDPOINT=your-endpoint AZURE_OPENAI_API_KEY=your-key"
echo ""
echo "2. Set your Azure Storage configuration:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_STORAGE_ACCOUNT_NAME=your-account AZURE_STORAGE_ACCOUNT_KEY=your-key"
echo ""
echo "3. Monitor your application:"
echo "   az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "4. Access your application:"
echo "   https://$APP_URL"
echo ""
echo "5. Health check:"
echo "   https://$APP_URL/health"
echo ""
echo -e "${GREEN}✅ Fresh deployment from scratch completed!${NC}"
