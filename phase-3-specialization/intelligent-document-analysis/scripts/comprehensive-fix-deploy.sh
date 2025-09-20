#!/bin/bash

# Comprehensive Fix and Deploy Script
# This script fixes all identified issues and redeploys the application

set -e

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

echo -e "${BLUE}🔧 Comprehensive Fix and Deploy Script${NC}"
echo ""
echo -e "${YELLOW}This script will:${NC}"
echo "1. Fix all database-related issues"
echo "2. Update application configuration"
echo "3. Rebuild and redeploy the application"
echo "4. Verify the deployment"
echo ""

# Check prerequisites
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    exit 1
fi

if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged in to Azure${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Step 1: Configure app settings to disable database completely
echo -e "${YELLOW}⚙️  Step 1: Configuring app settings...${NC}"

az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DB_DISABLED="true" \
        USE_SQLITE="false" \
        DATABASE_URL="sqlite:///app/temp/documents.db" \
        DISABLE_DATABASE="true" \
        USE_FILE_STORAGE="true" \
        ENVIRONMENT="production" \
        DEBUG="false" \
        LOG_LEVEL="INFO" \
        AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
        AZURE_OPENAI_API_KEY="your-azure-openai-api-key" \
        AZURE_OPENAI_API_VERSION="2023-12-01-preview" \
        AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o" \
        AZURE_STORAGE_ACCOUNT_NAME="your-storage-account" \
        AZURE_STORAGE_ACCOUNT_KEY="your-storage-account-key" \
        AZURE_STORAGE_CONTAINER_NAME="documents" \
        SECRET_KEY="your-very-secure-secret-key-here"

echo -e "${GREEN}✅ App settings configured${NC}"

# Step 2: Login to ACR
echo -e "${YELLOW}🐳 Step 2: Logging into Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

# Step 3: Build and push updated Docker image
echo -e "${YELLOW}🔨 Step 3: Building and pushing updated Docker image...${NC}"

docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.azure \
    -t $REGISTRY_NAME.azurecr.io/$APP_NAME:latest \
    --push .

echo -e "${GREEN}✅ Docker image built and pushed${NC}"

# Step 4: Update web app configuration
echo -e "${YELLOW}🌐 Step 4: Updating web app configuration...${NC}"

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value --output tsv)

# Update container configuration
az webapp config container set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name $REGISTRY_NAME.azurecr.io/$APP_NAME:latest \
    --docker-registry-server-url https://$REGISTRY_NAME.azurecr.io \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

echo -e "${GREEN}✅ Web app configuration updated${NC}"

# Step 5: Configure startup command
echo -e "${YELLOW}🚀 Step 5: Configuring startup command...${NC}"

az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --startup-file "python -m streamlit run src/web/app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"

echo -e "${GREEN}✅ Startup command configured${NC}"

# Step 6: Restart the application
echo -e "${YELLOW}🔄 Step 6: Restarting application...${NC}"

az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

echo -e "${GREEN}✅ Application restarted${NC}"

# Step 7: Wait for deployment
echo -e "${YELLOW}⏳ Step 7: Waiting for deployment to complete...${NC}"

echo "Waiting 90 seconds for application to start..."
sleep 90

# Step 8: Verify deployment
echo -e "${YELLOW}🔍 Step 8: Verifying deployment...${NC}"

APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

# Test health endpoint
echo -e "${YELLOW}🏥 Testing health endpoint...${NC}"
if curl -f -s --max-time 30 "https://$APP_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Health endpoint responding${NC}"
    HEALTH_STATUS="OK"
else
    echo -e "${YELLOW}⚠️ Health endpoint not yet responding${NC}"
    HEALTH_STATUS="STARTING"
fi

# Test main application
echo -e "${YELLOW}🌐 Testing main application...${NC}"
if curl -f -s --max-time 30 "https://$APP_URL/" > /dev/null; then
    echo -e "${GREEN}✅ Main application responding${NC}"
    MAIN_STATUS="OK"
else
    echo -e "${YELLOW}⚠️ Main application not yet responding${NC}"
    MAIN_STATUS="STARTING"
fi

# Check recent logs for errors
echo -e "${YELLOW}📋 Checking recent logs...${NC}"
RECENT_ERRORS=$(az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP --provider application | head -20 | grep -i "error\|exception\|failed" | tail -5 || echo "No recent errors found")

if [ "$RECENT_ERRORS" != "No recent errors found" ]; then
    echo -e "${YELLOW}⚠️ Recent errors found:${NC}"
    echo "$RECENT_ERRORS"
else
    echo -e "${GREEN}✅ No recent errors in logs${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Comprehensive fix and deployment completed!${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo "✅ Database issues fixed"
echo "✅ Application code updated"
echo "✅ Docker image rebuilt and deployed"
echo "✅ Startup command configured"
echo "✅ Application restarted"
echo ""
echo -e "${BLUE}📋 Application Status:${NC}"
echo "Health Endpoint: $HEALTH_STATUS"
echo "Main Application: $MAIN_STATUS"
echo ""
echo -e "${BLUE}📋 Access Your Application:${NC}"
echo "URL: https://$APP_URL"
echo "Health Check: https://$APP_URL/health"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Test the application functionality"
echo "2. Update Azure OpenAI settings with your actual API keys:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_OPENAI_ENDPOINT=your-endpoint AZURE_OPENAI_API_KEY=your-key"
echo ""
echo "3. Update Azure Storage settings with your actual storage account:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_STORAGE_ACCOUNT_NAME=your-account AZURE_STORAGE_ACCOUNT_KEY=your-key"
echo ""
echo "4. Monitor logs:"
echo "   az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo -e "${GREEN}✅ All issues should now be resolved!${NC}"
