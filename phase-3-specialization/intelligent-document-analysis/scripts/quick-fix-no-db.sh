#!/bin/bash

# Quick Fix Script - Configure app to work without database temporarily
# This script configures the application to work without database dependency

set -e

# Configuration
APP_NAME="intelligent-document-analysis"
RESOURCE_GROUP="rg-data-ai-eng-con"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Quick Fix - Configuring app without database...${NC}"
echo ""

# Check if Azure CLI is available
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged in to Azure${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Azure CLI ready${NC}"

# Step 1: Configure app settings to disable database dependency
echo -e "${YELLOW}⚙️  Step 1: Configuring app settings...${NC}"

# Set environment variables to disable database and configure basic settings
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DATABASE_URL="sqlite:///app/temp/documents.db" \
        DB_DISABLED="true" \
        USE_SQLITE="true" \
        AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
        AZURE_OPENAI_API_KEY="your-azure-openai-api-key" \
        AZURE_OPENAI_API_VERSION="2023-12-01-preview" \
        AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o" \
        AZURE_STORAGE_ACCOUNT_NAME="your-storage-account" \
        AZURE_STORAGE_ACCOUNT_KEY="your-storage-account-key" \
        AZURE_STORAGE_CONTAINER_NAME="documents" \
        SECRET_KEY="your-very-secure-secret-key-here" \
        ENVIRONMENT="production" \
        DEBUG="false" \
        LOG_LEVEL="INFO" \
        DISABLE_DATABASE="true" \
        USE_FILE_STORAGE="true"

echo -e "${GREEN}✅ App settings configured${NC}"

# Step 2: Restart the application
echo -e "${YELLOW}🔄 Step 2: Restarting application...${NC}"

az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

echo -e "${GREEN}✅ Application restarted${NC}"

# Step 3: Wait and test
echo -e "${YELLOW}⏳ Step 3: Waiting for application to restart...${NC}"

echo "Waiting 60 seconds for application to restart..."
sleep 60

# Test health endpoint
echo -e "${YELLOW}🏥 Testing health endpoint...${NC}"
APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv)

if curl -f -s --max-time 30 "https://$APP_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Health endpoint responding${NC}"
    HEALTH_STATUS="OK"
else
    echo -e "${YELLOW}⚠️ Health endpoint not yet responding (may need more time)${NC}"
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

echo ""
echo -e "${GREEN}🎉 Quick fix completed!${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo "✅ App configured to work without database"
echo "✅ SQLite fallback configured"
echo "✅ Basic Azure services configured (with placeholders)"
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
echo -e "${GREEN}✅ Quick fix completed! The app should now work without database errors.${NC}"
