#!/bin/bash

# RAG Conversational AI Assistant - Azure Deployment Verification Script
# Based on successful patterns from intelligent-document-analysis deployment
# This script verifies that the Azure deployment is working correctly

set -e

# Configuration
APP_NAME="rag-conversational-ai-assistant"
RESOURCE_GROUP="rg-data-ai-eng-con"
REGISTRY_NAME="1a27253794c8488f83ef31437e7d1248"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Verifying Azure deployment...${NC}"
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

# Get app URL
APP_URL=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query defaultHostName --output tsv 2>/dev/null)

if [ -z "$APP_URL" ]; then
    echo -e "${RED}❌ Could not retrieve app URL. App may not exist.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ App URL: https://$APP_URL${NC}"

# Check app status
echo -e "${YELLOW}📊 Checking app status...${NC}"
APP_STATE=$(az webapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query state --output tsv)
echo "App State: $APP_STATE"

if [ "$APP_STATE" != "Running" ]; then
    echo -e "${YELLOW}⚠️ App is not in Running state. Current state: $APP_STATE${NC}"
fi

# Check container configuration
echo -e "${YELLOW}🐳 Checking container configuration...${NC}"
CONTAINER_IMAGE=$(az webapp config container show --name $APP_NAME --resource-group $RESOURCE_GROUP --query dockerImageName --output tsv 2>/dev/null)
echo "Container Image: $CONTAINER_IMAGE"

# Check if image exists in registry
echo -e "${YELLOW}🔍 Verifying container image exists...${NC}"
if az acr repository show --name $REGISTRY_NAME --image rag-conversational-ai-assistant:latest &> /dev/null; then
    echo -e "${GREEN}✅ Container image exists in registry${NC}"
else
    echo -e "${RED}❌ Container image not found in registry${NC}"
fi

# Test health endpoint
echo -e "${YELLOW}🏥 Testing health endpoint...${NC}"
HEALTH_URL="https://$APP_URL/health"

# Wait a bit for the app to be ready
echo "Waiting 30 seconds for app to be ready..."
sleep 30

# Test health endpoint with timeout
if curl -f -s --max-time 30 "$HEALTH_URL" > /dev/null; then
    echo -e "${GREEN}✅ Health endpoint responding${NC}"
    HEALTH_STATUS="OK"
else
    echo -e "${RED}❌ Health endpoint not responding${NC}"
    HEALTH_STATUS="FAILED"
fi

# Test main application
echo -e "${YELLOW}🌐 Testing main application...${NC}"
MAIN_URL="https://$APP_URL/"

if curl -f -s --max-time 30 "$MAIN_URL" > /dev/null; then
    echo -e "${GREEN}✅ Main application responding${NC}"
    MAIN_STATUS="OK"
else
    echo -e "${RED}❌ Main application not responding${NC}"
    MAIN_STATUS="FAILED"
fi

# Test API endpoints
echo -e "${YELLOW}🔌 Testing API endpoints...${NC}"
DOCS_URL="https://$APP_URL/docs"

if curl -f -s --max-time 30 "$DOCS_URL" > /dev/null; then
    echo -e "${GREEN}✅ API documentation endpoint responding${NC}"
    API_STATUS="OK"
else
    echo -e "${RED}❌ API documentation endpoint not responding${NC}"
    API_STATUS="FAILED"
fi

# Check recent logs for errors
echo -e "${YELLOW}📋 Checking recent logs for errors...${NC}"
RECENT_ERRORS=$(az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP --provider application --timeout 10 2>/dev/null | grep -i "error\|exception\|failed" | tail -5 || echo "No recent errors found")

if [ "$RECENT_ERRORS" != "No recent errors found" ]; then
    echo -e "${YELLOW}⚠️ Recent errors found:${NC}"
    echo "$RECENT_ERRORS"
else
    echo -e "${GREEN}✅ No recent errors in logs${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}📊 Deployment Verification Summary${NC}"
echo "=================================="
echo "App Name: $APP_NAME"
echo "App URL: https://$APP_URL"
echo "App State: $APP_STATE"
echo "Container Image: $CONTAINER_IMAGE"
echo "Health Endpoint: $HEALTH_STATUS"
echo "Main Application: $MAIN_STATUS"
echo "API Documentation: $API_STATUS"
echo ""

if [ "$HEALTH_STATUS" = "OK" ] && [ "$MAIN_STATUS" = "OK" ] && [ "$APP_STATE" = "Running" ]; then
    echo -e "${GREEN}🎉 Deployment verification PASSED!${NC}"
    echo -e "${GREEN}Your RAG Conversational AI Assistant is running successfully.${NC}"
    echo ""
    echo -e "${BLUE}Access your application at:${NC}"
    echo "Main App: https://$APP_URL"
    echo "API Docs: https://$APP_URL/docs"
    echo "Health Check: https://$APP_URL/health"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Configure your AI service API keys"
    echo "2. Upload some documents to test the RAG functionality"
    echo "3. Monitor the application logs and performance"
    exit 0
else
    echo -e "${RED}❌ Deployment verification FAILED!${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo "1. Check the logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
    echo "2. Restart the app: az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP"
    echo "3. Check environment variables in Azure Portal"
    echo "4. Verify container image was built and pushed correctly"
    echo "5. Check if all required dependencies are installed"
    exit 1
fi
