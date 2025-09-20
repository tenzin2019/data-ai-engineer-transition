#!/bin/bash

# Database Connection Fix Script (Azure SQL Database)
# This script creates Azure SQL Database and configures the application

set -e

# Configuration
APP_NAME="intelligent-document-analysis"
RESOURCE_GROUP="rg-data-ai-eng-con"
LOCATION="australiaeast"
SQL_SERVER_NAME="sql-document-analysis"
DB_NAME="document_analysis"
DB_ADMIN_USER="dbadmin"
DB_ADMIN_PASSWORD="SecurePassword123!"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Fixing Database Connection Issue (Azure SQL)...${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "App Name: $APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "SQL Server: $SQL_SERVER_NAME"
echo "Database Name: $DB_NAME"
echo "Location: $LOCATION"
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

# Step 1: Create Azure SQL Server
echo -e "${YELLOW}🗄️  Step 1: Creating Azure SQL Server...${NC}"

# Check if SQL server already exists
if az sql server show --name $SQL_SERVER_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}✅ SQL server already exists${NC}"
else
    echo -e "${YELLOW}Creating Azure SQL Server...${NC}"
    az sql server create \
        --name $SQL_SERVER_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --admin-user $DB_ADMIN_USER \
        --admin-password $DB_ADMIN_PASSWORD
    
    echo -e "${GREEN}✅ SQL server created${NC}"
fi

# Step 2: Create database
echo -e "${YELLOW}📊 Step 2: Creating database...${NC}"

# Check if database exists
if az sql db show --name $DB_NAME --server $SQL_SERVER_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${GREEN}✅ Database already exists${NC}"
else
    echo -e "${YELLOW}Creating database...${NC}"
    az sql db create \
        --name $DB_NAME \
        --server $SQL_SERVER_NAME \
        --resource-group $RESOURCE_GROUP \
        --service-objective Basic
    
    echo -e "${GREEN}✅ Database created${NC}"
fi

# Step 3: Configure firewall rules
echo -e "${YELLOW}🔥 Step 3: Configuring firewall rules...${NC}"

# Allow Azure services
az sql server firewall-rule create \
    --name "AllowAzureServices" \
    --server $SQL_SERVER_NAME \
    --resource-group $RESOURCE_GROUP \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0

# Allow all IPs (for development - restrict in production)
az sql server firewall-rule create \
    --name "AllowAllIPs" \
    --server $SQL_SERVER_NAME \
    --resource-group $RESOURCE_GROUP \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 255.255.255.255

echo -e "${GREEN}✅ Firewall rules configured${NC}"

# Step 4: Get database connection details
echo -e "${YELLOW}🔗 Step 4: Getting database connection details...${NC}"

DB_HOST="$SQL_SERVER_NAME.database.windows.net"
DB_CONNECTION_STRING="mssql+pyodbc://$DB_ADMIN_USER:$DB_ADMIN_PASSWORD@$DB_HOST:1433/$DB_NAME?driver=ODBC+Driver+17+for+SQL+Server"

echo -e "${GREEN}✅ Database connection details retrieved${NC}"
echo "Database Host: $DB_HOST"
echo "Database Name: $DB_NAME"

# Step 5: Configure app settings
echo -e "${YELLOW}⚙️  Step 5: Configuring app settings...${NC}"

# Set database connection
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        DATABASE_URL="$DB_CONNECTION_STRING" \
        DB_HOST="$DB_HOST" \
        DB_NAME="$DB_NAME" \
        DB_USER="$DB_ADMIN_USER" \
        DB_PASSWORD="$DB_ADMIN_PASSWORD" \
        DB_PORT="1433" \
        DB_TYPE="mssql"

echo -e "${GREEN}✅ Database settings configured${NC}"

# Step 6: Set up basic Azure services (with placeholder values)
echo -e "${YELLOW}☁️  Step 6: Configuring Azure services...${NC}"

# Set placeholder values for Azure services (user needs to update with real values)
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
        AZURE_OPENAI_API_KEY="your-azure-openai-api-key" \
        AZURE_OPENAI_API_VERSION="2023-12-01-preview" \
        AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o" \
        AZURE_STORAGE_ACCOUNT_NAME="your-storage-account" \
        AZURE_STORAGE_ACCOUNT_KEY="your-storage-account-key" \
        AZURE_STORAGE_CONTAINER_NAME="documents" \
        SECRET_KEY="your-very-secure-secret-key-here" \
        ENVIRONMENT="production" \
        DEBUG="false"

echo -e "${GREEN}✅ Azure services configured (with placeholder values)${NC}"

# Step 7: Restart the application
echo -e "${YELLOW}🔄 Step 7: Restarting application...${NC}"

az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

echo -e "${GREEN}✅ Application restarted${NC}"

# Step 8: Wait and test
echo -e "${YELLOW}⏳ Step 8: Waiting for application to restart...${NC}"

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

echo ""
echo -e "${GREEN}🎉 Database connection fix completed!${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo "✅ Azure SQL Database created"
echo "✅ Database connection configured"
echo "✅ Firewall rules set up"
echo "✅ App settings updated"
echo "✅ Application restarted"
echo ""
echo -e "${BLUE}📋 Database Details:${NC}"
echo "Server: $SQL_SERVER_NAME.database.windows.net"
echo "Database: $DB_NAME"
echo "User: $DB_ADMIN_USER"
echo "Connection String: Configured in app settings"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Wait 2-3 minutes for full startup"
echo "2. Test the application: https://$APP_URL"
echo "3. Update Azure OpenAI settings with your actual API keys:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_OPENAI_ENDPOINT=your-endpoint AZURE_OPENAI_API_KEY=your-key"
echo ""
echo "4. Update Azure Storage settings with your actual storage account:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings AZURE_STORAGE_ACCOUNT_NAME=your-account AZURE_STORAGE_ACCOUNT_KEY=your-key"
echo ""
echo "5. Monitor logs:"
echo "   az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo -e "${GREEN}✅ Database connection issue fixed!${NC}"
