#!/bin/bash

# Azure Deployment Script for RAG Conversational AI Assistant
# This script deploys the application to Azure App Service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[AZURE DEPLOY]${NC} $1"
}

# Configuration
RESOURCE_GROUP="rag-assistant-rg"
LOCATION="eastus"
APP_SERVICE_PLAN="rag-assistant-plan"
WEB_APP_NAME="rag-assistant-api"
FUNCTION_APP_NAME="rag-assistant-functions"
STORAGE_ACCOUNT="ragassistantstorage"
KEY_VAULT="rag-assistant-keyvault"
POSTGRES_SERVER="rag-postgres-server"
REDIS_CACHE="rag-cache"
SEARCH_SERVICE="rag-search"
OPENAI_SERVICE="rag-openai"

# Check if Azure CLI is installed
check_azure_cli() {
    print_status "Checking Azure CLI installation..."
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed. Please install it first."
        print_status "Installation instructions: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    print_status "Azure CLI found ✓"
}

# Login to Azure
login_azure() {
    print_status "Logging in to Azure..."
    if ! az account show &> /dev/null; then
        az login
    fi
    print_status "Logged in to Azure ✓"
}

# Create resource group
create_resource_group() {
    print_status "Creating resource group: $RESOURCE_GROUP"
    if ! az group show --name $RESOURCE_GROUP &> /dev/null; then
        az group create --name $RESOURCE_GROUP --location $LOCATION
        print_status "Resource group created ✓"
    else
        print_status "Resource group already exists ✓"
    fi
}

# Create storage account
create_storage_account() {
    print_status "Creating storage account: $STORAGE_ACCOUNT"
    if ! az storage account show --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP &> /dev/null; then
        az storage account create \
            --name $STORAGE_ACCOUNT \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --sku Standard_LRS \
            --kind StorageV2
        print_status "Storage account created ✓"
    else
        print_status "Storage account already exists ✓"
    fi
}

# Create Key Vault
create_key_vault() {
    print_status "Creating Key Vault: $KEY_VAULT"
    if ! az keyvault show --name $KEY_VAULT --resource-group $RESOURCE_GROUP &> /dev/null; then
        az keyvault create \
            --name $KEY_VAULT \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --sku standard
        print_status "Key Vault created ✓"
    else
        print_status "Key Vault already exists ✓"
    fi
}

# Create PostgreSQL server
create_postgresql() {
    print_status "Creating PostgreSQL server: $POSTGRES_SERVER"
    if ! az postgres flexible-server show --name $POSTGRES_SERVER --resource-group $RESOURCE_GROUP &> /dev/null; then
        az postgres flexible-server create \
            --name $POSTGRES_SERVER \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --admin-user rag_admin \
            --admin-password $(openssl rand -base64 32) \
            --sku-name Standard_B1ms \
            --tier Burstable \
            --public-access 0.0.0.0 \
            --storage-size 32
        print_status "PostgreSQL server created ✓"
    else
        print_status "PostgreSQL server already exists ✓"
    fi
}

# Create Redis cache
create_redis_cache() {
    print_status "Creating Redis cache: $REDIS_CACHE"
    if ! az redis show --name $REDIS_CACHE --resource-group $RESOURCE_GROUP &> /dev/null; then
        az redis create \
            --name $REDIS_CACHE \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --sku Standard \
            --vm-size c1
        print_status "Redis cache created ✓"
    else
        print_status "Redis cache already exists ✓"
    fi
}

# Create Azure Search service
create_search_service() {
    print_status "Creating Azure Search service: $SEARCH_SERVICE"
    if ! az search service show --name $SEARCH_SERVICE --resource-group $RESOURCE_GROUP &> /dev/null; then
        az search service create \
            --name $SEARCH_SERVICE \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --sku Standard
        print_status "Azure Search service created ✓"
    else
        print_status "Azure Search service already exists ✓"
    fi
}

# Create App Service plan
create_app_service_plan() {
    print_status "Creating App Service plan: $APP_SERVICE_PLAN"
    if ! az appservice plan show --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP &> /dev/null; then
        az appservice plan create \
            --name $APP_SERVICE_PLAN \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --sku P1V2 \
            --is-linux
        print_status "App Service plan created ✓"
    else
        print_status "App Service plan already exists ✓"
    fi
}

# Create web app
create_web_app() {
    print_status "Creating web app: $WEB_APP_NAME"
    if ! az webapp show --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        az webapp create \
            --name $WEB_APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --plan $APP_SERVICE_PLAN \
            --runtime "PYTHON|3.11"
        print_status "Web app created ✓"
    else
        print_status "Web app already exists ✓"
    fi
}

# Configure web app settings
configure_web_app() {
    print_status "Configuring web app settings..."
    
    # Get connection strings
    POSTGRES_CONNECTION_STRING=$(az postgres flexible-server show-connection-string \
        --name $POSTGRES_SERVER \
        --resource-group $RESOURCE_GROUP \
        --admin-user rag_admin \
        --admin-password $(az postgres flexible-server show --name $POSTGRES_SERVER --resource-group $RESOURCE_GROUP --query "administratorLoginPassword" -o tsv) \
        --database-name rag_assistant \
        --query "connectionStrings.psql" -o tsv)
    
    REDIS_CONNECTION_STRING=$(az redis list-keys \
        --name $REDIS_CACHE \
        --resource-group $RESOURCE_GROUP \
        --query "primaryKey" -o tsv)
    
    # Configure app settings
    az webapp config appsettings set \
        --name $WEB_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --settings \
            DATABASE_URL="$POSTGRES_CONNECTION_STRING" \
            REDIS_URL="rediss://$REDIS_CACHE.redis.cache.windows.net:6380/0?password=$REDIS_CONNECTION_STRING&ssl=True" \
            AZURE_KEY_VAULT_URL="https://$KEY_VAULT.vault.azure.net/" \
            AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=$STORAGE_ACCOUNT;AccountKey=$(az storage account keys list --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP --query "[0].value" -o tsv);EndpointSuffix=core.windows.net" \
            AZURE_SEARCH_ENDPOINT="https://$SEARCH_SERVICE.search.windows.net" \
            AZURE_SEARCH_API_KEY="$(az search admin-key show --name $SEARCH_SERVICE --resource-group $RESOURCE_GROUP --query "primaryKey" -o tsv)" \
            ENABLE_AZURE_SERVICES="true"
    
    print_status "Web app settings configured ✓"
}

# Deploy application
deploy_application() {
    print_status "Deploying application..."
    
    # Build and push Docker image
    print_status "Building Docker image..."
    docker build -t rag-assistant-api .
    
    # Tag for Azure Container Registry (if using ACR)
    # docker tag rag-assistant-api:latest $ACR_NAME.azurecr.io/rag-assistant-api:latest
    # docker push $ACR_NAME.azurecr.io/rag-assistant-api:latest
    
    # Deploy to App Service
    print_status "Deploying to App Service..."
    az webapp deployment source config \
        --name $WEB_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --repo-url https://github.com/your-org/rag-conversational-ai-assistant \
        --branch main \
        --manual-integration
    
    print_status "Application deployed ✓"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Create Application Insights
    az monitor app-insights component create \
        --app rag-assistant-insights \
        --location $LOCATION \
        --resource-group $RESOURCE_GROUP \
        --application-type web
    
    # Get Application Insights connection string
    INSIGHTS_CONNECTION_STRING=$(az monitor app-insights component show \
        --app rag-assistant-insights \
        --resource-group $RESOURCE_GROUP \
        --query "connectionString" -o tsv)
    
    # Configure Application Insights
    az webapp config appsettings set \
        --name $WEB_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --settings APPLICATION_INSIGHTS_CONNECTION_STRING="$INSIGHTS_CONNECTION_STRING"
    
    print_status "Monitoring configured ✓"
}

# Main deployment function
main() {
    print_header "Starting Azure deployment for RAG Conversational AI Assistant"
    echo "=========================================="
    
    check_azure_cli
    login_azure
    create_resource_group
    create_storage_account
    create_key_vault
    create_postgresql
    create_redis_cache
    create_search_service
    create_app_service_plan
    create_web_app
    configure_web_app
    setup_monitoring
    deploy_application
    
    echo ""
    echo "=========================================="
    print_header "Deployment completed successfully!"
    echo "=========================================="
    echo ""
    echo "Web App URL: https://$WEB_APP_NAME.azurewebsites.net"
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Location: $LOCATION"
    echo ""
    echo "Next steps:"
    echo "1. Configure Azure OpenAI Service"
    echo "2. Set up secrets in Key Vault"
    echo "3. Run database migrations"
    echo "4. Test the application"
    echo ""
    echo "For more information, see the Azure deployment guide."
}

# Run main function
main "$@"
