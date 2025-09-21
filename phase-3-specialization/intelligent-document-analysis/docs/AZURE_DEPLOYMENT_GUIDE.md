# Azure App Service Deployment Guide

This guide provides step-by-step instructions for deploying the Intelligent Document Analysis System to Azure App Service.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Azure Resources Setup](#azure-resources-setup)
3. [Application Configuration](#application-configuration)
4. [Container Registry Setup](#container-registry-setup)
5. [Deployment Process](#deployment-process)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Cost Optimization](#cost-optimization)

## Prerequisites

### Required Tools
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) (latest version)
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Git](https://git-scm.com/downloads)
- [Python 3.11+](https://www.python.org/downloads/)

### Azure Account Requirements
- Active Azure subscription
- Owner or Contributor access to the subscription
- Sufficient quota for the following resources:
  - App Service Plan (Linux)
  - Container Registry
  - Database for PostgreSQL
  - Cache for Redis
  - Storage Account
  - Application Insights

## Azure Resources Setup

### 1. Login to Azure
```bash
az login
az account set --subscription "Your Subscription Name"
```

### 2. Create Resource Group
```bash
az group create \
  --name rg-document-analysis \
  --location "East US"
```

### 3. Create App Service Plan
```bash
# For production (recommended)
az appservice plan create \
  --name plan-document-analysis \
  --resource-group rg-document-analysis \
  --location "East US" \
  --sku P1V2 \
  --is-linux

# For development/testing (free tier)
az appservice plan create \
  --name plan-document-analysis-dev \
  --resource-group rg-document-analysis \
  --location "East US" \
  --sku FREE \
  --is-linux
```

### 4. Create Azure Database for PostgreSQL
```bash
az postgres flexible-server create \
  --resource-group rg-document-analysis \
  --name postgres-document-analysis \
  --location "East US" \
  --admin-user dbadmin \
  --admin-password "YourSecurePassword123!" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --public-access 0.0.0.0 \
  --storage-size 32

# Create database
az postgres flexible-server db create \
  --resource-group rg-document-analysis \
  --server-name postgres-document-analysis \
  --database-name document_analysis
```

### 5. Create Azure Cache for Redis
```bash
az redis create \
  --resource-group rg-document-analysis \
  --name redis-document-analysis \
  --location "East US" \
  --sku Standard \
  --vm-size c1
```

### 6. Create Storage Account
```bash
az storage account create \
  --name storageaccountanalysis \
  --resource-group rg-document-analysis \
  --location "East US" \
  --sku Standard_LRS \
  --kind StorageV2

# Create container
az storage container create \
  --name documents \
  --account-name storageaccountanalysis
```

### 7. Create Application Insights
```bash
az monitor app-insights component create \
  --app intelligent-document-analysis \
  --location "East US" \
  --resource-group rg-document-analysis \
  --application-type web
```

## Application Configuration

### 1. Environment Variables Setup

Copy the Azure environment template:
```bash
cp .azure/env.azure .env
```

Update the `.env` file with your Azure resource details:

```bash
# Database
DATABASE_URL=postgresql://dbadmin:YourSecurePassword123!@postgres-document-analysis.postgres.database.azure.com:5432/document_analysis?sslmode=require

# Redis
REDIS_URL=rediss://redis-document-analysis.redis.cache.windows.net:6380/0?ssl_cert_reqs=required

# Storage
AZURE_STORAGE_ACCOUNT_NAME=storageaccountanalysis
AZURE_STORAGE_ACCOUNT_KEY=your-storage-key

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=your-connection-string
```

### 2. Azure OpenAI Setup

1. Create an Azure OpenAI resource in the Azure portal
2. Deploy a GPT-4 model
3. Update the environment variables:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

## Container Registry Setup

### 1. Create Azure Container Registry
```bash
az acr create \
  --resource-group rg-document-analysis \
  --name acrdocumentanalysis \
  --sku Basic \
  --admin-enabled true
```

### 2. Login to ACR
```bash
az acr login --name acrdocumentanalysis
```

### 3. Build and Push Docker Image
```bash
# Build the Azure-optimized image
docker build -f Dockerfile.azure -t acrdocumentanalysis.azurecr.io/intelligent-document-analysis:latest .

# Push to registry
docker push acrdocumentanalysis.azurecr.io/intelligent-document-analysis:latest
```

## Deployment Process

### 1. Create Web App
```bash
az webapp create \
  --name intelligent-document-analysis \
  --resource-group rg-document-analysis \
  --plan plan-document-analysis \
  --deployment-container-image-name acrdocumentanalysis.azurecr.io/intelligent-document-analysis:latest
```

### 2. Configure App Settings
```bash
# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name acrdocumentanalysis --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name acrdocumentanalysis --query passwords[0].value --output tsv)

# Configure app settings
az webapp config appsettings set \
  --name intelligent-document-analysis \
  --resource-group rg-document-analysis \
  --settings \
    DOCKER_REGISTRY_SERVER_URL=https://acrdocumentanalysis.azurecr.io \
    DOCKER_REGISTRY_SERVER_USERNAME=$ACR_USERNAME \
    DOCKER_REGISTRY_SERVER_PASSWORD=$ACR_PASSWORD \
    WEBSITES_PORT=8000 \
    WEBSITES_ENABLE_APP_SERVICE_STORAGE=false \
    WEBSITES_CONTAINER_START_TIME_LIMIT=1800
```

### 3. Configure Environment Variables
```bash
# Set all environment variables from .env file
az webapp config appsettings set \
  --name intelligent-document-analysis \
  --resource-group rg-document-analysis \
  --settings @.env
```

### 4. Configure Custom Domain (Optional)
```bash
# Add custom domain
az webapp config hostname add \
  --webapp-name intelligent-document-analysis \
  --resource-group rg-document-analysis \
  --hostname your-domain.com
```

### 5. Enable HTTPS
```bash
# Configure SSL certificate
az webapp config ssl upload \
  --name intelligent-document-analysis \
  --resource-group rg-document-analysis \
  --certificate-file your-certificate.pfx \
  --certificate-password your-password
```

## Monitoring and Maintenance

### 1. Application Insights Integration
The application automatically integrates with Application Insights for:
- Performance monitoring
- Error tracking
- Usage analytics
- Custom metrics

### 2. Log Analytics
```bash
# View application logs
az webapp log tail \
  --name intelligent-document-analysis \
  --resource-group rg-document-analysis
```

### 3. Health Monitoring
The application includes a health check endpoint:
- URL: `https://your-app.azurewebsites.net/health`
- Returns application status and configuration

### 4. Backup Configuration
```bash
# Enable backup
az webapp config backup create \
  --resource-group rg-document-analysis \
  --webapp-name intelligent-document-analysis \
  --backup-name backup-$(date +%Y%m%d) \
  --container-url https://storageaccountanalysis.blob.core.windows.net/backups
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start
**Symptoms:** App shows "Application Error" or doesn't respond
**Solutions:**
- Check container logs: `az webapp log tail --name intelligent-document-analysis --resource-group rg-document-analysis`
- Verify environment variables are set correctly
- Check if the container image exists in ACR
- Ensure the startup command is correct

#### 2. Database Connection Issues
**Symptoms:** Database connection errors in logs
**Solutions:**
- Verify DATABASE_URL is correct
- Check if the PostgreSQL server allows connections from Azure App Service
- Ensure SSL is properly configured
- Verify firewall rules

#### 3. Memory Issues
**Symptoms:** App crashes or becomes unresponsive
**Solutions:**
- Upgrade to a higher App Service Plan
- Optimize application memory usage
- Check for memory leaks in logs
- Consider using Azure Container Instances for better resource control

#### 4. File Upload Issues
**Symptoms:** File uploads fail or timeout
**Solutions:**
- Check file size limits
- Verify storage account configuration
- Ensure proper permissions on storage account
- Check network connectivity

### Debug Commands

```bash
# Check app status
az webapp show --name intelligent-document-analysis --resource-group rg-document-analysis

# View configuration
az webapp config show --name intelligent-document-analysis --resource-group rg-document-analysis

# Restart app
az webapp restart --name intelligent-document-analysis --resource-group rg-document-analysis

# View metrics
az monitor metrics list --resource /subscriptions/{subscription-id}/resourceGroups/rg-document-analysis/providers/Microsoft.Web/sites/intelligent-document-analysis
```

## Cost Optimization

### 1. Right-size Resources
- Use appropriate App Service Plan for your workload
- Monitor usage and adjust accordingly
- Consider auto-scaling for variable workloads

### 2. Storage Optimization
- Use appropriate storage tiers
- Implement data lifecycle policies
- Regular cleanup of temporary files

### 3. Database Optimization
- Use appropriate database tier
- Implement connection pooling
- Regular maintenance and optimization

### 4. Monitoring Costs
```bash
# View resource costs
az consumption usage list --billing-period-name 202401

# Set up cost alerts
az monitor action-group create \
  --name cost-alerts \
  --resource-group rg-document-analysis \
  --short-name cost-alert
```

## Security Best Practices

### 1. Network Security
- Use VNet integration for database access
- Implement proper firewall rules
- Use private endpoints where possible

### 2. Application Security
- Store secrets in Azure Key Vault
- Use managed identities where possible
- Implement proper authentication and authorization

### 3. Data Security
- Enable encryption at rest
- Use HTTPS for all communications
- Implement proper data classification

## Scaling Considerations

### 1. Horizontal Scaling
- Configure auto-scaling rules
- Use multiple instances for high availability
- Implement load balancing

### 2. Vertical Scaling
- Monitor resource usage
- Upgrade App Service Plan when needed
- Consider Azure Container Instances for burst capacity

### 3. Database Scaling
- Use read replicas for read-heavy workloads
- Implement connection pooling
- Consider database sharding for large datasets

## Maintenance Tasks

### Daily
- Monitor application health
- Check error logs
- Verify backup completion

### Weekly
- Review performance metrics
- Check security alerts
- Update dependencies

### Monthly
- Review costs and optimize
- Update security patches
- Performance testing

## Support and Resources

- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [Azure Container Registry Documentation](https://docs.microsoft.com/en-us/azure/container-registry/)
- [Azure Database for PostgreSQL Documentation](https://docs.microsoft.com/en-us/azure/postgresql/)
- [Application Insights Documentation](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview)

For additional support, contact the development team or create an issue in the project repository.
