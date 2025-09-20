# Azure Deployment Troubleshooting Guide

## Root Cause Analysis Summary

### Primary Issue: Container Registry Authentication Failure
**Error**: `UNAUTHORIZED: authentication required`
**Cause**: Invalid or missing Container Registry credentials in App Service configuration
**Solution**: Update app settings with actual ACR credentials

### Secondary Issues:
1. **Architecture Mismatch**: `no matching manifest for linux/amd64`
2. **Missing Docker Image**: Image not properly built/pushed to registry
3. **Invalid Configuration**: Placeholder values in app settings

## Quick Fix Commands

### 1. Run the Automated Fix Script
```bash
./scripts/fix-azure-deployment.sh
```

### 2. Manual Fix Steps

#### Step 1: Get ACR Credentials
```bash
# Get Container Registry credentials
ACR_USERNAME=$(az acr credential show --name 1a27253794c8488f83ef31437e7d1248 --resource-group rg-data-ai-eng-con --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name 1a27253794c8488f83ef31437e7d1248 --resource-group rg-data-ai-eng-con --query passwords[0].value --output tsv)
```

#### Step 2: Update App Settings
```bash
az webapp config appsettings set \
    --name intelligent-document-analysis \
    --resource-group rg-data-ai-eng-con \
    --settings \
        DOCKER_REGISTRY_SERVER_URL=https://1a27253794c8488f83ef31437e7d1248.azurecr.io \
        DOCKER_REGISTRY_SERVER_USERNAME=$ACR_USERNAME \
        DOCKER_REGISTRY_SERVER_PASSWORD=$ACR_PASSWORD
```

#### Step 3: Build and Push Image
```bash
# Login to ACR
az acr login --name 1a27253794c8488f83ef31437e7d1248

# Build with correct platform
docker buildx build \
    --platform linux/amd64 \
    -f Dockerfile.azure \
    -t 1a27253794c8488f83ef31437e7d1248.azurecr.io/intelligent-document-analysis:latest \
    --push .
```

#### Step 4: Restart App
```bash
az webapp restart --name intelligent-document-analysis --resource-group rg-data-ai-eng-con
```

## Common Issues and Solutions

### Issue 1: Container Won't Start
**Symptoms**: App shows "Container failed to start" in Azure portal
**Causes**:
- Invalid Docker image reference
- Missing environment variables
- Port configuration issues

**Solutions**:
```bash
# Check container logs
az webapp log tail --name intelligent-document-analysis --resource-group rg-data-ai-eng-con

# Verify image exists
az acr repository show --name 1a27253794c8488f83ef31437e7d1248 --image intelligent-document-analysis:latest

# Check app settings
az webapp config appsettings list --name intelligent-document-analysis --resource-group rg-data-ai-eng-con
```

### Issue 2: Health Check Failures
**Symptoms**: Health endpoint returns 503 or timeout
**Causes**:
- Application not starting properly
- Port configuration mismatch
- Missing dependencies

**Solutions**:
```bash
# Check if app is running
curl -f https://intelligent-document-analysis.azurewebsites.net/health

# Check startup logs
az webapp log tail --name intelligent-document-analysis --resource-group rg-data-ai-eng-con --provider application

# Verify port configuration
az webapp config show --name intelligent-document-analysis --resource-group rg-data-ai-eng-con --query "siteConfig.appSettings[?name=='WEBSITES_PORT']"
```

### Issue 3: Database Connection Issues
**Symptoms**: Database connection errors in logs
**Causes**:
- Invalid connection string
- Firewall rules blocking access
- SSL configuration issues

**Solutions**:
```bash
# Test database connectivity
az postgres flexible-server show --name your-server --resource-group rg-data-ai-eng-con

# Check firewall rules
az postgres flexible-server firewall-rule list --name your-server --resource-group rg-data-ai-eng-con

# Update connection string
az webapp config appsettings set \
    --name intelligent-document-analysis \
    --resource-group rg-data-ai-eng-con \
    --settings DATABASE_URL="postgresql://user:pass@server.postgres.database.azure.com:5432/db?sslmode=require"
```

### Issue 4: Memory Issues
**Symptoms**: Container killed due to memory limits
**Causes**:
- Large model loading
- Memory leaks
- Insufficient App Service plan

**Solutions**:
```bash
# Check memory usage
az monitor metrics list \
    --resource /subscriptions/{subscription-id}/resourceGroups/rg-data-ai-eng-con/providers/Microsoft.Web/sites/intelligent-document-analysis \
    --metric "MemoryPercentage"

# Upgrade App Service plan
az appservice plan update \
    --name plan-document-analysis \
    --resource-group rg-data-ai-eng-con \
    --sku P2V2
```

## Monitoring and Debugging

### View Real-time Logs
```bash
# Application logs
az webapp log tail --name intelligent-document-analysis --resource-group rg-data-ai-eng-con

# Container logs
az webapp log tail --name intelligent-document-analysis --resource-group rg-data-ai-eng-con --provider docker

# All logs
az webapp log tail --name intelligent-document-analysis --resource-group rg-data-ai-eng-con --provider all
```

### Check Application Status
```bash
# App status
az webapp show --name intelligent-document-analysis --resource-group rg-data-ai-eng-con --query state

# Container status
az webapp show --name intelligent-document-analysis --resource-group rg-data-ai-eng-con --query "siteConfig.linuxFxVersion"

# Health check
curl -f https://intelligent-document-analysis.azurewebsites.net/health
```

### Performance Monitoring
```bash
# CPU and Memory metrics
az monitor metrics list \
    --resource /subscriptions/{subscription-id}/resourceGroups/rg-data-ai-eng-con/providers/Microsoft.Web/sites/intelligent-document-analysis \
    --metric "CpuPercentage,MemoryPercentage" \
    --interval PT1M \
    --start-time 2024-01-01T00:00:00Z

# Response time metrics
az monitor metrics list \
    --resource /subscriptions/{subscription-id}/resourceGroups/rg-data-ai-eng-con/providers/Microsoft.Web/sites/intelligent-document-analysis \
    --metric "AverageResponseTime" \
    --interval PT1M
```

## Environment Variables Checklist

Ensure these environment variables are set:

### Required Azure Services
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_STORAGE_ACCOUNT_NAME`
- `AZURE_STORAGE_ACCOUNT_KEY`
- `DATABASE_URL`
- `REDIS_URL`

### Application Configuration
- `WEBSITES_PORT=8000`
- `STREAMLIT_SERVER_PORT=8000`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `STREAMLIT_SERVER_HEADLESS=true`

### Security
- `SECRET_KEY`
- `DEBUG=false`
- `ENVIRONMENT=production`

## Deployment Verification

### 1. Health Check
```bash
curl -f https://intelligent-document-analysis.azurewebsites.net/health
```

### 2. Application Access
```bash
curl -f https://intelligent-document-analysis.azurewebsites.net/
```

### 3. Log Analysis
```bash
# Check for errors
az webapp log tail --name intelligent-document-analysis --resource-group rg-data-ai-eng-con | grep -i error

# Check startup sequence
az webapp log tail --name intelligent-document-analysis --resource-group rg-data-ai-eng-con | grep -i "streamlit\|startup\|ready"
```

## Emergency Recovery

### If Deployment Completely Fails
```bash
# Delete and recreate web app
az webapp delete --name intelligent-document-analysis --resource-group rg-data-ai-eng-con

# Recreate using deployment script
./scripts/deploy-azure.sh
```

### If Container Registry Issues
```bash
# Recreate ACR
az acr delete --name 1a27253794c8488f83ef31437e7d1248 --resource-group rg-data-ai-eng-con --yes
az acr create --name 1a27253794c8488f83ef31437e7d1248 --resource-group rg-data-ai-eng-con --sku Basic --admin-enabled true
```

## Support Resources

- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [Azure Container Registry Documentation](https://docs.microsoft.com/en-us/azure/container-registry/)
- [Docker Multi-platform Builds](https://docs.docker.com/buildx/working-with-buildx/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

## Contact Information

For additional support:
- Check Azure Service Health: https://status.azure.com/
- Azure Support: https://azure.microsoft.com/en-us/support/
- Streamlit Community: https://discuss.streamlit.io/
