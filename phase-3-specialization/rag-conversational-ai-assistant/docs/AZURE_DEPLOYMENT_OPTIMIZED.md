# Azure App Service Deployment Guide - RAG Conversational AI Assistant

This guide provides step-by-step instructions for deploying the RAG Conversational AI Assistant to Azure App Service, based on the successful deployment patterns from the intelligent-document-analysis project.

## 🚀 Live Application Architecture

Based on the proven architecture from: https://intelligent-document-analysis.azurewebsites.net

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Deployment](#quick-start-deployment)
3. [Detailed Setup](#detailed-setup)
4. [Configuration](#configuration)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Troubleshooting](#troubleshooting)
7. [Cost Optimization](#cost-optimization)

## Prerequisites

### Required Tools
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) (latest version)
- [Docker Desktop](https://www.docker.com/products/docker-desktop) with buildx support
- [Git](https://git-scm.com/downloads)
- [Python 3.11+](https://www.python.org/downloads/) (for local testing)

### Azure Account Requirements
- Active Azure subscription
- Owner or Contributor access to the subscription
- Sufficient quota for:
  - App Service Plan (Linux)
  - Container Registry (shared with existing deployment)
  - Optional: Database for PostgreSQL, Cache for Redis

### API Keys Required
- **Azure OpenAI**: Endpoint and API key (recommended)
- **OpenAI**: API key (fallback)
- **Anthropic**: API key (optional, for Claude models)

## Quick Start Deployment

### 1. Clone and Prepare

```bash
git clone <repository-url>
cd phase-3-specialization/rag-conversational-ai-assistant

# Make scripts executable
chmod +x scripts/*.sh
```

### 2. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys (minimum required)
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-azure-openai-key
# Or alternatively:
# OPENAI_API_KEY=your-openai-key
```

### 3. Deploy to Azure

```bash
# Login to Azure
az login

# Run optimized deployment script
./scripts/deploy-azure-optimized.sh

# Verify deployment
./scripts/verify-deployment.sh
```

### 4. Configure API Keys

```bash
# Set your AI service API keys
az webapp config appsettings set \
    --name rag-conversational-ai-assistant \
    --resource-group rg-data-ai-eng-con \
    --settings \
        AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
        AZURE_OPENAI_API_KEY="your-azure-openai-key"

# Or for OpenAI
az webapp config appsettings set \
    --name rag-conversational-ai-assistant \
    --resource-group rg-data-ai-eng-con \
    --settings \
        OPENAI_API_KEY="your-openai-key"
```

## Detailed Setup

### Azure Resources Configuration

The deployment uses the same proven infrastructure as the successful intelligent-document-analysis deployment:

- **Resource Group**: `rg-data-ai-eng-con`
- **Location**: `australiaeast`
- **Container Registry**: `1a27253794c8488f83ef31437e7d1248.azurecr.io`
- **App Service Plan**: `plan-rag-assistant` (P1V2 SKU)

### Architecture Decisions

Based on the successful deployment, we use:

1. **Single Container Deployment**: Unlike the original multi-container setup, this optimized version uses a single FastAPI container for better performance on Azure App Service.

2. **SQLite Database**: For simplicity and reliability, using SQLite instead of PostgreSQL reduces complexity while maintaining functionality.

3. **Shared Container Registry**: Leveraging the existing registry reduces costs and management overhead.

4. **Proven Configuration**: Using the exact same environment variables and health check patterns that work in production.

## Configuration

### Environment Variables

The deployment automatically configures these essential settings:

```bash
# Azure App Service Settings
WEBSITES_PORT=8000
WEBSITES_ENABLE_APP_SERVICE_STORAGE=false
WEBSITES_CONTAINER_START_TIME_LIMIT=1800
PYTHONPATH=/app

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL="sqlite:///./rag_assistant.db"
CHROMA_PERSIST_DIRECTORY="./chroma_db"
```

### AI Provider Configuration

Set your preferred AI provider:

```bash
# Azure OpenAI (Recommended for production)
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-key"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"

# OpenAI (Fallback)
OPENAI_API_KEY="your-openai-key"
OPENAI_MODEL_DEFAULT="gpt-4o"

# Anthropic (Optional)
ANTHROPIC_API_KEY="your-anthropic-key"
```

## Application Features

Your deployed RAG system will have:

### API Endpoints
- **Main API**: `https://rag-conversational-ai-assistant.azurewebsites.net/`
- **API Documentation**: `https://rag-conversational-ai-assistant.azurewebsites.net/docs`
- **Health Check**: `https://rag-conversational-ai-assistant.azurewebsites.net/health`
- **Metrics**: `https://rag-conversational-ai-assistant.azurewebsites.net/metrics`

### Core Functionality
- **Document Upload**: Upload and process PDF, DOCX, XLSX, TXT files
- **RAG Queries**: Ask questions about uploaded documents
- **Conversation Management**: Maintain conversation context
- **Multi-Provider LLM**: Automatic failover between AI providers
- **Vector Search**: Semantic search through document collections

## Monitoring and Maintenance

### Health Monitoring

The application includes comprehensive health checks:

```bash
# Check overall health
curl https://rag-conversational-ai-assistant.azurewebsites.net/health

# Check readiness (Kubernetes-style)
curl https://rag-conversational-ai-assistant.azurewebsites.net/health/ready

# Check liveness
curl https://rag-conversational-ai-assistant.azurewebsites.net/health/live

# Get metrics
curl https://rag-conversational-ai-assistant.azurewebsites.net/metrics
```

### Application Logs

```bash
# View live logs
az webapp log tail \
    --name rag-conversational-ai-assistant \
    --resource-group rg-data-ai-eng-con

# Download logs
az webapp log download \
    --name rag-conversational-ai-assistant \
    --resource-group rg-data-ai-eng-con
```

### Performance Monitoring

Monitor through Azure Portal:
- **Application Insights**: Performance metrics and error tracking
- **App Service Metrics**: CPU, memory, and request metrics
- **Container Metrics**: Docker container performance

## Troubleshooting

### Common Issues

#### 1. Application Won't Start
```bash
# Check container logs
az webapp log tail --name rag-conversational-ai-assistant --resource-group rg-data-ai-eng-con

# Restart application
az webapp restart --name rag-conversational-ai-assistant --resource-group rg-data-ai-eng-con

# Check environment variables
az webapp config appsettings list --name rag-conversational-ai-assistant --resource-group rg-data-ai-eng-con
```

#### 2. AI Provider Errors
```bash
# Verify API keys are set
curl -s https://rag-conversational-ai-assistant.azurewebsites.net/health | jq '.configuration'

# Test with simple query
curl -X POST https://rag-conversational-ai-assistant.azurewebsites.net/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, can you help me?"}'
```

#### 3. File Upload Issues
```bash
# Check file size limits
curl -s https://rag-conversational-ai-assistant.azurewebsites.net/health | jq '.system'

# Test upload endpoint
curl -X POST https://rag-conversational-ai-assistant.azurewebsites.net/upload \
  -F "file=@test.pdf"
```

### Debug Commands

```bash
# Check app status
az webapp show --name rag-conversational-ai-assistant --resource-group rg-data-ai-eng-con

# View configuration
az webapp config show --name rag-conversational-ai-assistant --resource-group rg-data-ai-eng-con

# Check container image
az acr repository show --name 1a27253794c8488f83ef31437e7d1248 --image rag-conversational-ai-assistant:latest
```

## Cost Optimization

### Resource Optimization
- **App Service Plan**: P1V2 SKU provides good performance/cost ratio
- **Shared Container Registry**: Reduces storage and management costs
- **SQLite Database**: Eliminates database hosting costs
- **Single Container**: Reduces complexity and resource usage

### Monitoring Costs
```bash
# View resource costs
az consumption usage list --top 10

# Set up cost alerts
az monitor action-group create \
    --name rag-cost-alerts \
    --resource-group rg-data-ai-eng-con
```

## Security Best Practices

### Application Security
- **Non-root Container**: Application runs as non-root user
- **Environment Variables**: Sensitive data stored securely
- **CORS Configuration**: Properly configured for production
- **Input Validation**: All inputs validated and sanitized

### Network Security
- **HTTPS Only**: All communication encrypted
- **Azure App Service**: Built-in DDoS protection
- **Private Container Registry**: Images stored securely

## Scaling Considerations

### Horizontal Scaling
```bash
# Scale out to multiple instances
az appservice plan update \
    --name plan-rag-assistant \
    --resource-group rg-data-ai-eng-con \
    --number-of-workers 3

# Enable auto-scaling
az monitor autoscale create \
    --resource-group rg-data-ai-eng-con \
    --resource /subscriptions/{subscription-id}/resourceGroups/rg-data-ai-eng-con/providers/Microsoft.Web/serverfarms/plan-rag-assistant \
    --name rag-autoscale
```

### Vertical Scaling
```bash
# Upgrade to higher SKU
az appservice plan update \
    --name plan-rag-assistant \
    --resource-group rg-data-ai-eng-con \
    --sku P2V2
```

## Next Steps

1. **Test the Deployment**: Upload documents and test RAG functionality
2. **Configure Monitoring**: Set up Application Insights and alerts
3. **Customize for Your Use Case**: Modify prompts and add domain-specific logic
4. **Set Up CI/CD**: Automate deployments with GitHub Actions
5. **Add Authentication**: Implement user authentication if needed

## Support Resources

- **Successful Reference**: https://intelligent-document-analysis.azurewebsites.net
- **Azure Documentation**: [Azure App Service](https://docs.microsoft.com/en-us/azure/app-service/)
- **FastAPI Documentation**: [FastAPI](https://fastapi.tiangolo.com/)
- **LangChain Documentation**: [LangChain](https://python.langchain.com/)

---

**Status**: ✅ Production Ready (Based on proven deployment patterns)  
**Last Updated**: December 2024  
**Version**: 1.0.0
