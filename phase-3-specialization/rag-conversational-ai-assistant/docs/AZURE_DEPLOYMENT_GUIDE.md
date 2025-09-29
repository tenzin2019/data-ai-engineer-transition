# Azure Cloud Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the RAG Conversational AI Assistant to Microsoft Azure cloud platform, leveraging Azure's managed services for optimal performance, security, and scalability.

## Azure Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Azure Cloud Architecture                 │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│   Frontend      │   API Gateway   │   Load Balancer │  CDN    │
│   (App Service) │   (API Mgmt)    │   (Front Door)  │ (CDN)   │
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│                 │                 │                 │         │
│  React SPA      │  Rate Limiting  │  Global Load    │ Static  │
│  Static Hosting │  Authentication│  Balancing      │ Assets  │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Application Services                     │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Main API       │  Background     │  WebSocket      │  Admin  │
│  (App Service)  │  Jobs (Functions)│  (SignalR)     │  Portal │
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│  FastAPI        │  Document       │  Real-time      │  Admin  │
│  Application    │  Processing     │  Communication  │  UI     │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Azure AI/ML Services                        │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  OpenAI Service │  Cognitive      │  Document       │  Search │
│  (Azure OpenAI) │  Services       │  Intelligence   │  (AI Search)│
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│  GPT-4, GPT-3.5 │  Speech, Vision │  Form Recognizer│  Vector │
│  Embeddings     │  Translator     │  Custom Models  │  Search │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Storage Services                     │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Database       │  Cache          │  File Storage   │  Search │
│  (PostgreSQL)   │  (Redis Cache)  │  (Blob Storage) │  (Elasticsearch)│
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│  Azure Database │  Azure Cache    │  Azure Blob     │  Azure  │
│  for PostgreSQL │  for Redis      │  Storage        │  Search │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring & Security                       │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Monitoring     │  Logging        │  Security       │  Backup │
│  (Application   │  (Log Analytics)│  (Key Vault)    │  (Backup)│
│   Insights)     │                 │                 │         │
├─────────────────┼─────────────────┼─────────────────┼─────────┤
│  Performance    │  Centralized    │  Secrets        │  Point  │
│  Monitoring     │  Logging        │  Management     │  in Time│
└─────────────────┴─────────────────┴─────────────────┴─────────┘
```

## Azure Services Used

### Core Application Services
- **Azure App Service**: Host the FastAPI backend and React frontend
- **Azure Functions**: Serverless background processing
- **Azure SignalR Service**: Real-time WebSocket communication
- **Azure API Management**: API gateway and management

### AI/ML Services
- **Azure OpenAI Service**: GPT models and embeddings
- **Azure Cognitive Services**: Speech, Vision, Translator
- **Azure Document Intelligence**: Document processing
- **Azure AI Search**: Vector search and full-text search

### Data Services
- **Azure Database for PostgreSQL**: Primary database
- **Azure Cache for Redis**: Caching and session storage
- **Azure Blob Storage**: File storage and document repository
- **Azure Cosmos DB**: NoSQL for feedback and analytics data

### Infrastructure Services
- **Azure Front Door**: Global load balancing and CDN
- **Azure Application Gateway**: Layer 7 load balancing
- **Azure Virtual Network**: Network isolation and security
- **Azure Key Vault**: Secrets and certificate management

### Monitoring & Security
- **Azure Application Insights**: Application performance monitoring
- **Azure Log Analytics**: Centralized logging
- **Azure Monitor**: Infrastructure monitoring
- **Azure Security Center**: Security monitoring and compliance

## Prerequisites

### Azure Account Setup
1. **Azure Subscription**: Active Azure subscription with appropriate permissions
2. **Azure CLI**: Install and configure Azure CLI
3. **Service Principal**: Create service principal for CI/CD
4. **Resource Groups**: Create resource groups for different environments

### Required Azure Services
1. **Azure OpenAI Service**: Request access and provision resources
2. **Azure Database for PostgreSQL**: Flexible Server or Single Server
3. **Azure Cache for Redis**: Standard or Premium tier
4. **Azure Storage Account**: Standard or Premium tier
5. **Azure Key Vault**: Standard tier

## Configuration Files

### Azure App Service Configuration

#### appsettings.json
```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "AllowedHosts": "*",
  "ConnectionStrings": {
    "DefaultConnection": "Server=tcp:rag-postgres-server.postgres.database.azure.com,1433;Initial Catalog=rag_assistant;Persist Security Info=False;User ID=rag_admin;Password={password};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;",
    "Redis": "rag-cache.redis.cache.windows.net:6380,password={password},ssl=True,abortConnect=False"
  },
  "Azure": {
    "OpenAI": {
      "Endpoint": "https://rag-openai.openai.azure.com/",
      "ApiKey": "{key-vault-reference}",
      "ApiVersion": "2024-02-15-preview",
      "DeploymentName": "gpt-4",
      "EmbeddingDeploymentName": "text-embedding-ada-002"
    },
    "Storage": {
      "ConnectionString": "{key-vault-reference}",
      "ContainerName": "documents"
    },
    "Search": {
      "Endpoint": "https://rag-search.search.windows.net",
      "ApiKey": "{key-vault-reference}",
      "IndexName": "rag-documents"
    },
    "KeyVault": {
      "VaultUrl": "https://rag-keyvault.vault.azure.net/",
      "ClientId": "{managed-identity}",
      "ClientSecret": "{managed-identity}"
    }
  },
  "ApplicationInsights": {
    "ConnectionString": "{application-insights-connection-string}"
  }
}
```

#### web.config (for App Service)
```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified"/>
    </handlers>
    <httpPlatform processPath="D:\home\Python39\python.exe"
                  arguments="D:\home\site\wwwroot\src\api\main.py"
                  stdoutLogEnabled="true"
                  stdoutLogFile="D:\home\LogFiles\python.log"
                  startupTimeLimit="60"
                  startupRetryCount="3">
      <environmentVariables>
        <environmentVariable name="PORT" value="8000" />
        <environmentVariable name="PYTHONPATH" value="D:\home\site\wwwroot" />
      </environmentVariables>
    </httpPlatform>
  </system.webServer>
</configuration>
```

### Azure Functions Configuration

#### host.json
```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "excludedTypes": "Request"
      }
    }
  },
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[3.*, 4.0.0)"
  },
  "functionTimeout": "00:10:00",
  "retry": {
    "strategy": "exponentialBackoff",
    "maxRetryCount": 3,
    "minimumInterval": "00:00:02",
    "maximumInterval": "00:00:30"
  }
}
```

#### local.settings.json
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AzureWebJobsFeatureFlags": "EnableWorkerIndexing",
    "AZURE_OPENAI_ENDPOINT": "https://rag-openai.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "{your-api-key}",
    "AZURE_STORAGE_CONNECTION_STRING": "{your-storage-connection-string}",
    "AZURE_SEARCH_ENDPOINT": "https://rag-search.search.windows.net",
    "AZURE_SEARCH_API_KEY": "{your-search-api-key}"
  }
}
```

## Deployment Strategies

### 1. Azure App Service Deployment

#### Using Azure CLI
```bash
# Create resource group
az group create --name rag-assistant-rg --location eastus

# Create App Service plan
az appservice plan create \
  --name rag-assistant-plan \
  --resource-group rag-assistant-rg \
  --sku P1V2 \
  --is-linux

# Create web app
az webapp create \
  --resource-group rag-assistant-rg \
  --plan rag-assistant-plan \
  --name rag-assistant-api \
  --runtime "PYTHON|3.11"

# Configure app settings
az webapp config appsettings set \
  --resource-group rag-assistant-rg \
  --name rag-assistant-api \
  --settings @appsettings.json

# Deploy from GitHub
az webapp deployment source config \
  --resource-group rag-assistant-rg \
  --name rag-assistant-api \
  --repo-url https://github.com/your-org/rag-conversational-ai-assistant \
  --branch main \
  --manual-integration
```

#### Using Azure DevOps Pipeline
```yaml
# azure-pipelines.yml
trigger:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  azureServiceConnection: 'rag-assistant-connection'
  webAppName: 'rag-assistant-api'
  resourceGroupName: 'rag-assistant-rg'

stages:
- stage: Build
  displayName: Build stage
  jobs:
  - job: Build
    displayName: Build
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.11'
        addToPath: true
        architecture: 'x64'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e ".[dev]"
      displayName: 'Install dependencies'
    
    - script: |
        pytest tests/ -v --cov=src --cov-report=xml
      displayName: 'Run tests'
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: '**/test-*.xml'
        testRunTitle: 'Publish test results for Python $(python.version)'
    
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: Cobertura
        summaryFileLocation: '$(System.DefaultWorkingDirectory)/**/coverage.xml'
        failIfCoverageEmpty: true

- stage: Deploy
  displayName: 'Deploy to Azure'
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: Deploy
    displayName: 'Deploy to Azure App Service'
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureWebApp@1
            displayName: 'Deploy to Azure Web App'
            inputs:
              azureSubscription: $(azureServiceConnection)
              appName: $(webAppName)
              resourceGroupName: $(resourceGroupName)
              package: $(System.DefaultWorkingDirectory)
```

### 2. Azure Container Instances (Alternative)

#### Docker Compose for Azure
```yaml
# docker-compose.azure.yml
version: '3.8'

services:
  api:
    image: rag-assistant-api:latest
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  frontend:
    image: rag-assistant-frontend:latest
    environment:
      - REACT_APP_API_URL=${API_URL}
    ports:
      - "3000:3000"
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
```

## Azure-Specific Features

### 1. Managed Identity Integration
```python
# src/core/azure/identity.py
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import os

class AzureIdentityManager:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
        self.secret_client = SecretClient(
            vault_url=self.key_vault_url,
            credential=self.credential
        )
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault."""
        secret = self.secret_client.get_secret(secret_name)
        return secret.value
    
    def get_storage_client(self) -> BlobServiceClient:
        """Get Azure Storage client using managed identity."""
        account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        return BlobServiceClient(
            account_url=account_url,
            credential=self.credential
        )
```

### 2. Azure Application Insights Integration
```python
# src/monitoring/azure_insights.py
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace import config_integration
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer
import logging

class AzureInsightsManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.setup_logging()
        self.setup_tracing()
    
    def setup_logging(self):
        """Setup Azure Application Insights logging."""
        logger = logging.getLogger(__name__)
        logger.addHandler(AzureLogHandler(
            connection_string=self.connection_string
        ))
        logger.setLevel(logging.INFO)
    
    def setup_tracing(self):
        """Setup Azure Application Insights tracing."""
        exporter = AzureExporter(
            connection_string=self.connection_string
        )
        tracer = Tracer(
            exporter=exporter,
            sampler=ProbabilitySampler(rate=1.0)
        )
        config_integration.trace_integrations(['requests'])
```

### 3. Azure AI Search Integration
```python
# src/core/azure/search.py
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Any

class AzureSearchManager:
    def __init__(self, endpoint: str, api_key: str, index_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.index_name = index_name
        self.credential = AzureKeyCredential(api_key)
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=self.credential
        )
    
    async def search_documents(self, query: str, top: int = 10) -> List[Dict[str, Any]]:
        """Search documents using Azure AI Search."""
        results = self.search_client.search(
            search_text=query,
            top=top,
            include_total_count=True
        )
        return [dict(result) for result in results]
    
    async def upload_documents(self, documents: List[Dict[str, Any]]):
        """Upload documents to Azure AI Search index."""
        self.search_client.upload_documents(documents)
```

## Security Configuration

### 1. Azure Key Vault Integration
```python
# src/security/azure_keyvault.py
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import os

class AzureKeyVaultManager:
    def __init__(self):
        self.vault_url = os.getenv("AZURE_KEY_VAULT_URL")
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=self.vault_url,
            credential=self.credential
        )
    
    def get_database_connection_string(self) -> str:
        """Get database connection string from Key Vault."""
        return self.client.get_secret("database-connection-string").value
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from Key Vault."""
        return self.client.get_secret("openai-api-key").value
    
    def get_redis_connection_string(self) -> str:
        """Get Redis connection string from Key Vault."""
        return self.client.get_secret("redis-connection-string").value
```

### 2. Azure Active Directory Integration
```python
# src/security/azure_ad.py
from azure.identity import DefaultAzureCredential
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
import msal

class AzureADManager:
    def __init__(self, tenant_id: str, client_id: str, client_secret: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.app = msal.ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=f"https://login.microsoftonline.com/{tenant_id}"
        )
    
    def get_access_token(self, scope: str) -> str:
        """Get access token for Azure AD."""
        result = self.app.acquire_token_silent([scope], account=None)
        if not result:
            result = self.app.acquire_token_for_client(scopes=[scope])
        return result.get("access_token")
```

## Monitoring and Logging

### 1. Azure Application Insights Configuration
```python
# src/monitoring/azure_monitoring.py
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import stats as stats_module
from opencensus.stats import measure as measure_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
import time

class AzureMonitoringManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.setup_metrics()
    
    def setup_metrics(self):
        """Setup Azure Application Insights metrics."""
        # Create custom metrics
        self.query_count = measure_module.MeasureInt(
            "query_count", "Number of queries processed", "1"
        )
        self.query_duration = measure_module.MeasureFloat(
            "query_duration", "Query processing duration", "ms"
        )
        self.error_count = measure_module.MeasureInt(
            "error_count", "Number of errors", "1"
        )
        
        # Create views
        query_count_view = view_module.View(
            "query_count_view",
            "Number of queries processed",
            [],
            self.query_count,
            view_module.AggregationType.COUNT
        )
        
        query_duration_view = view_module.View(
            "query_duration_view",
            "Query processing duration",
            [],
            self.query_duration,
            view_module.AggregationType.DISTRIBUTION
        )
        
        # Register views
        stats = stats_module.stats
        view_manager = stats.view_manager
        view_manager.register_view(query_count_view)
        view_manager.register_view(query_duration_view)
        
        # Setup exporter
        exporter = metrics_exporter.new_metrics_exporter(
            connection_string=self.connection_string
        )
        view_manager.register_exporter(exporter)
    
    def record_query(self, duration: float, success: bool = True):
        """Record query metrics."""
        stats = stats_module.stats
        measure_map = stats.measure_map
        measure_map.record_int_measure(self.query_count, 1)
        measure_map.record_float_measure(self.query_duration, duration)
        if not success:
            measure_map.record_int_measure(self.error_count, 1)
```

## Cost Optimization

### 1. Azure Cost Management
```python
# src/utils/azure_cost_optimization.py
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.costmanagement.models import QueryDefinition
from datetime import datetime, timedelta
import pandas as pd

class AzureCostOptimizer:
    def __init__(self, credential, subscription_id: str):
        self.credential = credential
        self.subscription_id = subscription_id
        self.client = CostManagementClient(credential, subscription_id)
    
    def get_cost_analysis(self, start_date: datetime, end_date: datetime):
        """Get cost analysis for the specified period."""
        query = QueryDefinition(
            type="ActualCost",
            timeframe="Custom",
            time_period={
                "from": start_date,
                "to": end_date
            },
            dataset={
                "granularity": "Daily",
                "aggregation": {
                    "totalCost": {"name": "PreTaxCost", "function": "Sum"}
                },
                "grouping": [
                    {"type": "Dimension", "name": "ResourceGroupName"},
                    {"type": "Dimension", "name": "ResourceType"}
                ]
            }
        )
        
        result = self.client.query.usage(
            scope=f"/subscriptions/{self.subscription_id}",
            parameters=query
        )
        
        return result
```

## Deployment Checklist

### Pre-Deployment
- [ ] Azure subscription and resource group created
- [ ] Azure OpenAI Service provisioned and configured
- [ ] Database and cache services provisioned
- [ ] Key Vault configured with secrets
- [ ] Application Insights workspace created
- [ ] Azure AD app registration completed

### Deployment
- [ ] App Service applications deployed
- [ ] Azure Functions deployed
- [ ] SignalR service configured
- [ ] API Management configured
- [ ] Front Door/CDN configured
- [ ] Custom domain configured (if applicable)

### Post-Deployment
- [ ] Health checks passing
- [ ] Monitoring and alerting configured
- [ ] Backup policies configured
- [ ] Security scanning completed
- [ ] Performance testing completed
- [ ] Documentation updated

## Troubleshooting

### Common Issues
1. **Authentication Failures**: Check managed identity configuration
2. **Database Connection Issues**: Verify connection string and firewall rules
3. **Performance Issues**: Check App Service plan scaling and resource limits
4. **Monitoring Issues**: Verify Application Insights configuration

### Support Resources
- Azure Documentation: https://docs.microsoft.com/azure/
- Azure Support: https://azure.microsoft.com/support/
- Community Forums: https://docs.microsoft.com/answers/topics/azure.html

This comprehensive Azure deployment guide ensures a robust, scalable, and secure deployment of the RAG Conversational AI Assistant on Microsoft Azure cloud platform.
