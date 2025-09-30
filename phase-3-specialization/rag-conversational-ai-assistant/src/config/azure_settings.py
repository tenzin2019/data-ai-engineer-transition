"""
Azure-specific configuration settings for RAG Conversational AI Assistant.
Based on successful patterns from intelligent-document-analysis deployment.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class AzureSettings(BaseSettings):
    """Azure-specific application settings with environment variable support."""
    
    # Application
    app_name: str = "RAG Conversational AI Assistant"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"
    
    # Database Configuration (with SQLite fallback for Azure)
    database_url: str = Field(
        default="sqlite:///./rag_assistant.db",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Vector Store Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    # Azure OpenAI Services
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", env="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment_name: str = Field(default="gpt-4o", env="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-ada-002", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    # OpenAI Configuration (fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model_default: str = Field(default="gpt-4o", env="OPENAI_MODEL_DEFAULT")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model_default: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL_DEFAULT")
    
    # Model Selection Strategy (inherited from successful deployment)
    primary_model: str = Field(default="gpt-4o", env="PRIMARY_MODEL")
    secondary_model: str = Field(default="gpt-4o-mini", env="SECONDARY_MODEL")
    budget_model: str = Field(default="gpt-3.5-turbo", env="BUDGET_MODEL")
    
    # Azure Storage (if needed for documents)
    azure_storage_account_name: Optional[str] = Field(default=None, env="AZURE_STORAGE_ACCOUNT_NAME")
    azure_storage_account_key: Optional[str] = Field(default=None, env="AZURE_STORAGE_ACCOUNT_KEY")
    azure_storage_container_name: str = Field(default="documents", env="AZURE_STORAGE_CONTAINER_NAME")
    
    # File Upload Configuration (Azure App Service optimized)
    max_file_size_standard: int = Field(default=15 * 1024 * 1024, env="MAX_FILE_SIZE_STANDARD")  # 15MB
    max_file_size_large: int = Field(default=200 * 1024 * 1024, env="MAX_FILE_SIZE_LARGE")  # 200MB
    allowed_file_types: list = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/msword",
        "application/vnd.ms-excel",
        "text/plain"
    ]
    
    # AI Processing Configuration
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    temperature: float = Field(default=0.3, env="TEMPERATURE")
    max_document_length: int = Field(default=100000, env="MAX_DOCUMENT_LENGTH")  # characters
    
    # Security (inherited from successful deployment)
    secret_key: str = Field(default="azure-rag-assistant-secret-key", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = "HS256"
    
    # CORS Configuration (Azure App Service optimized)
    cors_origins: list = Field(
        default=["*"],  # Configure properly for production
        env="CORS_ORIGINS"
    )
    
    # Logging (Azure App Service compatible)
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Redis Configuration (optional for Azure)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Azure App Service specific settings
    websites_port: int = Field(default=8000, env="WEBSITES_PORT")
    websites_enable_app_service_storage: str = Field(default="true", env="WEBSITES_ENABLE_APP_SERVICE_STORAGE")
    websites_container_start_time_limit: int = Field(default=1800, env="WEBSITES_CONTAINER_START_TIME_LIMIT")
    
    # Production optimizations
    uvicorn_workers: int = Field(default=1, env="UVICORN_WORKERS")
    uvicorn_host: str = Field(default="0.0.0.0", env="UVICORN_HOST")
    uvicorn_port: int = Field(default=8000, env="UVICORN_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_ai_provider_priority(self) -> list[str]:
        """Get AI provider priority based on available configurations."""
        providers = []
        
        # Azure OpenAI first (best for production)
        if self.azure_openai_endpoint and self.azure_openai_api_key:
            providers.append("azure_openai")
        
        # OpenAI second
        if self.openai_api_key:
            providers.append("openai")
        
        # Anthropic third
        if self.anthropic_api_key:
            providers.append("anthropic")
        
        return providers
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def get_database_config(self) -> dict:
        """Get database configuration for Azure deployment."""
        return {
            "url": self.database_url,
            "echo": self.database_echo and not self.is_production(),
            "pool_pre_ping": True,  # Important for Azure PostgreSQL
            "pool_recycle": 3600,   # Recycle connections every hour
        }
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration for Azure App Service."""
        if self.is_production():
            # In production, configure specific origins
            return {
                "allow_origins": self.cors_origins,
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["*"],
            }
        else:
            # In development, allow all origins
            return {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
            }


# Global settings instance
azure_settings = AzureSettings()


def get_azure_settings() -> AzureSettings:
    """Get Azure-specific application settings."""
    return azure_settings


def validate_azure_configuration() -> dict[str, bool]:
    """Validate Azure configuration and return status."""
    settings = get_azure_settings()
    
    validation_results = {
        "basic_config": bool(settings.app_name and settings.app_version),
        "database_config": bool(settings.database_url),
        "ai_provider_config": len(settings.get_ai_provider_priority()) > 0,
        "azure_openai_config": bool(settings.azure_openai_endpoint and settings.azure_openai_api_key),
        "openai_fallback_config": bool(settings.openai_api_key),
        "anthropic_fallback_config": bool(settings.anthropic_api_key),
        "storage_config": settings.azure_storage_account_name is not None,
        "redis_config": settings.redis_url is not None,
    }
    
    validation_results["overall_ready"] = all([
        validation_results["basic_config"],
        validation_results["database_config"],
        validation_results["ai_provider_config"],
    ])
    
    return validation_results
