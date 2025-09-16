"""
Configuration settings for the Intelligent Document Analysis System.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Intelligent Document Analysis System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/document_analysis",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Azure AI Services
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(default="2023-12-01-preview", env="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment_name: str = Field(default="gpt-4o", env="AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Model Selection Strategy
    primary_model: str = Field(default="gpt-4o", env="PRIMARY_MODEL")  # For complex documents
    secondary_model: str = Field(default="gpt-4o-mini", env="SECONDARY_MODEL")  # For simple documents
    budget_model: str = Field(default="gpt-3.5-turbo", env="BUDGET_MODEL")  # For high volume
    
    # Model selection criteria
    use_primary_for_complex: bool = Field(default=True, env="USE_PRIMARY_FOR_COMPLEX")
    complex_document_types: list = Field(default=["legal", "financial", "technical", "medical"], env="COMPLEX_DOCUMENT_TYPES")
    max_tokens_budget_threshold: int = Field(default=2000, env="MAX_TOKENS_BUDGET_THRESHOLD")  # Use budget model for shorter docs
    
    # Azure Document Intelligence
    azure_document_intelligence_endpoint: Optional[str] = Field(
        default=None, env="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
    )
    azure_document_intelligence_api_key: Optional[str] = Field(
        default=None, env="AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
    )
    
    # Azure Storage
    azure_storage_account_name: Optional[str] = Field(default=None, env="AZURE_STORAGE_ACCOUNT_NAME")
    azure_storage_account_key: Optional[str] = Field(default=None, env="AZURE_STORAGE_ACCOUNT_KEY")
    azure_storage_container_name: str = Field(default="documents", env="AZURE_STORAGE_CONTAINER_NAME")
    
    # File Upload Configuration
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
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
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = "HS256"
    
    # CORS Configuration
    cors_origins: list = Field(default=["http://localhost:3000", "http://localhost:8501"], env="CORS_ORIGINS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    streamlit_host: str = Field(default="localhost", env="STREAMLIT_HOST")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
