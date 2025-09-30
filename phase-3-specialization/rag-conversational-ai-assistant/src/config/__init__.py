"""
Configuration module for RAG Conversational AI Assistant.
Azure-optimized settings and configuration management.
"""

from .azure_settings import (
    AzureSettings,
    azure_settings,
    get_azure_settings,
    validate_azure_configuration
)

__all__ = [
    "AzureSettings",
    "azure_settings", 
    "get_azure_settings",
    "validate_azure_configuration"
]
