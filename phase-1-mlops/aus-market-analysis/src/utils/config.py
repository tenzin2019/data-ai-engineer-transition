"""
Configuration management for the ASX market analysis project.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager for the application."""
    
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', 'asx-stock-prediction'),
            'prediction_days': int(os.getenv('PREDICTION_DAYS', '5')),
            'test_size': float(os.getenv('TEST_SIZE', '0.2')),
            'random_state': int(os.getenv('RANDOM_STATE', '42'))
        }
    
    @staticmethod
    def get_api_config() -> Dict[str, Any]:
        """Get API configuration."""
        return {
            'host': os.getenv('API_HOST', '0.0.0.0'),
            'port': int(os.getenv('API_PORT', '8000')),
            'debug': os.getenv('API_DEBUG', 'False').lower() == 'true'
        }
    
    @staticmethod
    def get_mlflow_config() -> Dict[str, Any]:
        """Get MLflow configuration."""
        return {
            'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            'registry_uri': os.getenv('MLFLOW_REGISTRY_URI', 'http://localhost:5000')
        }
    
    @staticmethod
    def get_logging_config() -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': os.getenv('LOG_FORMAT', 'json')
        } 