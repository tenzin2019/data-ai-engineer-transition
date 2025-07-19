"""
Configuration Management for Financial Behavior Insights

This module provides centralized configuration management following MLOps best practices.
All configuration values are centralized here for easy maintenance and deployment.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AzureConfig:
    """Azure ML configuration settings."""
    subscription_id: str
    resource_group: str
    workspace_name: str
    
    @classmethod
    def from_env(cls) -> 'AzureConfig':
        """Create AzureConfig from environment variables."""
        return cls(
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
            resource_group=os.getenv("AZURE_RESOURCE_GROUP", ""),
            workspace_name=os.getenv("AZURE_WORKSPACE_NAME", "")
        )
    
    def validate(self) -> bool:
        """Validate that all required Azure configuration is present."""
        return all([
            self.subscription_id,
            self.resource_group,
            self.workspace_name
        ])

@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str = "financial-behavior-model-fixed"
    endpoint_name: str = "fin-behavior-ep-fixed"
    deployment_name: str = "blue"
    version: Optional[str] = None
    
    # Model training parameters
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 3
    n_iter: int = 20
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    
    # Compatibility settings
    sklearn_version: str = "1.1.3"
    target_column: str = "HighAmount"

@dataclass
class DeploymentConfig:
    """Deployment configuration settings."""
    instance_type: str = "Standard_F4s_v2"
    instance_count: int = 1
    environment: str = "azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest"
    
    # Scaling settings
    min_instances: int = 0
    max_instances: int = 10
    
    # Traffic settings
    traffic_allocation: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.traffic_allocation is None:
            self.traffic_allocation = {"blue": 100}

@dataclass
class DataConfig:
    """Data configuration settings."""
    raw_data_path: str = "data/Banking-Dataset/Comprehensive_Banking_Database.csv"
    processed_data_path: str = "data/processed/Comprehensive_Banking_Database_processed.csv"
    output_dir: str = "outputs"
    
    # Data processing settings
    chunk_size: Optional[int] = None
    encoding: str = "utf-8"
    
    # Feature engineering settings
    categorical_columns: Optional[list] = None
    numerical_columns: Optional[list] = None
    
    def __post_init__(self):
        if self.categorical_columns is None:
            self.categorical_columns = ["Transaction Type", "Gender"]
        if self.numerical_columns is None:
            self.numerical_columns = ["Age", "Transaction Amount", "Account Balance", "AccountAgeDays"]

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"
    file_path: str = "workflow.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def setup_logging(self, logger_name: Optional[str] = None) -> logging.Logger:
        """Setup logging configuration."""
        import logging.handlers
        
        # Create formatter
        formatter = logging.Formatter(self.format)
        
        # Create handlers
        file_handler = logging.handlers.RotatingFileHandler(
            self.file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        logger = logging.getLogger(logger_name or __name__)
        logger.setLevel(getattr(logging, self.level.upper()))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

@dataclass
class TestingConfig:
    """Testing configuration settings."""
    test_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    
    # Test data settings
    test_samples: int = 5
    test_data_seed: int = 42

class Config:
    """Main configuration class that aggregates all configuration sections."""
    
    def __init__(self):
        """Initialize configuration with all sections."""
        self.azure = AzureConfig.from_env()
        self.model = ModelConfig()
        self.deployment = DeploymentConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.testing = TestingConfig()
        
        # Setup logging
        self.logger = self.logging.setup_logging("financial_behavior_insights")
    
    def validate(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Validate Azure configuration
            if not self.azure.validate():
                self.logger.error("❌ Azure configuration validation failed")
                return False
            
            # Validate data paths
            if not os.path.exists(self.data.raw_data_path):
                self.logger.warning(f"⚠️ Raw data path not found: {self.data.raw_data_path}")
            
            # Validate output directory
            os.makedirs(self.data.output_dir, exist_ok=True)
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_model_path(self, version: Optional[str] = None) -> str:
        """Get the path to the model file."""
        if version:
            return os.path.join(self.data.output_dir, f"model_v{version}.joblib")
        return os.path.join(self.data.output_dir, "model_compatible.joblib")
    
    def get_model_info_path(self) -> str:
        """Get the path to the model info file."""
        return os.path.join(self.data.output_dir, "model_info.json")
    
    def get_endpoint_url(self) -> str:
        """Get the endpoint URL."""
        return f"https://{self.model.endpoint_name}.australiaeast.inference.ml.azure.com/score"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "azure": {
                "subscription_id": self.azure.subscription_id,
                "resource_group": self.azure.resource_group,
                "workspace_name": self.azure.workspace_name
            },
            "model": {
                "name": self.model.name,
                "endpoint_name": self.model.endpoint_name,
                "deployment_name": self.model.deployment_name,
                "random_state": self.model.random_state,
                "test_size": self.model.test_size,
                "cv_folds": self.model.cv_folds,
                "sklearn_version": self.model.sklearn_version
            },
            "deployment": {
                "instance_type": self.deployment.instance_type,
                "instance_count": self.deployment.instance_count,
                "environment": self.deployment.environment,
                "traffic_allocation": self.deployment.traffic_allocation
            },
            "data": {
                "raw_data_path": self.data.raw_data_path,
                "processed_data_path": self.data.processed_data_path,
                "output_dir": self.data.output_dir
            }
        }
    
    def save_config(self, file_path: str = "config.json"):
        """Save configuration to file."""
        import json
        
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"✅ Configuration saved to {file_path}")
    
    @classmethod
    def load_config(cls, file_path: str = "config.json") -> 'Config':
        """Load configuration from file."""
        import json
        
        config = cls()
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration from file
            if "azure" in config_dict:
                config.azure.subscription_id = config_dict["azure"].get("subscription_id", "")
                config.azure.resource_group = config_dict["azure"].get("resource_group", "")
                config.azure.workspace_name = config_dict["azure"].get("workspace_name", "")
            
            config.logger.info(f"✅ Configuration loaded from {file_path}")
        else:
            config.logger.warning(f"⚠️ Configuration file not found: {file_path}")
        
        return config

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def validate_config() -> bool:
    """Validate the global configuration."""
    return config.validate()

def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Setup logging with the global configuration."""
    return config.logging.setup_logging(name)

# Export commonly used configuration values
AZURE_CONFIG = config.azure
MODEL_CONFIG = config.model
DEPLOYMENT_CONFIG = config.deployment
DATA_CONFIG = config.data
LOGGING_CONFIG = config.logging
TESTING_CONFIG = config.testing 