#!/usr/bin/env python3
"""
fix_model_format.py

Fix model format issues that cause "User container has crashed or terminated" error.
Converts joblib model to proper MLflow format with optimized conda environment.

Based on Azure ML troubleshooting best practices:
https://learn.microsoft.com/en-gb/azure/machine-learning/how-to-troubleshoot-online-endpoints
"""

import os
import sys
import logging
import joblib
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_optimized_conda_env() -> Dict[str, Any]:
    """
    Create optimized conda environment to prevent container crashes.
    
    Based on Azure ML troubleshooting best practices:
    - Minimal dependencies to reduce container startup time
    - Stable package versions to prevent conflicts
    - Use current MLflow version for compatibility
    """
    import mlflow
    
    return {
        "channels": ["conda-forge", "defaults"],
        "dependencies": [
            "python=3.11",  # Match current environment
            "scikit-learn=1.7.0",  # Match current environment
            "pandas=2.3.1",  # Match current environment
            "numpy=1.26.4",  # Match current environment
            "joblib=1.5.1",  # Match current environment
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",  # Use current MLflow version
                    # Removed unnecessary packages to minimize container startup time
                ]
            }
        ]
    }

def create_model_signature():
    """Create model signature for proper input validation."""
    # Import MLflow types
    from mlflow.types import Schema, ColSpec
    from mlflow.models.signature import ModelSignature
    
    # Define input schema based on the actual financial behavior model features
    input_schema = Schema([
        ColSpec("double", "Age"),
        ColSpec("double", "Transaction Amount"),
        ColSpec("double", "Account Balance"),
        ColSpec("double", "AccountAgeDays"),
        ColSpec("double", "TransactionHour"),
        ColSpec("double", "TransactionDayOfWeek"),
        ColSpec("double", "Transaction Type_Deposit"),
        ColSpec("double", "Transaction Type_Transfer"),
        ColSpec("double", "Transaction Type_Withdrawal"),
        ColSpec("double", "Gender_Female"),
        ColSpec("double", "Gender_Male"),
        ColSpec("double", "Gender_Other")
    ])
    
    # Define output schema
    output_schema = Schema([
        ColSpec("double", "prediction"),
        ColSpec("double", "probability")
    ])
    
    return ModelSignature(inputs=input_schema, outputs=output_schema)

def create_input_example() -> pd.DataFrame:
    """Create input example for model testing."""
    return pd.DataFrame({
        "Age": [35.0],
        "Transaction Amount": [100.0],
        "Account Balance": [5000.0],
        "AccountAgeDays": [365.0],
        "TransactionHour": [14.0],
        "TransactionDayOfWeek": [3.0],
        "Transaction Type_Deposit": [0.0],
        "Transaction Type_Transfer": [1.0],
        "Transaction Type_Withdrawal": [0.0],
        "Gender_Female": [0.0],
        "Gender_Male": [1.0],
        "Gender_Other": [0.0]
    })

def fix_model_format(model_path: str, output_dir: str = "mlflow_model") -> str:
    """
    Convert joblib model to proper MLflow format with optimized environment.
    
    Args:
        model_path: Path to the joblib model file
        output_dir: Directory to save the MLflow model
        
    Returns:
        Path to the created MLflow model
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Load the joblib model
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully: {type(model).__name__}")
    
    # Create optimized conda environment
    conda_env = create_optimized_conda_env()
    logger.info("Created optimized conda environment")
    
    # Create model signature
    signature = create_model_signature()
    logger.info("Created model signature")
    
    # Create input example
    input_example = create_input_example()
    logger.info("Created input example")
    
    # Set MLflow tracking URI to local
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Import MLflow sklearn
    from mlflow.sklearn import log_model, save_model
    
    # Log the model with MLflow
    with mlflow.start_run():
        log_model(
            sk_model=model,
            artifact_path="model",
            conda_env=conda_env,
            signature=signature,
            input_example=input_example,
            registered_model_name="financial-behavior-model-fixed"
        )
        logger.info("Model logged to MLflow successfully")
    
    # Save model locally in MLflow format
    save_model(
        sk_model=model,
        path=output_dir,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example
    )
    
    logger.info(f"Model saved in MLflow format to: {output_dir}")
    return output_dir

def verify_model_structure(model_dir: str) -> bool:
    """Verify that the MLflow model has the correct structure."""
    required_files = ["MLmodel", "conda.yaml", "python_env.yaml"]
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"Missing required file: {file}")
            return False
    
    # Check MLmodel file content
    mlmodel_path = os.path.join(model_dir, "MLmodel")
    with open(mlmodel_path, 'r') as f:
        content = f.read()
        if "model_path: model.pkl" not in content:
            logger.error("MLmodel file missing model_path")
            return False
    
    logger.info("✅ Model structure verification passed")
    return True

def main():
    """Main function to fix model format."""
    load_dotenv()
    
    # Check if model file exists
    model_path = "outputs/model.joblib"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please run 'make train' first to create the model")
        return False
    
    try:
        # Fix model format
        output_dir = fix_model_format(model_path)
        
        # Verify model structure
        if not verify_model_structure(output_dir):
            logger.error("Model structure verification failed")
            return False
        
        logger.info("✅ Model format fixed successfully!")
        logger.info(f"MLflow model saved to: {output_dir}")
        logger.info("You can now deploy using: make deploy")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to fix model format: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 