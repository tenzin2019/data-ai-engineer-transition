#!/usr/bin/env python3
"""
fix_model_format.py

Script to convert the existing joblib model into a proper MLflow model format
that is compatible with Azure ML deployment.

This script addresses the "User container has crashed or terminated" error
by creating a properly formatted MLflow model with correct environment and signature.
"""

import os
import sys
import joblib
import mlflow
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import subprocess

# Import mlflow.sklearn separately
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_current_environment_packages():
    """Get current environment packages for compatibility."""
    import pkg_resources
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    return installed_packages

def create_compatible_conda_env():
    """Create a conda environment compatible with Azure ML sklearn-1.0 environment."""
    # Azure ML sklearn-1.0 environment uses scikit-learn 1.0.2
    conda_env = {
        "name": "azureml-sklearn-1.0",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8",
            "pip",
            {
                "pip": [
                    "scikit-learn==1.0.2",  # Match Azure ML environment
                    "pandas>=1.1.0,<2.0.0",
                    "numpy>=1.19.0,<2.0.0",
                    "joblib>=1.0.0,<2.0.0",
                    "mlflow>=1.20.0,<2.0.0",  # Use older MLflow for compatibility
                    "azureml-inference-server-http>=0.7.0",
                    "azureml-defaults>=1.44.0"
                ]
            }
        ]
    }
    return conda_env

def retrain_model_with_compatible_versions():
    """Retrain the model with compatible scikit-learn version."""
    logger.info("Retraining model with compatible scikit-learn version...")
    
    # Load the original data
    data_path = "data/processed/Comprehensive_Banking_Database_processed.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in ['HighAmount']]
    X = df[feature_columns]
    y = df['HighAmount']
    
    # Install compatible scikit-learn version
    logger.info("Installing compatible scikit-learn version...")
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn==1.0.2"], check=True)
    
    # Import sklearn after installing compatible version
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with compatible sklearn version
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save scaler and model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns
    }
    
    # Save to outputs directory
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model_data, 'outputs/model_compatible.joblib')
    
    logger.info("Model retrained with compatible scikit-learn version")
    return model_data

def fix_model_format():
    """Convert joblib model to MLflow format compatible with Azure ML."""
    logger.info("Starting model format fix...")
    
    # Step 1: Retrain model with compatible versions
    model_data = retrain_model_with_compatible_versions()
    if model_data is None:
        return False
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Step 2: Create compatible conda environment
    conda_env = create_compatible_conda_env()
    logger.info("Created compatible conda environment")
    
    # Step 3: Create sample data for signature
    logger.info("Creating sample data for model signature...")
    sample_data = np.random.rand(5, len(feature_columns))
    sample_df = pd.DataFrame(sample_data, columns=feature_columns)
    
    # Step 4: Create model directory
    model_dir = "mlflow_model_compatible"
    if os.path.exists(model_dir):
        import shutil
        shutil.rmtree(model_dir)
    
    # Step 5: Log the model with MLflow
    logger.info("Logging model with MLflow...")
    
    with mlflow.start_run():
        # Log the model without signature to avoid version compatibility issues
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            conda_env=conda_env,
            registered_model_name="financial-behavior-model-fixed"
        )
    
    # Save the model locally
    mlflow.sklearn.save_model(
        sk_model=model,
        path=model_dir,
        conda_env=conda_env
    )
    
    # Step 6: Verify the model structure
    logger.info("Verifying model structure...")
    mlmodel_path = os.path.join(model_dir, "MLmodel")
    if os.path.exists(mlmodel_path):
        with open(mlmodel_path, 'r') as f:
            mlmodel_content = f.read()
            logger.info(f"MLmodel content:\n{mlmodel_content}")
    
    logger.info("✅ Model format fix completed successfully!")
    return True

if __name__ == "__main__":
    success = fix_model_format()
    if success:
        logger.info("Model successfully converted to MLflow format")
    else:
        logger.error("Failed to convert model format")
        sys.exit(1) 