"""
deploy_model.py

Handles the deployment of the financial behavior model to various serving environments.
Supports both local serving and cloud deployment (Azure ML).

Usage:
    python deploy_model.py --model-uri <model_uri> [--deployment-type local|azure] [--endpoint-name <name>]

MLOps Best Practices:
    - Model validation before deployment
    - Health checks
    - Proper error handling
    - Configuration management
    - Deployment rollback support
"""

import argparse
import os
import sys
import logging
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Try importing Azure ML dependencies
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        Model,
        Environment,
        CodeConfiguration
    )
    from azure.identity import DefaultAzureCredential
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    logger.warning("Azure ML SDK not available. Azure deployment will not be possible.")

from dotenv import load_dotenv
load_dotenv()


def validate_model(model_uri: str) -> bool:
    """
    Validate that the model can be loaded and used for predictions.
    
    Args:
        model_uri: URI of the MLflow model
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        logger.info(f"Validating model from {model_uri}")
        
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Create test input with correct feature names
        feature_names = [
            'Age', 'Transaction Amount', 'Account Balance', 'AccountAgeDays',
            'TransactionHour', 'TransactionDayOfWeek', 'Transaction Type_Deposit',
            'Transaction Type_Transfer', 'Transaction Type_Withdrawal',
            'Gender_Female', 'Gender_Male', 'Gender_Other'
        ]
        test_input = pd.DataFrame({
            name: [0.5] for name in feature_names
        })
        
        # Test prediction
        predictions = model.predict(test_input)
        
        # Validate output
        if predictions is None:
            logger.error("Model returned None predictions")
            return False
            
        if isinstance(predictions, np.ndarray):
            if len(predictions) != len(test_input):
                logger.error(f"Prediction count mismatch: expected {len(test_input)}, got {len(predictions)}")
                return False
        
        logger.info("Model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


def serve_locally(model_uri: str, port: int = 5000) -> None:
    """
    Serve the model locally using MLflow's built-in server.
    
    Args:
        model_uri: URI of the MLflow model
        port: Port to serve on
    """
    logger.info(f"Starting local model server on port {port}")
    
    try:
        # Start MLflow model serving
        os.system(f"mlflow models serve -m {model_uri} -p {port} --no-conda")
        
    except KeyboardInterrupt:
        logger.info("Stopping local server")
    except Exception as e:
        logger.error(f"Failed to start local server: {e}")
        raise


def deploy_to_azure(
    model_uri: str,
    endpoint_name: str,
    deployment_name: str = "blue",
    instance_type: str = "Standard_F2s_v2",
    instance_count: int = 1
) -> None:
    """
    Deploy model to Azure ML managed online endpoint.
    
    Args:
        model_uri: URI of the MLflow model
        endpoint_name: Name of the endpoint
        deployment_name: Name of the deployment
        instance_type: Azure VM instance type
        instance_count: Number of instances
    """
    if not AZURE_ML_AVAILABLE:
        raise ImportError("Azure ML SDK is not installed. Run: pip install azure-ai-ml")
    
    # Load Azure configuration
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Missing required Azure ML environment variables")
    
    logger.info(f"Deploying to Azure ML endpoint: {endpoint_name}")
    
    try:
        # Initialize ML client
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id,
            resource_group,
            workspace_name
        )
        
        # Create or update endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Financial behavior prediction endpoint",
            auth_mode="key"
        )
        
        logger.info("Creating/updating endpoint...")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        # Handle different model URI formats and register in Azure ML
        if model_uri.startswith("runs:/"):
            # Extract run ID and artifact path
            parts = model_uri.split("/")
            run_id = parts[1]
            artifact_path = "/".join(parts[2:])
            
            # Download model artifacts
            import mlflow.artifacts
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path="./model_artifacts"
            )
            
            # Register model
            model = Model(
                path="./model_artifacts",
                name="financial-behavior-model",
                description="Random Forest model for financial behavior prediction",
                type="mlflow_model"
            )
            registered_model = ml_client.models.create_or_update(model)
            model_name = registered_model.name
            model_version = registered_model.version
            
        elif model_uri.startswith("models:/"):
            # This is a model registry URI - need to download and register in Azure ML
            logger.info("Downloading model from MLflow registry...")
            
            # Parse model URI to get model name and version/stage
            model_uri_clean = model_uri.replace("models:/", "")
            if "@" in model_uri_clean:
                # Format: models:/model_name@stage
                mlflow_model_name, stage = model_uri_clean.split("@")
                # Get the latest version for this stage
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                model_versions = client.search_model_versions(f"name='{mlflow_model_name}'")
                # Filter by stage manually
                stage_versions = [v for v in model_versions if v.current_stage == stage]
                if not stage_versions:
                    raise ValueError(f"No model versions found for {mlflow_model_name} in stage {stage}")
                model_version_obj = stage_versions[0]  # Get the first (latest) version
                mlflow_model_version = str(model_version_obj.version)
            elif "/" in model_uri_clean:
                # Format: models:/model_name/version
                mlflow_model_name, mlflow_model_version = model_uri_clean.split("/")
            else:
                # Format: models:/model_name (use latest)
                mlflow_model_name = model_uri_clean
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                model_versions = client.search_model_versions(f"name='{mlflow_model_name}'")
                if not model_versions:
                    raise ValueError(f"No model versions found for {mlflow_model_name}")
                model_version_obj = model_versions[0]  # Get the first (latest) version
                mlflow_model_version = str(model_version_obj.version)
            
            # Get the model URI for downloading
            import mlflow
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Download model artifacts to temp directory
                logger.info(f"Downloading model {mlflow_model_name}:{mlflow_model_version}...")
                
                # Use the correct MLflow client to download the model
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                model_version = client.get_model_version(mlflow_model_name, mlflow_model_version)
                
                # Download the model artifacts
                mlflow.artifacts.download_artifacts(
                    artifact_uri=model_version.source,
                    dst_path=temp_dir
                )
                
                # Register model in Azure ML
                # Ensure we're pointing to the model directory, not the temp directory root
                model_dir = os.path.join(temp_dir, "model")
                if not os.path.exists(model_dir):
                    # If model directory doesn't exist, use temp_dir
                    model_dir = temp_dir
                
                azure_model = Model(
                    path=model_dir,
                    name="financial-behavior-model",
                    description="Random Forest model for financial behavior prediction",
                    type="mlflow_model"
                )
                registered_model = ml_client.models.create_or_update(azure_model)
                model_name = registered_model.name
                model_version = registered_model.version
                
                logger.info(f"Model registered in Azure ML: {model_name}:{model_version}")
                
            finally:
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        else:
            # Assume it's already registered in Azure ML format
            model_parts = model_uri.split(":")
            model_name = model_parts[1].split("/")[0]
            model_version = model_parts[1].split("/")[1]
        
        # Create deployment
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=f"{model_name}:{model_version}",
            instance_type=instance_type,
            instance_count=instance_count
        )
        
        logger.info(f"Creating deployment {deployment_name}...")
        ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        # Update traffic to new deployment
        endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"Model deployed successfully to {endpoint_name}")
        
        # Get endpoint details
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        logger.info(f"Endpoint URL: {endpoint.scoring_uri}")
        
    except Exception as e:
        logger.error(f"Azure deployment failed: {e}")
        raise


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy financial behavior model")
    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="URI of the MLflow model (e.g., runs:/RUN_ID/model)"
    )
    parser.add_argument(
        "--deployment-type",
        type=str,
        choices=["local", "azure"],
        default="local",
        help="Type of deployment"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="financial-behavior-endpoint",
        help="Name of the endpoint (for Azure deployment)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for local serving"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model validation"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate model unless skipped
        if not args.skip_validation:
            if not validate_model(args.model_uri):
                logger.error("Model validation failed. Aborting deployment.")
                sys.exit(1)
        
        # Deploy based on type
        if args.deployment_type == "local":
            serve_locally(args.model_uri, args.port)
        elif args.deployment_type == "azure":
            deploy_to_azure(args.model_uri, args.endpoint_name)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 