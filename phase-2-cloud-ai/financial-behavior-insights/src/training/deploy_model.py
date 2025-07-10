import os
import uuid
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Azure dependencies
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure ML dependencies not available. Deployment will be skipped.")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def validate_environment():
    """Validate required environment variables and dependencies."""
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure ML dependencies not available. Install with: pip install azure-ai-ml azure-identity")
    
    required_vars = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("Environment validation passed")

def validate_model_file(model_path: str):
    """Validate that the model file exists and is valid."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try to load the model to validate it
    try:
        import joblib
        model = joblib.load(model_path)
        logger.info(f"Model validation passed: {type(model).__name__}")
        return True
    except Exception as e:
        raise ValueError(f"Model file is invalid: {e}")

def deploy_model(
    model_path: str = "outputs/model.joblib",
    model_name: str = "financial-behavior-insights-model",
    model_version: int = None,
    endpoint_name: str = None,
    instance_type: str = "Standard_F2s_v2"
):
    """
    Deploy the trained model to Azure ML.
    
    Args:
        model_path: Path to the trained model file
        model_name: Name for the model in Azure ML
        model_version: Model version (None for latest)
        endpoint_name: Endpoint name (None for auto-generated)
        instance_type: Azure VM instance type
    """
    try:
        logger.info("Running deploy_model function")
        # Validate environment
        validate_environment()
        
        # Validate model file
        validate_model_file(model_path)
        
        # Generate endpoint name if not provided
        if not endpoint_name:
            base_name = "fbinsight"
            endpoint_name = f"{base_name}-ep-{uuid.uuid4().hex[:6]}"
        
        # Get Azure ML workspace details
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
        
        # Use built-in AzureML sklearn environment
        sklearn_env = "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1"
        
        # Authenticate and create MLClient
        logger.info("Authenticating with Azure ML...")
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        
        # Create the managed online endpoint
        logger.info(f"Creating endpoint: {endpoint_name}")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=f"{model_name} scoring endpoint",
            auth_mode="key"
        )
        
        ml_client.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint created: {endpoint_name}")
        
        # Get model version
        if not model_version:
            try:
                model = ml_client.models.get(name=model_name, label="latest")
                model_version = model.version
                logger.info(f"Using latest model version: {model_version}")
            except Exception as e:
                logger.warning(f"Could not get latest model version: {e}")
                logger.info("You may need to register the model first using --register-model")
                return
        else:
            model = f"{model_name}:{model_version}"
        
        # Define and create deployment
        logger.info("Creating deployment...")
        deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=model,
            environment=sklearn_env,
            instance_type=instance_type,
            instance_count=1,
            app_insights_enabled=True
        )
        
        ml_client.begin_create_or_update(deployment).result()
        logger.info("Deployment created successfully")
        
        # Allocate traffic to the deployment
        endpoint.traffic = {"blue": 100}
        ml_client.begin_create_or_update(endpoint).result()
        logger.info("Traffic allocated to deployment")
        
        # Retrieve and print scoring URI and key
        endpoint_obj = ml_client.online_endpoints.get(name=endpoint_name)
        keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
        
        logger.info("=== DEPLOYMENT SUCCESSFUL ===")
        logger.info(f"Endpoint: {endpoint_name}")
        logger.info(f"Scoring URI: {endpoint_obj.scoring_uri}")
        logger.info(f"Primary Key: {keys.primary_key}")
        logger.info("=== TESTING INSTRUCTIONS ===")
        logger.info("Test with curl:")
        logger.info(f'curl -X POST "{endpoint_obj.scoring_uri}" \\')
        logger.info(f'  -H "Authorization: Bearer {keys.primary_key}" \\')
        logger.info(f'  -H "Content-Type: application/json" \\')
        logger.info(f'  -d \'{{"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]}}\'')
        
        return {
            "endpoint_name": endpoint_name,
            "scoring_uri": endpoint_obj.scoring_uri,
            "primary_key": keys.primary_key
        }
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy model to Azure ML")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/model.joblib",
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="financial-behavior-insights-model",
        help="Name for the model in Azure ML"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help="Endpoint name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="Standard_F2s_v2",
        help="Azure VM instance type"
    )
    
    args = parser.parse_args()
    
    try:
        deploy_model(
            model_path=args.model_path,
            model_name=args.model_name,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type
        )
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())