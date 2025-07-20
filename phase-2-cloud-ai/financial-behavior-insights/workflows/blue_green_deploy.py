#!/usr/bin/env python3
"""
blue_green_deploy.py

Advanced blue-green deployment script for Azure ML managed endpoints.
Implements MLOps best practices for safe production deployments.

Features:
- Blue-green deployment strategy
- Health checks and smoke tests
- Automatic traffic routing
- Rollback capabilities
- Comprehensive logging and monitoring

Usage:
    python blue_green_deploy.py --model-uri <uri> --endpoint-name <name>

Author: Data AI Engineer
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Azure ML imports
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        ManagedOnlineEndpoint,
        ManagedOnlineDeployment,
        Model
    )
    from azure.identity import DefaultAzureCredential
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    logger.error("Azure ML SDK not available. Please install: pip install azure-ai-ml")

from dotenv import load_dotenv
load_dotenv()


class BlueGreenDeployer:
    """Blue-green deployment manager for Azure ML."""
    
    def __init__(self, endpoint_name: str, model_uri: str):
        if not AZURE_ML_AVAILABLE:
            raise ImportError("Azure ML SDK is required for deployment")
        
        self.endpoint_name = endpoint_name
        self.model_uri = model_uri
        
        # Azure configuration
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
        
        if not all([self.subscription_id, self.resource_group, self.workspace_name]):
            raise ValueError("Missing required Azure ML environment variables")
        
        # Initialize ML client
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            self.subscription_id,
            self.resource_group,
            self.workspace_name
        )
        
        logger.info(f"Initialized deployer for endpoint: {endpoint_name}")
    
    def get_endpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get current endpoint information."""
        try:
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            return {
                "name": endpoint.name,
                "scoring_uri": endpoint.scoring_uri,
                "traffic": endpoint.traffic or {},
                "provisioning_state": endpoint.provisioning_state
            }
        except Exception as e:
            logger.warning(f"Could not get endpoint info: {e}")
            return None
    
    def get_current_deployments(self) -> List[str]:
        """Get list of current deployments."""
        try:
            deployments = self.ml_client.online_deployments.list(self.endpoint_name)
            return [dep.name for dep in deployments]
        except Exception as e:
            logger.warning(f"Could not list deployments: {e}")
            return []
    
    def determine_deployment_color(self) -> str:
        """Determine which color (blue/green) to use for new deployment."""
        current_deployments = self.get_current_deployments()
        endpoint_info = self.get_endpoint_info()
        
        if not current_deployments:
            logger.info("No existing deployments, using blue")
            return "blue"
        
        # Check current traffic distribution
        current_traffic = endpoint_info.get("traffic", {}) if endpoint_info else {}
        
        if "blue" in current_traffic and current_traffic["blue"] > 0:
            logger.info("Blue is currently serving traffic, deploying to green")
            return "green"
        elif "green" in current_traffic and current_traffic["green"] > 0:
            logger.info("Green is currently serving traffic, deploying to blue")
            return "blue"
        else:
            # Default to blue if no traffic is assigned
            logger.info("No traffic assigned, using blue")
            return "blue"
    
    def register_model(self) -> str:
        """Register the model in Azure ML if needed."""
        logger.info("Registering model in Azure ML...")
        
        # Extract model name from URI
        if self.model_uri.startswith("models:/"):
            # Already registered in MLflow, need to register in Azure ML
            model_name = "financial-behavior-model-azure"
        else:
            model_name = "financial-behavior-model-azure"
        
        try:
            # For this example, assume the model is in local artifacts
            model_path = "outputs/model.joblib"
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = Model(
                path=model_path,
                name=model_name,
                description="Financial behavior prediction model",
                type="custom_model"
            )
            
            registered_model = self.ml_client.models.create_or_update(model)
            logger.info(f"Model registered: {registered_model.name} v{registered_model.version}")
            
            return f"{registered_model.name}:{registered_model.version}"
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise
    
    def create_or_update_endpoint(self):
        """Create or update the managed endpoint."""
        logger.info(f"Creating/updating endpoint: {self.endpoint_name}")
        
        try:
            endpoint = ManagedOnlineEndpoint(
                name=self.endpoint_name,
                description="Financial behavior prediction endpoint with blue-green deployment",
                auth_mode="key"
            )
            
            # This will create if not exists, or update if exists
            result = self.ml_client.online_endpoints.begin_create_or_update(endpoint)
            logger.info("Waiting for endpoint creation/update...")
            result.result()
            
            logger.info("Endpoint ready")
            
        except Exception as e:
            logger.error(f"Endpoint creation failed: {e}")
            raise
    
    def deploy_new_version(self, deployment_color: str, model_ref: str) -> bool:
        """Deploy new model version to specified color."""
        logger.info(f"Deploying to {deployment_color} deployment...")
        
        try:
            deployment = ManagedOnlineDeployment(
                name=deployment_color,
                endpoint_name=self.endpoint_name,
                model=model_ref,
                instance_type="Standard_F2s_v2",
                instance_count=1,
                environment_variables={
                    "DEPLOYMENT_COLOR": deployment_color,
                    "DEPLOYMENT_TIMESTAMP": datetime.now().isoformat()
                }
            )
            
            # Deploy
            result = self.ml_client.online_deployments.begin_create_or_update(deployment)
            logger.info("Waiting for deployment...")
            result.result()
            
            logger.info(f"{deployment_color} deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"Deployment to {deployment_color} failed: {e}")
            return False
    
    def run_health_checks(self, deployment_color: str) -> bool:
        """Run health checks on the new deployment."""
        logger.info(f"Running health checks on {deployment_color} deployment...")
        
        try:
            # Get endpoint details
            endpoint_info = self.get_endpoint_info()
            if not endpoint_info:
                logger.error("Could not get endpoint information")
                return False
            
            scoring_uri = endpoint_info["scoring_uri"]
            
            # Get endpoint keys
            keys = self.ml_client.online_endpoints.get_keys(self.endpoint_name)
            primary_key = keys.primary_key
            
            # Test with sample data
            test_data = {
                "data": [
                    [25, 1000.0, 5000.0, 365, 14, 1, 0, 0, 1, 1, 0, 0]  # Sample features
                ]
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {primary_key}"
            }
            
            # Make test request (this would go to current traffic, not specific deployment)
            # For deployment-specific testing, you'd need to use deployment-specific endpoints
            logger.info("Testing endpoint response...")
            
            # Note: This is a simplified health check
            # In practice, you'd want more comprehensive testing
            response = requests.post(
                scoring_uri,
                json=test_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Health check passed. Sample prediction: {result}")
                return True
            else:
                logger.error(f"Health check failed. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def route_traffic(self, deployment_color: str, percentage: int = 100) -> bool:
        """Route traffic to specified deployment."""
        logger.info(f"Routing {percentage}% traffic to {deployment_color}")
        
        try:
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            
            # Update traffic allocation
            if percentage == 100:
                # Route all traffic to new deployment
                endpoint.traffic = {deployment_color: 100}
            else:
                # Gradual traffic shift (for canary deployments)
                other_color = "green" if deployment_color == "blue" else "blue"
                endpoint.traffic = {
                    deployment_color: percentage,
                    other_color: 100 - percentage
                }
            
            result = self.ml_client.online_endpoints.begin_create_or_update(endpoint)
            result.result()
            
            logger.info(f"Traffic routing updated: {endpoint.traffic}")
            return True
            
        except Exception as e:
            logger.error(f"Traffic routing failed: {e}")
            return False
    
    def cleanup_old_deployment(self, deployment_color: str) -> bool:
        """Remove old deployment that's no longer receiving traffic."""
        logger.info(f"Cleaning up old {deployment_color} deployment...")
        
        try:
            # Wait a bit before cleanup to ensure traffic has shifted
            time.sleep(30)
            
            self.ml_client.online_deployments.begin_delete(
                name=deployment_color,
                endpoint_name=self.endpoint_name
            ).result()
            
            logger.info(f"Old {deployment_color} deployment removed")
            return True
            
        except Exception as e:
            logger.warning(f"Cleanup of {deployment_color} failed: {e}")
            return False
    
    def rollback(self, target_deployment: str) -> bool:
        """Rollback to previous deployment."""
        logger.info(f"Rolling back to {target_deployment} deployment...")
        
        try:
            # Route all traffic back to target deployment
            return self.route_traffic(target_deployment, 100)
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def deploy(self, cleanup_old: bool = True) -> bool:
        """Execute complete blue-green deployment."""
        logger.info("Starting blue-green deployment...")
        
        try:
            # Step 1: Determine deployment color
            new_color = self.determine_deployment_color()
            old_color = "green" if new_color == "blue" else "blue"
            
            # Step 2: Register model
            model_ref = self.register_model()
            
            # Step 3: Create/update endpoint
            self.create_or_update_endpoint()
            
            # Step 4: Deploy to new color
            if not self.deploy_new_version(new_color, model_ref):
                logger.error("Deployment failed")
                return False
            
            # Step 5: Health checks
            if not self.run_health_checks(new_color):
                logger.error("Health checks failed")
                # Optionally rollback here
                return False
            
            # Step 6: Route traffic
            if not self.route_traffic(new_color, 100):
                logger.error("Traffic routing failed")
                return False
            
            # Step 7: Cleanup old deployment (optional)
            if cleanup_old and old_color in self.get_current_deployments():
                if not self.cleanup_old_deployment(old_color):
                    logger.warning("Old deployment cleanup failed, but deployment is successful")
            
            logger.info("✓ Blue-green deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Blue-Green Deployment for Azure ML")
    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="URI of the model to deploy"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="financial-behavior-endpoint",
        help="Name of the Azure ML endpoint"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup old deployment"
    )
    
    args = parser.parse_args()
    
    try:
        deployer = BlueGreenDeployer(args.endpoint_name, args.model_uri)
        success = deployer.deploy(cleanup_old=not args.no_cleanup)
        
        if success:
            logger.info("Deployment completed successfully!")
            sys.exit(0)
        else:
            logger.error("Deployment failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Deployment script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 